package cuda

import (
	"context"
	"encoding/binary"
	"errors"
	"net"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"golang.org/x/sys/unix"
)

type fakeVMMAgent struct {
	t             *testing.T
	process       PeerMappingProcess
	listener      *net.UnixListener
	records       []vmmPacket
	mu            sync.Mutex
	generation    uint64
	phase         uint32
	operations    []uint16
	importCount   int
	peerActive    bool
	failOperation uint16
	failAuditAt   int
	driftAuditAt  int
	auditCount    int
	hangOperation uint16
	dropOperation uint16
	badFreshID    bool
	unblock       chan struct{}
}

func (agent *fakeVMMAgent) setDriftAuditAt(count int) {
	agent.mu.Lock()
	defer agent.mu.Unlock()
	agent.driftAuditAt = count
}

func (agent *fakeVMMAgent) hasImporter() bool {
	for _, record := range agent.records {
		if record.Role == vmmRoleImporter {
			return true
		}
	}
	return false
}

func (agent *fakeVMMAgent) setFailOperation(operation uint16) {
	agent.mu.Lock()
	defer agent.mu.Unlock()
	agent.failOperation = operation
}

func (agent *fakeVMMAgent) setFailAuditAt(count int) {
	agent.mu.Lock()
	defer agent.mu.Unlock()
	agent.failAuditAt = count
}

func (agent *fakeVMMAgent) setHangOperation(operation uint16) {
	agent.mu.Lock()
	defer agent.mu.Unlock()
	agent.hangOperation = operation
}

func (agent *fakeVMMAgent) setBadFreshID(value bool) {
	agent.mu.Lock()
	defer agent.mu.Unlock()
	agent.badFreshID = value
}

func (agent *fakeVMMAgent) setState(
	generation uint64,
	phase uint32,
	peerActive bool,
) {
	agent.mu.Lock()
	defer agent.mu.Unlock()
	agent.generation = generation
	agent.phase = phase
	agent.peerActive = peerActive
}

func (agent *fakeVMMAgent) snapshot() (
	uint64,
	uint32,
	int,
	bool,
) {
	agent.mu.Lock()
	defer agent.mu.Unlock()
	return agent.generation, agent.phase, agent.importCount,
		agent.peerActive
}

func requireAgentsReset(
	t *testing.T,
	agents ...*fakeVMMAgent,
) {
	t.Helper()
	for _, agent := range agents {
		generation, phase, _, active := agent.snapshot()
		if generation != 0 || phase != vmmPhaseActive ||
			!active {
			t.Fatalf(
				"agent %d generation/phase/active = %d/%d/%t, want 0/ACTIVE/true",
				agent.process.NamespacePID,
				generation,
				phase,
				active,
			)
		}
	}
}

func startFakeVMMAgent(
	t *testing.T,
	namespacePID int,
	records ...vmmPacket,
) *fakeVMMAgent {
	t.Helper()
	path := filepath.Join(
		t.TempDir(), "cuda-vmm-"+time.Now().Format("150405.000000000"),
	)
	address := &net.UnixAddr{Name: path, Net: "unixpacket"}
	listener, err := net.ListenUnix("unixpacket", address)
	if err != nil {
		t.Fatal(err)
	}
	agent := &fakeVMMAgent{
		t: t,
		process: PeerMappingProcess{
			ObservedPID:  os.Getpid(),
			NamespacePID: namespacePID,
			UID:          uint32(os.Geteuid()),
			SocketPath:   path,
		},
		listener:   listener,
		records:    records,
		peerActive: true,
		unblock:    make(chan struct{}),
	}
	t.Cleanup(func() {
		_ = listener.Close()
		close(agent.unblock)
	})
	go agent.serve()
	return agent
}

func (agent *fakeVMMAgent) serve() {
	for {
		conn, err := agent.listener.AcceptUnix()
		if err != nil {
			return
		}
		go func() {
			defer conn.Close()
			request, receivedFD, err := readVMMPacket(conn)
			if err == nil {
				agent.handle(conn, request, receivedFD)
			}
			closeFD(receivedFD)
		}()
	}
}

func (agent *fakeVMMAgent) handle(
	conn *net.UnixConn,
	request vmmPacket,
	receivedFD int,
) {
	agent.mu.Lock()
	defer agent.mu.Unlock()
	agent.operations = append(agent.operations, request.Operation)
	if request.Operation == vmmOpAudit {
		agent.auditCount++
	}
	if request.Operation == agent.hangOperation {
		agent.mu.Unlock()
		<-agent.unblock
		agent.mu.Lock()
		return
	}
	response := newVMMPacket(request.Operation, agent.process)
	response.Generation = agent.generation
	response.Phase = agent.phase
	if request.Operation == agent.failOperation ||
		(request.Operation == vmmOpAudit &&
			agent.auditCount == agent.failAuditAt) {
		if request.Operation == agent.failOperation {
			agent.failOperation = 0
		}
		response.Status = -1
		copy(response.Message[:], "injected failure")
		setFakeVMMState(&response)
		_ = writeVMMPacket(conn, response, -1)
		return
	}
	if request.Operation != vmmOpAudit &&
		request.Generation != agent.generation &&
		!(request.Operation == vmmOpAbort &&
			agent.phase == vmmPhaseActive &&
			agent.generation == 0) {
		response.Status = -1
		copy(response.Message[:], "generation mismatch")
		setFakeVMMState(&response)
		_ = writeVMMPacket(conn, response, -1)
		return
	}
	responseFD := -1
	var exportFile *os.File
	switch request.Operation {
	case vmmOpAudit:
		if agent.phase == vmmPhaseActive &&
			agent.generation == 0 &&
			request.Generation != 0 {
			agent.generation = request.Generation
			agent.phase = vmmPhaseAudited
		}
		response.Count = uint32(len(agent.records))
	case vmmOpSyncAllContexts:
		if agent.phase == vmmPhaseAudited {
			agent.phase = vmmPhaseContextsSynced
		}
	case vmmOpDetachImports:
		if agent.phase == vmmPhaseContextsSynced {
			agent.phase = vmmPhaseLockReady
			if agent.hasImporter() {
				agent.peerActive = false
			}
		}
	case vmmOpVerifyLockReady:
	case vmmOpExport:
		exportFile, _ = os.CreateTemp("", "fresh-peer-fd")
		if exportFile == nil {
			response.Status = -1
			copy(response.Message[:], "fresh export failed")
			break
		}
		_ = os.Remove(exportFile.Name())
		var status unix.Stat_t
		if unix.Fstat(int(exportFile.Fd()), &status) != nil {
			response.Status = -1
			copy(response.Message[:], "fresh fstat failed")
			break
		}
		response.ObjectDev = request.ObjectDev
		response.ObjectIno = request.ObjectIno
		response.FreshDev = uint64(status.Dev)
		response.FreshIno = status.Ino
		if agent.badFreshID {
			response.FreshIno++
		}
		responseFD = int(exportFile.Fd())
	case vmmOpImport:
		var status unix.Stat_t
		if receivedFD < 0 ||
			unix.Fstat(receivedFD, &status) != nil ||
			uint64(status.Dev) != request.FreshDev ||
			status.Ino != request.FreshIno {
			response.Status = -1
			copy(response.Message[:], "fresh import FD mismatch")
		} else {
			agent.phase = vmmPhaseReattaching
			agent.peerActive = true
			agent.importCount++
		}
	case vmmOpVerifyActive:
		if agent.phase == vmmPhaseLockReady ||
			agent.phase == vmmPhaseReattaching {
			if agent.peerActive {
				agent.phase = vmmPhaseActiveVerified
			} else {
				response.Status = -1
				copy(response.Message[:], "peer mapping is detached")
			}
		}
	case vmmOpCommit:
		if agent.phase == vmmPhaseActiveVerified {
			agent.phase = vmmPhaseActive
		}
	case vmmOpAbort:
		if agent.phase == vmmPhaseActive &&
			agent.generation == 0 {
		} else if agent.phase <= vmmPhaseContextsSynced ||
			agent.phase == vmmPhaseActiveVerified {
			agent.phase = vmmPhaseActive
			agent.generation = 0
		} else {
			response.Status = -1
			copy(
				response.Message[:],
				"detached peer mapping cannot abort",
			)
		}
	default:
		response.Status = -1
		copy(response.Message[:], "unknown operation")
	}
	response.Generation = agent.generation
	response.Phase = agent.phase
	setFakeVMMState(&response)
	if request.Operation == agent.dropOperation {
		agent.dropOperation = 0
		if exportFile != nil {
			_ = exportFile.Close()
		}
		return
	}
	_ = writeVMMPacket(conn, response, responseFD)
	if exportFile != nil {
		_ = exportFile.Close()
	}
	if request.Operation == vmmOpAudit && response.Status == 0 {
		for _, record := range agent.records {
			record.Magic = vmmProtocolMagic
			record.Version = vmmProtocolVersion
			record.Operation = vmmOpAudit
			record.PID = uint32(agent.process.NamespacePID)
			record.Generation = agent.generation
			record.Phase = agent.phase
			if agent.auditCount == agent.driftAuditAt &&
				record.Role == vmmRoleImporter {
				record.MappingCount++
			}
			setFakeVMMState(&record)
			_ = writeVMMPacket(conn, record, -1)
		}
	}
}

func setFakeVMMState(packet *vmmPacket) {
	packet.Flags &^= vmmStateDetached | vmmStatePoisoned
	if packet.Phase == vmmPhaseLockReady ||
		packet.Phase == vmmPhaseReattaching {
		packet.Flags |= vmmStateDetached
	}
	if packet.Phase == vmmPhasePoisoned {
		packet.Flags |= vmmStatePoisoned
	}
}

func objectRecord(role uint32, dev, ino uint64) vmmPacket {
	return vmmPacket{
		Role: role, ObjectDev: dev, ObjectIno: ino, MappingCount: 1,
	}
}

func (agent *fakeVMMAgent) ops() []uint16 {
	agent.mu.Lock()
	defer agent.mu.Unlock()
	return append([]uint16(nil), agent.operations...)
}

func TestVMMPacketABIIsFixed(t *testing.T) {
	if size := binary.Size(vmmPacket{}); size != vmmPacketSize {
		t.Fatalf("packet size = %d, want %d", size, vmmPacketSize)
	}
}

func TestDetectVMMInterposeDefaultAndUniformOptIn(t *testing.T) {
	procRoot := t.TempDir()
	writeEnv := func(pid int, value string) {
		dir := filepath.Join(procRoot, string(rune('0'+pid)))
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(
			filepath.Join(dir, "environ"), []byte(value), 0o600,
		); err != nil {
			t.Fatal(err)
		}
	}
	writeEnv(1, "OTHER=1\x00")
	writeEnv(2, "OTHER=1\x00")
	enabled, err := DetectVMMInterpose(procRoot, []int{1, 2})
	if err != nil || enabled {
		t.Fatalf("default opt-in = %t, %v", enabled, err)
	}
	writeEnv(1, VMMInterposeEnv+"=1\x00"+VMMForcePOSIXEnv+"=1\x00")
	if _, err := DetectVMMInterpose(procRoot, []int{1, 2}); err == nil {
		t.Fatal("expected mixed registration failure")
	}
	writeEnv(2, VMMInterposeEnv+"=1\x00")
	if _, err := DetectVMMInterpose(procRoot, []int{1, 2}); err == nil {
		t.Fatal("expected missing force-POSIX failure")
	}
	writeEnv(2, VMMInterposeEnv+"=1\x00"+VMMForcePOSIXEnv+"=1\x00")
	enabled, err = DetectVMMInterpose(procRoot, []int{1, 2})
	if err != nil || !enabled {
		t.Fatalf("uniform opt-in = %t, %v", enabled, err)
	}
}

func listenUnixPacket(t *testing.T, path string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	listener, err := net.ListenUnix(
		"unixpacket",
		&net.UnixAddr{Name: path, Net: "unixpacket"},
	)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = listener.Close()
	})
}

func shortTempDir(t *testing.T) string {
	t.Helper()
	path, err := os.MkdirTemp("", "vmm-test-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = os.RemoveAll(path)
	})
	return path
}

func TestCheckpointPeerMappingProcessesSelectsLiveEndpoints(t *testing.T) {
	procRoot := shortTempDir(t)
	observedPIDs := []int{101, 102}
	namespacePIDs := []int{11, 12}
	for _, pid := range observedPIDs {
		if err := os.MkdirAll(
			filepath.Join(procRoot, strconv.Itoa(pid), "root"),
			0o755,
		); err != nil {
			t.Fatal(err)
		}
	}
	socketPath := filepath.Join(
		procRoot, "102", "root", "snapshot-control", "cuda-vmm-12.sock",
	)
	listenUnixPacket(t, socketPath)

	processes, err := CheckpointPeerMappingProcesses(
		procRoot, observedPIDs, namespacePIDs,
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(processes) != 1 {
		t.Fatalf("peer process count = %d, want 1", len(processes))
	}
	process := processes[0]
	if process.ObservedPID != 102 || process.NamespacePID != 12 ||
		process.UID != uint32(os.Geteuid()) ||
		process.SocketPath != socketPath {
		t.Fatalf("peer process = %+v", process)
	}
}

func TestRestorePeerMappingProcessesSelectsLiveEndpoints(t *testing.T) {
	procRoot := t.TempDir()
	controlMount := shortTempDir(t)
	restoredPIDs := []int{201, 202}
	for _, pid := range restoredPIDs {
		if err := os.MkdirAll(
			filepath.Join(procRoot, strconv.Itoa(pid)),
			0o755,
		); err != nil {
			t.Fatal(err)
		}
	}
	socketPath := filepath.Join(controlMount, "cuda-vmm-202.sock")
	listenUnixPacket(t, socketPath)

	processes, err := restorePeerMappingProcesses(
		procRoot, controlMount, restoredPIDs,
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(processes) != 1 {
		t.Fatalf("peer process count = %d, want 1", len(processes))
	}
	process := processes[0]
	if process.ObservedPID != 202 || process.NamespacePID != 202 ||
		process.UID != uint32(os.Geteuid()) ||
		process.SocketPath != socketPath {
		t.Fatalf("peer process = %+v", process)
	}
}

func TestPeerMappingProcessValidationIncludesEndpointlessPIDs(t *testing.T) {
	requireError := func(err error, want string) {
		t.Helper()
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Fatalf("error = %v, want %q", err, want)
		}
	}
	procRoot := t.TempDir()
	controlMount := t.TempDir()

	_, err := CheckpointPeerMappingProcesses(
		procRoot, []int{1, 1}, []int{10, 11},
	)
	requireError(err, "duplicate observed CUDA PID 1")
	_, err = CheckpointPeerMappingProcesses(
		procRoot, []int{1, 2}, []int{10, 10},
	)
	requireError(err, "duplicate namespace CUDA PID 10")
	_, err = CheckpointPeerMappingProcesses(
		procRoot, []int{1, 0}, []int{10, 11},
	)
	requireError(err, "invalid CUDA PID mapping 0:11")
	_, err = restorePeerMappingProcesses(
		procRoot, controlMount, []int{1, 1},
	)
	requireError(err, "duplicate restored CUDA PID 1")
	_, err = restorePeerMappingProcesses(
		procRoot, controlMount, []int{1, 0},
	)
	requireError(err, "invalid restored CUDA PID 0")
}

func TestCheckpointPeerMappingProcessesRejectsInvalidEndpoint(t *testing.T) {
	tests := []struct {
		name       string
		createPath func(string) error
		check      func(error) bool
	}{
		{
			name: "not a socket",
			createPath: func(path string) error {
				return os.WriteFile(path, nil, 0o600)
			},
			check: func(err error) bool {
				return strings.Contains(
					err.Error(), "endpoint is not a Unix socket",
				)
			},
		},
		{
			name: "inspection error",
			createPath: func(path string) error {
				return os.Symlink(filepath.Base(path), path)
			},
			check: func(err error) bool {
				return errors.Is(err, unix.ELOOP)
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			procRoot := t.TempDir()
			endpointDir := filepath.Join(
				procRoot, "101", "root", "snapshot-control",
			)
			if err := os.MkdirAll(endpointDir, 0o755); err != nil {
				t.Fatal(err)
			}
			if err := test.createPath(
				filepath.Join(endpointDir, "cuda-vmm-11.sock"),
			); err != nil {
				t.Fatal(err)
			}
			_, err := CheckpointPeerMappingProcesses(
				procRoot, []int{101}, []int{11},
			)
			if err == nil || !test.check(err) {
				t.Fatalf("unexpected error = %v", err)
			}
		})
	}
}

func TestPrepareAndRestorePeerMappingsOrderAndFreshFDBroker(t *testing.T) {
	const dev, ino = 7, 11
	owner := startFakeVMMAgent(
		t, 101, objectRecord(vmmRoleOwner, dev, ino),
	)
	importer := startFakeVMMAgent(
		t, 102, objectRecord(vmmRoleImporter, dev, ino),
	)
	processes := []PeerMappingProcess{owner.process, importer.process}

	if err := PreparePeerMappings(
		context.Background(), processes, 41, logr.Discard(),
	); err != nil {
		t.Fatalf("prepare: %v", err)
	}
	if err := RestorePeerMappings(
		context.Background(), processes, 41, logr.Discard(),
	); err != nil {
		t.Fatalf("restore: %v", err)
	}
	wantOwner := []uint16{
		vmmOpAudit,
		vmmOpSyncAllContexts,
		vmmOpDetachImports,
		vmmOpVerifyLockReady,
		vmmOpAudit,
		vmmOpAudit,
		vmmOpExport,
		vmmOpVerifyActive,
		vmmOpCommit,
	}
	if !slices.Equal(owner.ops(), wantOwner) {
		t.Fatalf("owner operations = %v, want %v", owner.ops(), wantOwner)
	}
	_, _, imports, _ := importer.snapshot()
	if imports != 1 {
		t.Fatalf("fresh import count = %d, want 1", imports)
	}
}

func TestVMMGraphAcceptsZeroDeviceIdentityComponent(t *testing.T) {
	owner := startFakeVMMAgent(
		t, 103, objectRecord(vmmRoleOwner, 0, 12),
	)
	importer := startFakeVMMAgent(
		t, 104, objectRecord(vmmRoleImporter, 0, 12),
	)
	if err := PreparePeerMappings(
		context.Background(),
		[]PeerMappingProcess{owner.process, importer.process},
		42,
		logr.Discard(),
	); err != nil {
		t.Fatalf("prepare zero-device identity: %v", err)
	}
}

func TestPrepareRejectsAmbiguousOwnerAndMissingRegistration(t *testing.T) {
	const dev, ino = 13, 17
	ownerA := startFakeVMMAgent(
		t, 201, objectRecord(vmmRoleOwner, dev, ino),
	)
	ownerB := startFakeVMMAgent(
		t, 202, objectRecord(vmmRoleOwner, dev, ino),
	)
	importer := startFakeVMMAgent(
		t, 203, objectRecord(vmmRoleImporter, dev, ino),
	)
	if err := PreparePeerMappings(
		context.Background(),
		[]PeerMappingProcess{
			ownerA.process, ownerB.process, importer.process,
		},
		1,
		logr.Discard(),
	); err == nil {
		t.Fatal("expected ambiguous owner failure")
	}
	requireAgentsReset(t, ownerA, ownerB, importer)
	missing := PeerMappingProcess{
		ObservedPID:  os.Getpid(),
		NamespacePID: 204,
		SocketPath:   filepath.Join(t.TempDir(), "missing.sock"),
	}
	if err := PreparePeerMappings(
		context.Background(),
		[]PeerMappingProcess{missing},
		1,
		logr.Discard(),
	); err == nil {
		t.Fatal("expected missing private shim endpoint")
	}
}

func TestPrepareRejectsEnabledPathWithoutManagedPeerMappings(t *testing.T) {
	agent := startFakeVMMAgent(t, 205)
	err := PreparePeerMappings(
		context.Background(),
		[]PeerMappingProcess{agent.process},
		1,
		logr.Discard(),
	)
	if err == nil ||
		!strings.Contains(
			err.Error(), "no managed CUDA POSIX peer mappings",
		) {
		t.Fatalf("empty managed graph error = %v", err)
	}
	requireAgentsReset(t, agent)
}

func TestPrepareFailureBestEffortRemapsDetachedPeers(t *testing.T) {
	const dev, ino = 19, 23
	owner := startFakeVMMAgent(
		t, 301, objectRecord(vmmRoleOwner, dev, ino),
	)
	importer := startFakeVMMAgent(
		t, 302, objectRecord(vmmRoleImporter, dev, ino),
	)
	importer.setFailOperation(vmmOpVerifyLockReady)
	err := PreparePeerMappings(
		context.Background(),
		[]PeerMappingProcess{owner.process, importer.process},
		2,
		logr.Discard(),
	)
	if err == nil {
		t.Fatal("expected prepare failure")
	}
	_, _, imports, active := importer.snapshot()
	if imports != 1 || !active {
		t.Fatalf(
			"best-effort recovery imports/active = %d/%t, want 1/true",
			imports,
			active,
		)
	}
	requireAgentsReset(t, owner, importer)
}

func TestPreparePartialContextSyncResetsAllAgents(t *testing.T) {
	const dev, ino = 20, 21
	owner := startFakeVMMAgent(
		t, 316, objectRecord(vmmRoleOwner, dev, ino),
	)
	importer := startFakeVMMAgent(
		t, 317, objectRecord(vmmRoleImporter, dev, ino),
	)
	importer.setFailOperation(vmmOpSyncAllContexts)
	err := PreparePeerMappings(
		context.Background(),
		[]PeerMappingProcess{owner.process, importer.process},
		13,
		logr.Discard(),
	)
	if err == nil {
		t.Fatal("expected partial context-sync failure")
	}
	_, _, imports, _ := importer.snapshot()
	if imports != 0 {
		t.Fatalf("context-sync recovery imports = %d, want 0", imports)
	}
	requireAgentsReset(t, owner, importer)
}

func TestPrepareRecoveryFailureIncludesOriginalAndRecovery(t *testing.T) {
	const dev, ino = 22, 23
	owner := startFakeVMMAgent(
		t, 318, objectRecord(vmmRoleOwner, dev, ino),
	)
	importer := startFakeVMMAgent(
		t, 319, objectRecord(vmmRoleImporter, dev, ino),
	)
	importer.setFailOperation(vmmOpVerifyLockReady)
	owner.setFailOperation(vmmOpExport)
	err := PreparePeerMappings(
		context.Background(),
		[]PeerMappingProcess{owner.process, importer.process},
		14,
		logr.Discard(),
	)
	if err == nil ||
		!strings.Contains(err.Error(), "verify lock-ready state") ||
		!strings.Contains(err.Error(), "best-effort peer remap") {
		t.Fatalf("combined prepare/recovery error = %v", err)
	}
	if generation, phase, _, active := importer.snapshot(); generation != 14 || phase != vmmPhaseLockReady || active {
		t.Fatalf(
			"failed recovery importer generation/phase/active = %d/%d/%t",
			generation,
			phase,
			active,
		)
	}
}

func TestPrepareReauditFailureBestEffortRemapsDetachedPeers(t *testing.T) {
	const dev, ino = 24, 25
	owner := startFakeVMMAgent(
		t, 303, objectRecord(vmmRoleOwner, dev, ino),
	)
	importer := startFakeVMMAgent(
		t, 304, objectRecord(vmmRoleImporter, dev, ino),
	)
	importer.setFailAuditAt(2)
	err := PreparePeerMappings(
		context.Background(),
		[]PeerMappingProcess{owner.process, importer.process},
		5,
		logr.Discard(),
	)
	if err == nil {
		t.Fatal("expected detached re-audit failure")
	}
	_, _, imports, active := importer.snapshot()
	if imports != 1 || !active {
		t.Fatalf(
			"best-effort recovery imports/active = %d/%t, want 1/true",
			imports,
			active,
		)
	}
}

func TestPreparePreservesInitialAuditFailureWithGenerationZero(
	t *testing.T,
) {
	agent := startFakeVMMAgent(t, 305)
	agent.setFailOperation(vmmOpAudit)
	err := PreparePeerMappings(
		context.Background(),
		[]PeerMappingProcess{agent.process},
		6,
		logr.Discard(),
	)
	if err == nil || !strings.Contains(err.Error(), "injected failure") {
		t.Fatalf("initial audit error = %v", err)
	}
	if strings.Contains(err.Error(), "generation mismatch") {
		t.Fatalf("initial audit failure was masked: %v", err)
	}
	requireAgentsReset(t, agent)
}

func TestPreparePartialInitialAuditResetsContactedAgents(t *testing.T) {
	const dev, ino = 26, 27
	owner := startFakeVMMAgent(
		t, 306, objectRecord(vmmRoleOwner, dev, ino),
	)
	importer := startFakeVMMAgent(
		t, 307, objectRecord(vmmRoleImporter, dev, ino),
	)
	importer.setFailAuditAt(1)
	processes := []PeerMappingProcess{owner.process, importer.process}
	err := PreparePeerMappings(
		context.Background(), processes, 7, logr.Discard(),
	)
	if err == nil {
		t.Fatal("expected partial initial audit failure")
	}
	requireAgentsReset(t, owner, importer)
	if err := PreparePeerMappings(
		context.Background(), processes, 8, logr.Discard(),
	); err != nil {
		t.Fatalf("next generation after partial audit: %v", err)
	}
	if err := AbortPeerMappings(processes, 8, logr.Discard()); err != nil {
		t.Fatalf("abort next generation: %v", err)
	}
	requireAgentsReset(t, owner, importer)
}

func TestPrepareMissingOwnerResetsEveryContactedAgent(t *testing.T) {
	importer := startFakeVMMAgent(
		t, 308, objectRecord(vmmRoleImporter, 31, 37),
	)
	err := PreparePeerMappings(
		context.Background(),
		[]PeerMappingProcess{importer.process},
		9,
		logr.Discard(),
	)
	if err == nil ||
		!strings.Contains(err.Error(), "has no owner") {
		t.Fatalf("missing owner error = %v", err)
	}
	requireAgentsReset(t, importer)
}

func TestPreparePartialDetachRemapsAndResetsAllAgents(t *testing.T) {
	const dev, ino = 41, 43
	owner := startFakeVMMAgent(
		t, 309, objectRecord(vmmRoleOwner, dev, ino),
	)
	importerA := startFakeVMMAgent(
		t, 310, objectRecord(vmmRoleImporter, dev, ino),
	)
	importerB := startFakeVMMAgent(
		t, 311, objectRecord(vmmRoleImporter, dev, ino),
	)
	importerB.setFailOperation(vmmOpDetachImports)
	err := PreparePeerMappings(
		context.Background(),
		[]PeerMappingProcess{
			owner.process, importerA.process, importerB.process,
		},
		10,
		logr.Discard(),
	)
	if err == nil {
		t.Fatal("expected partial detach failure")
	}
	_, _, importsA, _ := importerA.snapshot()
	_, _, importsB, _ := importerB.snapshot()
	if importsA != 1 || importsB != 0 {
		t.Fatalf(
			"recovery import counts = %d/%d, want 1/0",
			importsA,
			importsB,
		)
	}
	requireAgentsReset(t, owner, importerA, importerB)
}

func TestPrepareGraphDriftRemapsAndResetsAllAgents(t *testing.T) {
	const dev, ino = 47, 53
	owner := startFakeVMMAgent(
		t, 312, objectRecord(vmmRoleOwner, dev, ino),
	)
	importer := startFakeVMMAgent(
		t, 313, objectRecord(vmmRoleImporter, dev, ino),
	)
	importer.setDriftAuditAt(2)
	err := PreparePeerMappings(
		context.Background(),
		[]PeerMappingProcess{owner.process, importer.process},
		11,
		logr.Discard(),
	)
	if err == nil ||
		!strings.Contains(err.Error(), "resource graph drifted") {
		t.Fatalf("graph drift error = %v", err)
	}
	_, _, imports, _ := importer.snapshot()
	if imports != 1 {
		t.Fatalf("graph-drift recovery imports = %d, want 1", imports)
	}
	requireAgentsReset(t, owner, importer)
}

func TestAbortPeerMappingsRemapsPreparedGeneration(t *testing.T) {
	const dev, ino = 59, 61
	owner := startFakeVMMAgent(
		t, 314, objectRecord(vmmRoleOwner, dev, ino),
	)
	importer := startFakeVMMAgent(
		t, 315, objectRecord(vmmRoleImporter, dev, ino),
	)
	processes := []PeerMappingProcess{owner.process, importer.process}
	if err := PreparePeerMappings(
		context.Background(), processes, 12, logr.Discard(),
	); err != nil {
		t.Fatal(err)
	}
	_, phase, _, active := importer.snapshot()
	if phase != vmmPhaseLockReady || active {
		t.Fatalf(
			"prepared importer phase/active = %d/%t",
			phase,
			active,
		)
	}
	if err := AbortPeerMappings(
		processes, 12, logr.Discard(),
	); err != nil {
		t.Fatal(err)
	}
	requireAgentsReset(t, owner, importer)
}

func TestRestoreRejectsIncorrectFreshFDIdentity(t *testing.T) {
	const dev, ino = 29, 31
	owner := startFakeVMMAgent(
		t, 401, objectRecord(vmmRoleOwner, dev, ino),
	)
	importer := startFakeVMMAgent(
		t, 402, objectRecord(vmmRoleImporter, dev, ino),
	)
	processes := []PeerMappingProcess{owner.process, importer.process}
	if err := PreparePeerMappings(
		context.Background(), processes, 3, logr.Discard(),
	); err != nil {
		t.Fatal(err)
	}
	owner.setBadFreshID(true)
	if err := RestorePeerMappings(
		context.Background(), processes, 3, logr.Discard(),
	); err == nil {
		t.Fatal("expected fresh FD identity failure")
	}
	_, _, imports, _ := importer.snapshot()
	if imports != 0 {
		t.Fatalf("fresh imports = %d, want 0", imports)
	}
}

func TestPrepareTimeoutIsBounded(t *testing.T) {
	agent := startFakeVMMAgent(t, 501)
	agent.setHangOperation(vmmOpAudit)
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	err := PreparePeerMappings(
		ctx, []PeerMappingProcess{agent.process}, 4, logr.Discard(),
	)
	if err == nil {
		t.Fatal("expected bounded timeout")
	}
	var netErr net.Error
	if !errors.Is(err, context.DeadlineExceeded) &&
		!errors.As(err, &netErr) {
		t.Fatalf("timeout error = %v", err)
	}
}

func TestAbortPeerMappingsDoesNotStarveLaterHealthyAgents(t *testing.T) {
	const generation = 74
	const objectDev, objectIno = 79, 83
	unresponsive := startFakeVMMAgent(t, 502)
	unresponsive.setHangOperation(vmmOpAudit)
	owner := startFakeVMMAgent(
		t, 503,
		objectRecord(vmmRoleOwner, objectDev, objectIno),
	)
	importer := startFakeVMMAgent(
		t, 504,
		objectRecord(vmmRoleImporter, objectDev, objectIno),
	)
	owner.setState(generation, vmmPhaseLockReady, true)
	importer.setState(generation, vmmPhaseLockReady, false)

	err := abortPeerMappings(
		[]PeerMappingProcess{
			unresponsive.process,
			owner.process,
			importer.process,
		},
		generation,
		logr.Discard(),
		20*time.Millisecond,
	)
	if err == nil ||
		!strings.Contains(
			err.Error(), "namespace PID 502",
		) {
		t.Fatalf("combined unresponsive-agent error = %v", err)
	}
	requireAgentsReset(t, owner, importer)
	if _, _, imports, active := importer.snapshot(); imports != 1 ||
		!active {
		t.Fatalf(
			"healthy importer recovery imports/active = %d/%t, want 1/true",
			imports,
			active,
		)
	}
}

func TestWriteReadVMMPacketTransfersOneFD(t *testing.T) {
	left, right, err := socketPair()
	if err != nil {
		t.Fatal(err)
	}
	defer left.Close()
	defer right.Close()
	file, err := os.CreateTemp(t.TempDir(), "rights")
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	process := PeerMappingProcess{NamespacePID: 1}

	written := make(chan error, 1)
	go func() {
		written <- writeVMMPacket(
			left,
			newVMMPacket(vmmOpImport, process),
			int(file.Fd()),
		)
	}()
	packet, fd, err := readVMMPacket(right)
	if err != nil {
		t.Fatal(err)
	}
	defer closeFD(fd)
	if packet.Operation != vmmOpImport || fd < 0 {
		t.Fatalf("packet operation=%d fd=%d", packet.Operation, fd)
	}
	var original syscall.Stat_t
	var received syscall.Stat_t
	if syscall.Fstat(int(file.Fd()), &original) != nil ||
		syscall.Fstat(fd, &received) != nil {
		t.Fatal("fstat failed")
	}
	if original.Dev != received.Dev || original.Ino != received.Ino {
		t.Fatal("SCM_RIGHTS changed file identity")
	}
	if err := <-written; err != nil {
		t.Fatal(err)
	}
}

func TestReadVMMPacketRejectsMultipleRights(t *testing.T) {
	for _, count := range []int{2, 8} {
		t.Run(strconv.Itoa(count), func(t *testing.T) {
			left, right, err := socketPair()
			if err != nil {
				t.Fatal(err)
			}
			defer left.Close()
			defer right.Close()
			files := make([]*os.File, count)
			fds := make([]int, count)
			for index := range files {
				files[index], err = os.CreateTemp(t.TempDir(), "rights")
				if err != nil {
					t.Fatal(err)
				}
				defer files[index].Close()
				fds[index] = int(files[index].Fd())
			}
			buffer := make([]byte, vmmPacketSize)
			if err := binary.Write(
				newFixedWriter(buffer),
				binary.LittleEndian,
				newVMMPacket(
					vmmOpImport,
					PeerMappingProcess{NamespacePID: 1},
				),
			); err != nil {
				t.Fatal(err)
			}
			go func() {
				_, _, _ = left.WriteMsgUnix(
					buffer, unix.UnixRights(fds...), nil,
				)
			}()
			if _, fd, err := readVMMPacket(right); err == nil {
				closeFD(fd)
				t.Fatal("expected multiple SCM_RIGHTS rejection")
			}
		})
	}
}

func socketPair() (*net.UnixConn, *net.UnixConn, error) {
	fds, err := unix.Socketpair(
		unix.AF_UNIX,
		unix.SOCK_SEQPACKET|unix.SOCK_CLOEXEC,
		0,
	)
	if err != nil {
		return nil, nil, err
	}
	leftFile := os.NewFile(uintptr(fds[0]), "left")
	rightFile := os.NewFile(uintptr(fds[1]), "right")
	leftRaw, err := net.FileConn(leftFile)
	_ = leftFile.Close()
	if err != nil {
		_ = rightFile.Close()
		return nil, nil, err
	}
	rightRaw, err := net.FileConn(rightFile)
	_ = rightFile.Close()
	if err != nil {
		_ = leftRaw.Close()
		return nil, nil, err
	}
	return leftRaw.(*net.UnixConn), rightRaw.(*net.UnixConn), nil
}
