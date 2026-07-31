package cuda

import (
	"context"
	"crypto/rand"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/go-logr/logr"
	"golang.org/x/sys/unix"
)

const (
	VMMInterposeEnv  = "DYN_SNAPSHOT_CUDA_VMM_INTERPOSE"
	VMMForcePOSIXEnv = "DYN_SNAPSHOT_CUDA_VMM_FORCE_POSIX"

	vmmControlMount       = "/snapshot-control"
	vmmSocketPrefix       = "cuda-vmm-"
	vmmProtocolMagic      = 0x44564d4d
	vmmProtocolVersion    = 3
	vmmPacketSize         = 256
	vmmMessageSize        = 112
	vmmMaxObjectsPerAgent = 4096
	vmmControlTimeout     = 10 * time.Second
	vmmRecoveryTimeout    = 2 * time.Second

	vmmOpAudit           = 1
	vmmOpSyncAllContexts = 2
	vmmOpDetachImports   = 3
	vmmOpVerifyLockReady = 4
	vmmOpExport          = 5
	vmmOpImport          = 6
	vmmOpVerifyActive    = 7
	vmmOpCommit          = 8
	vmmOpAbort           = 9

	vmmRoleOwner    = 1
	vmmRoleImporter = 2

	vmmStateDetached = 1 << 16
	vmmStatePoisoned = 1 << 17

	vmmPhaseActive         = 0
	vmmPhaseAudited        = 1
	vmmPhaseContextsSynced = 2
	vmmPhaseLockReady      = 3
	vmmPhaseReattaching    = 4
	vmmPhaseActiveVerified = 5
	vmmPhasePoisoned       = 6
)

// PeerMappingProcess identifies one private in-process shim endpoint from both
// sides of its PID namespace.
type PeerMappingProcess struct {
	ObservedPID  int
	NamespacePID int
	UID          uint32
	SocketPath   string
}

type vmmPacket struct {
	Magic        uint32
	Version      uint16
	Operation    uint16
	Status       int32
	Flags        uint32
	Generation   uint64
	Revision     uint64
	ObjectDev    uint64
	ObjectIno    uint64
	FreshDev     uint64
	FreshIno     uint64
	Address      uint64
	Size         uint64
	Offset       uint64
	Logical      uint64
	PID          uint32
	Count        uint32
	Role         uint32
	MappingCount uint32
	Message      [vmmMessageSize]byte
	Phase        uint32
	Outcome      uint32
	Reserved     [24]byte
}

type vmmObjectID struct {
	dev uint64
	ino uint64
}

type vmmObject struct {
	owner     PeerMappingProcess
	importers []PeerMappingProcess
	mappings  map[int]uint32
}

type vmmAudit struct {
	process    PeerMappingProcess
	generation uint64
	phase      uint32
	records    []vmmPacket
}

// NewPeerMappingGeneration returns the nonzero generation Snapshot persists for
// one checkpoint/restore cycle.
func NewPeerMappingGeneration() (uint64, error) {
	var encoded [8]byte
	if _, err := rand.Read(encoded[:]); err != nil {
		return 0, fmt.Errorf("generate CUDA peer-mapping generation: %w", err)
	}
	generation := binary.LittleEndian.Uint64(encoded[:])
	if generation == 0 {
		generation = 1
	}
	return generation, nil
}

// ValidatePeerMappingProcessSet rejects CUDA process-tree drift.
func ValidatePeerMappingProcessSet(expected, current []int) error {
	expectedCopy := append([]int(nil), expected...)
	currentCopy := append([]int(nil), current...)
	slices.Sort(expectedCopy)
	slices.Sort(currentCopy)
	if !slices.Equal(expectedCopy, currentCopy) {
		return fmt.Errorf(
			"CUDA process set drift: expected=%v current=%v",
			expectedCopy, currentCopy,
		)
	}
	return nil
}

func (packet vmmPacket) message() string {
	return strings.TrimRight(string(packet.Message[:]), "\x00")
}

func (id vmmObjectID) String() string {
	return fmt.Sprintf("%d:%d", id.dev, id.ino)
}

// DetectVMMInterpose requires uniform opt-in across all CUDA processes.
func DetectVMMInterpose(procRoot string, pids []int) (bool, error) {
	enabled := 0
	forcePOSIX := 0
	for _, pid := range pids {
		content, err := os.ReadFile(
			filepath.Join(procRoot, strconv.Itoa(pid), "environ"),
		)
		if err != nil {
			return false, fmt.Errorf(
				"read CUDA process %d environment: %w", pid, err,
			)
		}
		for _, entry := range strings.Split(string(content), "\x00") {
			if entry == VMMInterposeEnv+"=1" {
				enabled++
			}
			if entry == VMMForcePOSIXEnv+"=1" {
				forcePOSIX++
			}
		}
	}
	if enabled != 0 && enabled != len(pids) {
		return false, fmt.Errorf(
			"CUDA VMM interposition enabled for %d of %d CUDA processes",
			enabled, len(pids),
		)
	}
	if enabled != 0 && forcePOSIX != len(pids) {
		return false, fmt.Errorf(
			"CUDA VMM force-POSIX enabled for %d of %d CUDA processes",
			forcePOSIX, len(pids),
		)
	}
	return enabled > 0, nil
}

// CheckpointPeerMappingProcesses resolves shim endpoints through host /proc.
func CheckpointPeerMappingProcesses(
	procRoot string,
	observedPIDs []int,
	namespacePIDs []int,
) ([]PeerMappingProcess, error) {
	if len(observedPIDs) != len(namespacePIDs) {
		return nil, fmt.Errorf(
			"CUDA PID mapping count mismatch: observed=%d namespace=%d",
			len(observedPIDs), len(namespacePIDs),
		)
	}
	processes := make([]PeerMappingProcess, len(observedPIDs))
	seenObserved := make(map[int]struct{}, len(observedPIDs))
	seenNamespace := make(map[int]struct{}, len(namespacePIDs))
	for index := range observedPIDs {
		observed := observedPIDs[index]
		namespace := namespacePIDs[index]
		if observed <= 0 || namespace <= 0 {
			return nil, fmt.Errorf(
				"invalid CUDA PID mapping %d:%d", observed, namespace,
			)
		}
		if _, duplicate := seenObserved[observed]; duplicate {
			return nil, fmt.Errorf("duplicate observed CUDA PID %d", observed)
		}
		if _, duplicate := seenNamespace[namespace]; duplicate {
			return nil, fmt.Errorf("duplicate namespace CUDA PID %d", namespace)
		}
		seenObserved[observed] = struct{}{}
		seenNamespace[namespace] = struct{}{}
		uid, err := processUID(
			filepath.Join(procRoot, strconv.Itoa(observed)),
		)
		if err != nil {
			return nil, fmt.Errorf(
				"inspect CUDA process %d credentials: %w",
				observed, err,
			)
		}
		processes[index] = PeerMappingProcess{
			ObservedPID:  observed,
			NamespacePID: namespace,
			UID:          uid,
			SocketPath: filepath.Join(
				procRoot,
				strconv.Itoa(observed),
				"root",
				strings.TrimPrefix(vmmControlMount, "/"),
				fmt.Sprintf("%s%d.sock", vmmSocketPrefix, namespace),
			),
		}
	}
	return processes, nil
}

func processUID(path string) (uint32, error) {
	var status unix.Stat_t
	if err := unix.Stat(path, &status); err != nil {
		return 0, err
	}
	return status.Uid, nil
}

// RestorePeerMappingProcesses resolves shim endpoints from inside the restored
// PID/mount namespace.
func RestorePeerMappingProcesses(
	restoredPIDs []int,
) ([]PeerMappingProcess, error) {
	processes := make([]PeerMappingProcess, len(restoredPIDs))
	seen := make(map[int]struct{}, len(restoredPIDs))
	for index, pid := range restoredPIDs {
		if pid <= 0 {
			return nil, fmt.Errorf("invalid restored CUDA PID %d", pid)
		}
		if _, duplicate := seen[pid]; duplicate {
			return nil, fmt.Errorf("duplicate restored CUDA PID %d", pid)
		}
		seen[pid] = struct{}{}
		uid, err := processUID(
			filepath.Join("/proc", strconv.Itoa(pid)),
		)
		if err != nil {
			return nil, fmt.Errorf(
				"inspect restored CUDA process %d credentials: %w",
				pid, err,
			)
		}
		processes[index] = PeerMappingProcess{
			ObservedPID:  pid,
			NamespacePID: pid,
			UID:          uid,
			SocketPath: filepath.Join(
				vmmControlMount,
				fmt.Sprintf("%s%d.sock", vmmSocketPrefix, pid),
			),
		}
	}
	return processes, nil
}

// PreparePeerMappings synchronizes participating contexts, detaches only
// imported POSIX peer mappings/handles, and verifies every process is ready for
// CUDA lock. The caller already owns workload quiescence and process discovery.
func PreparePeerMappings(
	ctx context.Context,
	processes []PeerMappingProcess,
	generation uint64,
	log logr.Logger,
) error {
	if generation == 0 {
		return errors.New("CUDA peer-mapping generation must be nonzero")
	}
	audits, objects, err := auditVMMProcesses(ctx, processes, generation)
	if err != nil {
		return abortPeerMappingPrepare(
			auditProcesses(audits),
			generation,
			fmt.Errorf("audit CUDA peer mappings: %w", err),
		)
	}
	if !hasVMMImporter(objects) {
		return abortPeerMappingPrepare(
			processes,
			generation,
			errors.New("no managed CUDA POSIX peer mappings discovered"),
		)
	}
	for _, step := range []struct {
		operation uint16
		phase     uint32
		name      string
	}{
		{vmmOpSyncAllContexts, vmmPhaseContextsSynced, "synchronize contexts"},
		{vmmOpDetachImports, vmmPhaseLockReady, "detach imported peers"},
		{vmmOpVerifyLockReady, vmmPhaseLockReady, "verify lock-ready state"},
	} {
		if err := runVMMBarrier(
			ctx, audits, generation, step.operation, step.phase,
		); err != nil {
			return abortPeerMappingPrepare(
				processes,
				generation,
				fmt.Errorf("%s: %w", step.name, err),
			)
		}
	}
	after, afterObjects, err := auditVMMProcesses(
		ctx, processes, generation,
	)
	if err != nil {
		return abortPeerMappingPrepare(
			processes,
			generation,
			fmt.Errorf("re-audit detached CUDA peer mappings: %w", err),
		)
	}
	if err := compareVMMGraphs(objects, afterObjects); err != nil {
		return abortPeerMappingPrepare(
			processes,
			generation,
			fmt.Errorf(
				"CUDA peer resource graph drifted during prepare: %w",
				err,
			),
		)
	}
	for _, audit := range after {
		if audit.phase != vmmPhaseLockReady {
			return abortPeerMappingPrepare(
				processes,
				generation,
				fmt.Errorf(
					"CUDA process %d ended prepare in phase %d",
					audit.process.ObservedPID, audit.phase,
				),
			)
		}
	}
	log.Info(
		"Detached CUDA POSIX peer mappings before CUDA lock",
		"processes", len(processes),
		"objects", len(objects),
		"generation", generation,
	)
	return nil
}

func auditProcesses(audits []vmmAudit) []PeerMappingProcess {
	processes := make([]PeerMappingProcess, len(audits))
	for index, audit := range audits {
		processes[index] = audit.process
	}
	return processes
}

func hasVMMImporter(objects map[vmmObjectID]vmmObject) bool {
	for _, object := range objects {
		if len(object.importers) != 0 {
			return true
		}
	}
	return false
}

// RestorePeerMappings brokers fresh owner FDs and commands importers to restore
// their exact peer VAs and access after every CUDA process is restored/unlocked.
func RestorePeerMappings(
	ctx context.Context,
	processes []PeerMappingProcess,
	generation uint64,
	log logr.Logger,
) error {
	if generation == 0 {
		return errors.New("CUDA peer-mapping generation must be nonzero")
	}
	audits, objects, err := auditVMMProcesses(ctx, processes, generation)
	if err != nil {
		return fmt.Errorf("audit restored CUDA peer mappings: %w", err)
	}
	for _, audit := range audits {
		if audit.generation != generation ||
			audit.phase != vmmPhaseLockReady {
			return fmt.Errorf(
				"restored CUDA process %d generation/phase=%d/%d, want %d/LOCK_READY",
				audit.process.ObservedPID,
				audit.generation,
				audit.phase,
				generation,
			)
		}
	}
	if err := brokerVMMObjects(ctx, objects, generation); err != nil {
		return err
	}
	if err := runVMMBarrier(
		ctx, audits, generation,
		vmmOpVerifyActive, vmmPhaseActiveVerified,
	); err != nil {
		return fmt.Errorf("verify restored CUDA peer mappings: %w", err)
	}
	if err := runVMMBarrier(
		ctx, audits, generation, vmmOpCommit, vmmPhaseActive,
	); err != nil {
		return fmt.Errorf("commit restored CUDA peer mappings: %w", err)
	}
	log.Info(
		"Restored CUDA POSIX peer mappings after CUDA unlock",
		"processes", len(processes),
		"objects", len(objects),
		"generation", generation,
	)
	return nil
}

func abortPeerMappingPrepare(
	processes []PeerMappingProcess,
	generation uint64,
	prepareErr error,
) error {
	recoveryErr := AbortPeerMappings(
		processes, generation, logr.Discard(),
	)
	if recoveryErr == nil {
		return prepareErr
	}
	return errors.Join(
		prepareErr,
		fmt.Errorf(
			"abort CUDA peer mappings after pre-lock failure: %w",
			recoveryErr,
		),
	)
}

// AbortPeerMappings best-effort reattaches detached imported peers and resets
// the generation on every contacted shim after a failure before CUDA lock.
// Audits, separable peer edges, verification, and resets use independent
// bounded attempts so cancellation or one stalled endpoint cannot strand
// healthy process-local shim state.
func AbortPeerMappings(
	processes []PeerMappingProcess,
	generation uint64,
	log logr.Logger,
) error {
	return abortPeerMappings(
		processes, generation, log, vmmRecoveryTimeout,
	)
}

func abortPeerMappings(
	processes []PeerMappingProcess,
	generation uint64,
	log logr.Logger,
	attemptTimeout time.Duration,
) error {
	if generation == 0 {
		return errors.New("CUDA peer-mapping generation must be nonzero")
	}
	if len(processes) == 0 {
		return nil
	}

	audits := make([]vmmAudit, 0, len(processes))
	var recoveryErr error
	type auditResult struct {
		audit vmmAudit
		err   error
	}
	auditResults := make([]auditResult, len(processes))
	var wait sync.WaitGroup
	for index, process := range processes {
		wait.Add(1)
		go func() {
			defer wait.Done()
			ctx, cancel := context.WithTimeout(
				context.Background(), attemptTimeout,
			)
			defer cancel()
			auditResults[index].audit, auditResults[index].err =
				auditVMMProcess(ctx, process, generation)
		}()
	}
	wait.Wait()
	for index, result := range auditResults {
		if result.err != nil {
			recoveryErr = errors.Join(
				recoveryErr,
				fmt.Errorf(
					"audit CUDA process %d (namespace PID %d) for peer recovery: %w",
					processes[index].ObservedPID,
					processes[index].NamespacePID,
					result.err,
				),
			)
		}
		if result.audit.generation == generation {
			audits = append(audits, result.audit)
		}
	}

	detached := make([]vmmAudit, 0, len(audits))
	selectedImporters := make(map[int]struct{})
	for _, audit := range audits {
		switch audit.phase {
		case vmmPhaseLockReady, vmmPhaseReattaching:
			detached = append(detached, audit)
			for _, record := range audit.records {
				if record.Role == vmmRoleImporter {
					selectedImporters[audit.process.NamespacePID] = struct{}{}
					break
				}
			}
		case vmmPhaseActive, vmmPhaseAudited,
			vmmPhaseContextsSynced, vmmPhaseActiveVerified:
		case vmmPhasePoisoned:
			recoveryErr = errors.Join(
				recoveryErr,
				fmt.Errorf(
					"CUDA process %d is poisoned after prepare failure",
					audit.process.ObservedPID,
				),
			)
		default:
			recoveryErr = errors.Join(
				recoveryErr,
				fmt.Errorf(
					"CUDA process %d cannot recover from phase %d",
					audit.process.ObservedPID,
					audit.phase,
				),
			)
		}
	}

	if len(detached) > 0 {
		objects, graphErr := buildVMMGraph(audits)
		if graphErr != nil {
			recoveryErr = errors.Join(
				recoveryErr,
				fmt.Errorf(
					"build peer-remap recovery graph: %w",
					graphErr,
				),
			)
		} else if len(selectedImporters) > 0 {
			if err := brokerSelectedVMMObjectsBestEffort(
				objects,
				selectedImporters,
				generation,
				attemptTimeout,
			); err != nil {
				recoveryErr = errors.Join(
					recoveryErr,
					fmt.Errorf("best-effort peer remap: %w", err),
				)
			}
		}
		if err := runVMMRecoveryBarrier(
			detached, generation,
			vmmOpVerifyActive, vmmPhaseActiveVerified,
			attemptTimeout,
		); err != nil {
			recoveryErr = errors.Join(
				recoveryErr,
				fmt.Errorf("verify best-effort peer remap: %w", err),
			)
		}
	}

	resetErrs := make([]error, len(processes))
	for index, process := range processes {
		wait.Add(1)
		go func() {
			defer wait.Done()
			ctx, cancel := context.WithTimeout(
				context.Background(), attemptTimeout,
			)
			defer cancel()
			request := newVMMPacket(vmmOpAbort, process)
			request.Generation = generation
			response, fd, err := exchangeVMMPacket(
				ctx, process, request, -1,
			)
			closeFD(fd)
			if err != nil {
				resetErrs[index] = fmt.Errorf(
					"reset CUDA peer-mapping generation on process %d (namespace PID %d): %w",
					process.ObservedPID,
					process.NamespacePID,
					err,
				)
				return
			}
			if response.Generation != 0 ||
				response.Phase != vmmPhaseActive {
				resetErrs[index] = fmt.Errorf(
					"reset CUDA peer-mapping generation on process %d (namespace PID %d) reported generation/phase %d/%d",
					process.ObservedPID,
					process.NamespacePID,
					response.Generation,
					response.Phase,
				)
			}
		}()
	}
	wait.Wait()
	for _, err := range resetErrs {
		recoveryErr = errors.Join(recoveryErr, err)
	}
	if recoveryErr == nil {
		log.Info(
			"Best-effort CUDA peer remap and generation abort completed",
			"processes", len(processes),
			"generation", generation,
		)
	}
	return recoveryErr
}

func runVMMBarrier(
	ctx context.Context,
	audits []vmmAudit,
	generation uint64,
	operation uint16,
	expectedPhase uint32,
) error {
	for _, audit := range audits {
		request := newVMMPacket(operation, audit.process)
		request.Generation = generation
		response, fd, err := exchangeVMMPacket(
			ctx, audit.process, request, -1,
		)
		closeFD(fd)
		if err != nil {
			return fmt.Errorf(
				"CUDA process %d operation %d: %w",
				audit.process.ObservedPID, operation, err,
			)
		}
		if response.Generation != generation ||
			response.Phase != expectedPhase {
			return fmt.Errorf(
				"CUDA process %d operation %d reported generation/phase %d/%d, want %d/%d",
				audit.process.ObservedPID,
				operation,
				response.Generation,
				response.Phase,
				generation,
				expectedPhase,
			)
		}
	}
	return nil
}

func runVMMRecoveryBarrier(
	audits []vmmAudit,
	generation uint64,
	operation uint16,
	expectedPhase uint32,
	attemptTimeout time.Duration,
) error {
	errs := make([]error, len(audits))
	var wait sync.WaitGroup
	for index, audit := range audits {
		wait.Add(1)
		go func() {
			defer wait.Done()
			ctx, cancel := context.WithTimeout(
				context.Background(), attemptTimeout,
			)
			defer cancel()
			errs[index] = runVMMBarrier(
				ctx,
				[]vmmAudit{audit},
				generation,
				operation,
				expectedPhase,
			)
		}()
	}
	wait.Wait()
	var barrierErr error
	for _, err := range errs {
		barrierErr = errors.Join(barrierErr, err)
	}
	return barrierErr
}

func brokerVMMObjects(
	ctx context.Context,
	objects map[vmmObjectID]vmmObject,
	generation uint64,
) error {
	return brokerSelectedVMMObjects(ctx, objects, nil, generation)
}

func brokerSelectedVMMObjects(
	ctx context.Context,
	objects map[vmmObjectID]vmmObject,
	selected map[int]struct{},
	generation uint64,
) error {
	var brokerErr error
	for id, object := range objects {
		importers := object.importers
		if selected != nil {
			importers = make([]PeerMappingProcess, 0, len(object.importers))
			for _, importer := range object.importers {
				if _, ok := selected[importer.NamespacePID]; ok {
					importers = append(importers, importer)
				}
			}
		}
		if len(importers) == 0 {
			continue
		}
		exportRequest := newVMMPacket(vmmOpExport, object.owner)
		exportRequest.Generation = generation
		exportRequest.ObjectDev = id.dev
		exportRequest.ObjectIno = id.ino
		exportResponse, fd, err := exchangeVMMPacket(
			ctx, object.owner, exportRequest, -1,
		)
		if err != nil {
			closeFD(fd)
			exportErr := fmt.Errorf(
				"fresh export for object %s from process %d: %w",
				id, object.owner.ObservedPID, err,
			)
			if selected == nil {
				return exportErr
			}
			brokerErr = errors.Join(brokerErr, exportErr)
			continue
		}
		if fd < 0 {
			exportErr := fmt.Errorf(
				"fresh export for object %s returned no FD", id,
			)
			if selected == nil {
				return exportErr
			}
			brokerErr = errors.Join(brokerErr, exportErr)
			continue
		}
		if exportResponse.ObjectDev != id.dev ||
			exportResponse.ObjectIno != id.ino {
			closeFD(fd)
			exportErr := fmt.Errorf(
				"fresh export identity is %d:%d, want original object %s",
				exportResponse.ObjectDev,
				exportResponse.ObjectIno,
				id,
			)
			if selected == nil {
				return exportErr
			}
			brokerErr = errors.Join(brokerErr, exportErr)
			continue
		}
		var status unix.Stat_t
		if err := unix.Fstat(fd, &status); err != nil {
			closeFD(fd)
			exportErr := fmt.Errorf(
				"fstat fresh FD for object %s: %w", id, err,
			)
			if selected == nil {
				return exportErr
			}
			brokerErr = errors.Join(brokerErr, exportErr)
			continue
		}
		if uint64(status.Dev) != exportResponse.FreshDev ||
			status.Ino != exportResponse.FreshIno {
			closeFD(fd)
			exportErr := fmt.Errorf(
				"fresh FD for object %s is %d:%d, owner reported %d:%d",
				id,
				status.Dev,
				status.Ino,
				exportResponse.FreshDev,
				exportResponse.FreshIno,
			)
			if selected == nil {
				return exportErr
			}
			brokerErr = errors.Join(brokerErr, exportErr)
			continue
		}
		for _, importer := range importers {
			importRequest := newVMMPacket(vmmOpImport, importer)
			importRequest.Generation = generation
			importRequest.ObjectDev = id.dev
			importRequest.ObjectIno = id.ino
			importRequest.FreshDev = exportResponse.FreshDev
			importRequest.FreshIno = exportResponse.FreshIno
			_, receivedFD, importErr := exchangeVMMPacket(
				ctx, importer, importRequest, fd,
			)
			closeFD(receivedFD)
			if importErr != nil {
				wrapped := fmt.Errorf(
					"fresh import for object %s in process %d: %w",
					id, importer.ObservedPID, importErr,
				)
				if selected == nil {
					closeFD(fd)
					return wrapped
				}
				brokerErr = errors.Join(brokerErr, wrapped)
			}
		}
		closeFD(fd)
	}
	return brokerErr
}

func brokerSelectedVMMObjectsBestEffort(
	objects map[vmmObjectID]vmmObject,
	selected map[int]struct{},
	generation uint64,
	attemptTimeout time.Duration,
) error {
	type recoveryEdge struct {
		id       vmmObjectID
		object   vmmObject
		importer PeerMappingProcess
	}
	var edges []recoveryEdge
	for id, object := range objects {
		for _, importer := range object.importers {
			if _, ok := selected[importer.NamespacePID]; ok {
				edges = append(edges, recoveryEdge{
					id:       id,
					object:   object,
					importer: importer,
				})
			}
		}
	}
	errs := make([]error, len(edges))
	var wait sync.WaitGroup
	for index, edge := range edges {
		wait.Add(1)
		go func() {
			defer wait.Done()
			ctx, cancel := context.WithTimeout(
				context.Background(), attemptTimeout,
			)
			defer cancel()
			object := edge.object
			object.importers = []PeerMappingProcess{edge.importer}
			errs[index] = brokerSelectedVMMObjects(
				ctx,
				map[vmmObjectID]vmmObject{edge.id: object},
				nil,
				generation,
			)
		}()
	}
	wait.Wait()
	var brokerErr error
	for _, err := range errs {
		brokerErr = errors.Join(brokerErr, err)
	}
	return brokerErr
}

func auditVMMProcesses(
	ctx context.Context,
	processes []PeerMappingProcess,
	generation uint64,
) ([]vmmAudit, map[vmmObjectID]vmmObject, error) {
	if len(processes) == 0 {
		return nil, nil, errors.New("no CUDA shim processes registered")
	}
	audits := make([]vmmAudit, 0, len(processes))
	for _, process := range processes {
		audit, err := auditVMMProcess(ctx, process, generation)
		audits = append(audits, audit)
		if err != nil {
			return audits, nil, fmt.Errorf(
				"process %d: %w", process.ObservedPID, err,
			)
		}
	}
	objects, err := buildVMMGraph(audits)
	if err != nil {
		return audits, nil, err
	}
	return audits, objects, nil
}

func auditVMMProcess(
	ctx context.Context,
	process PeerMappingProcess,
	generation uint64,
) (vmmAudit, error) {
	request := newVMMPacket(vmmOpAudit, process)
	request.Generation = generation
	response, fd, conn, err := beginVMMExchange(
		ctx, process, request, -1,
	)
	closeFD(fd)
	audit := vmmAudit{
		process:    process,
		generation: response.Generation,
		phase:      response.Phase,
	}
	if conn != nil {
		defer conn.Close()
	}
	if err != nil {
		return audit, err
	}
	audit.records = make([]vmmPacket, 0, response.Count)
	if response.Count > vmmMaxObjectsPerAgent {
		return audit, fmt.Errorf(
			"shim advertised %d objects, maximum is %d",
			response.Count, vmmMaxObjectsPerAgent,
		)
	}
	for range response.Count {
		record, recordFD, err := readVMMPacket(conn)
		if recordFD >= 0 {
			closeFD(recordFD)
			return audit, errors.New("audit record carried an FD")
		}
		if err != nil {
			return audit, fmt.Errorf("read audit record: %w", err)
		}
		if err := validateVMMResponse(
			record, request, process, -1,
		); err != nil {
			return audit, fmt.Errorf("validate audit record: %w", err)
		}
		if record.Generation != response.Generation ||
			record.Phase != response.Phase {
			return audit, errors.New("audit record state mismatch")
		}
		audit.records = append(audit.records, record)
	}
	return audit, nil
}

func buildVMMGraph(
	audits []vmmAudit,
) (map[vmmObjectID]vmmObject, error) {
	objects := make(map[vmmObjectID]vmmObject)
	for _, audit := range audits {
		for _, record := range audit.records {
			id := vmmObjectID{
				dev: record.ObjectDev,
				ino: record.ObjectIno,
			}
			object := objects[id]
			switch record.Role {
			case vmmRoleOwner:
				if object.owner.ObservedPID != 0 {
					return nil, fmt.Errorf(
						"object %s has multiple owners %d and %d",
						id,
						object.owner.ObservedPID,
						audit.process.ObservedPID,
					)
				}
				object.owner = audit.process
			case vmmRoleImporter:
				if record.MappingCount == 0 {
					return nil, fmt.Errorf(
						"object %s importer process %d has no peer mapping",
						id, audit.process.ObservedPID,
					)
				}
				object.importers = append(
					object.importers, audit.process,
				)
				if object.mappings == nil {
					object.mappings = make(map[int]uint32)
				}
				if _, duplicate := object.mappings[audit.process.NamespacePID]; duplicate {
					return nil, fmt.Errorf(
						"object %s has duplicate importer process %d",
						id, audit.process.ObservedPID,
					)
				}
				object.mappings[audit.process.NamespacePID] = record.MappingCount
			default:
				return nil, fmt.Errorf(
					"object %s has unknown role %d",
					id, record.Role,
				)
			}
			objects[id] = object
		}
	}
	for id, object := range objects {
		if len(object.importers) != 0 &&
			object.owner.ObservedPID == 0 {
			return nil, fmt.Errorf(
				"imported object %s has no owner in process tree", id,
			)
		}
	}
	return objects, nil
}

func compareVMMGraphs(
	before map[vmmObjectID]vmmObject,
	after map[vmmObjectID]vmmObject,
) error {
	if len(before) != len(after) {
		return fmt.Errorf(
			"object count changed from %d to %d",
			len(before), len(after),
		)
	}
	for id, expected := range before {
		actual, ok := after[id]
		if !ok {
			return fmt.Errorf("object %s disappeared", id)
		}
		if expected.owner.NamespacePID != actual.owner.NamespacePID {
			return fmt.Errorf("object %s owner changed", id)
		}
		expectedImporters := make([]int, len(expected.importers))
		actualImporters := make([]int, len(actual.importers))
		for index := range expected.importers {
			expectedImporters[index] = expected.importers[index].NamespacePID
		}
		for index := range actual.importers {
			actualImporters[index] = actual.importers[index].NamespacePID
		}
		slices.Sort(expectedImporters)
		slices.Sort(actualImporters)
		if !slices.Equal(expectedImporters, actualImporters) {
			return fmt.Errorf("object %s importer set changed", id)
		}
		for pid, mappingCount := range expected.mappings {
			if actual.mappings[pid] != mappingCount {
				return fmt.Errorf(
					"object %s importer %d mapping count changed from %d to %d",
					id, pid, mappingCount, actual.mappings[pid],
				)
			}
		}
	}
	return nil
}

func newVMMPacket(
	operation uint16,
	process PeerMappingProcess,
) vmmPacket {
	return vmmPacket{
		Magic:     vmmProtocolMagic,
		Version:   vmmProtocolVersion,
		Operation: operation,
		PID:       uint32(process.NamespacePID),
	}
}

func exchangeVMMPacket(
	ctx context.Context,
	process PeerMappingProcess,
	request vmmPacket,
	fd int,
) (vmmPacket, int, error) {
	response, receivedFD, conn, err := beginVMMExchange(
		ctx, process, request, fd,
	)
	if conn != nil {
		_ = conn.Close()
	}
	return response, receivedFD, err
}

func beginVMMExchange(
	ctx context.Context,
	process PeerMappingProcess,
	request vmmPacket,
	fd int,
) (vmmPacket, int, *net.UnixConn, error) {
	var response vmmPacket
	dialer := net.Dialer{Timeout: vmmControlTimeout}
	raw, err := dialer.DialContext(ctx, "unixpacket", process.SocketPath)
	if err != nil {
		return response, -1, nil, fmt.Errorf(
			"connect private CUDA shim endpoint %q: %w",
			process.SocketPath, err,
		)
	}
	conn := raw.(*net.UnixConn)
	if err := verifyVMMPeer(conn, process); err != nil {
		_ = conn.Close()
		return response, -1, nil, err
	}
	deadline := time.Now().Add(vmmControlTimeout)
	if contextDeadline, ok := ctx.Deadline(); ok &&
		contextDeadline.Before(deadline) {
		deadline = contextDeadline
	}
	if err := conn.SetDeadline(deadline); err != nil {
		_ = conn.Close()
		return response, -1, nil, err
	}
	if err := writeVMMPacket(conn, request, fd); err != nil {
		_ = conn.Close()
		return response, -1, nil, err
	}
	response, receivedFD, err := readVMMPacket(conn)
	if err != nil {
		_ = conn.Close()
		return response, receivedFD, nil, err
	}
	if err := validateVMMResponse(
		response, request, process, receivedFD,
	); err != nil {
		closeFD(receivedFD)
		_ = conn.Close()
		return response, -1, nil, err
	}
	return response, receivedFD, conn, nil
}

func verifyVMMPeer(
	conn *net.UnixConn,
	process PeerMappingProcess,
) error {
	raw, err := conn.SyscallConn()
	if err != nil {
		return fmt.Errorf("get shim raw connection: %w", err)
	}
	var credentials *unix.Ucred
	var socketErr error
	if err := raw.Control(func(fd uintptr) {
		credentials, socketErr = unix.GetsockoptUcred(
			int(fd), unix.SOL_SOCKET, unix.SO_PEERCRED,
		)
	}); err != nil {
		return fmt.Errorf("inspect shim peer: %w", err)
	}
	if socketErr != nil {
		return fmt.Errorf("inspect shim peer credentials: %w", socketErr)
	}
	if int(credentials.Pid) != process.ObservedPID ||
		credentials.Uid != process.UID {
		return fmt.Errorf(
			"shim peer credentials pid=%d uid=%d, want pid=%d uid=%d",
			credentials.Pid,
			credentials.Uid,
			process.ObservedPID,
			process.UID,
		)
	}
	return nil
}

func validateVMMResponse(
	response vmmPacket,
	request vmmPacket,
	process PeerMappingProcess,
	receivedFD int,
) error {
	if response.Magic != vmmProtocolMagic ||
		response.Version != vmmProtocolVersion ||
		response.Operation != request.Operation {
		return errors.New("invalid CUDA shim response envelope")
	}
	if response.PID != uint32(process.NamespacePID) {
		return fmt.Errorf(
			"shim PID mismatch: got %d, want %d",
			response.PID, process.NamespacePID,
		)
	}
	if request.Generation != 0 &&
		response.Generation != request.Generation &&
		!(response.Status != 0 && response.Generation == 0) &&
		!(request.Operation == vmmOpAbort &&
			response.Generation == 0) {
		return fmt.Errorf(
			"shim generation mismatch: got %d, want %d",
			response.Generation, request.Generation,
		)
	}
	if response.Phase > vmmPhasePoisoned {
		return fmt.Errorf("shim reported unknown phase %d", response.Phase)
	}
	detached := response.Phase == vmmPhaseLockReady ||
		response.Phase == vmmPhaseReattaching
	if detached != (response.Flags&vmmStateDetached != 0) {
		return errors.New("shim phase/detached flag mismatch")
	}
	poisoned := response.Phase == vmmPhasePoisoned
	if poisoned != (response.Flags&vmmStatePoisoned != 0) {
		return errors.New("shim phase/poison flag mismatch")
	}
	expectedFD := request.Operation == vmmOpExport &&
		response.Status == 0
	if expectedFD != (receivedFD >= 0) {
		return fmt.Errorf(
			"operation %d response has invalid FD presence",
			request.Operation,
		)
	}
	if response.Status != 0 {
		if receivedFD >= 0 {
			return errors.New("failed shim response carried an FD")
		}
		message := response.message()
		if message == "" {
			message = "unspecified CUDA shim failure"
		}
		return errors.New(message)
	}
	return nil
}

func writeVMMPacket(
	conn *net.UnixConn,
	packet vmmPacket,
	fd int,
) error {
	buffer := make([]byte, vmmPacketSize)
	if err := binary.Write(
		newFixedWriter(buffer), binary.LittleEndian, packet,
	); err != nil {
		return err
	}
	var rights []byte
	if fd >= 0 {
		rights = unix.UnixRights(fd)
	}
	written, ancillary, err := conn.WriteMsgUnix(buffer, rights, nil)
	if err != nil {
		return err
	}
	if written != len(buffer) || ancillary != len(rights) {
		return io.ErrShortWrite
	}
	return nil
}

func readVMMPacket(
	conn *net.UnixConn,
) (vmmPacket, int, error) {
	var packet vmmPacket
	buffer := make([]byte, vmmPacketSize)
	ancillary := make([]byte, unix.CmsgSpace(4*4))
	size, ancillarySize, flags, _, err := conn.ReadMsgUnix(
		buffer, ancillary,
	)
	if err != nil {
		return packet, -1, err
	}
	if size != vmmPacketSize ||
		flags&(unix.MSG_TRUNC|unix.MSG_CTRUNC) != 0 {
		closeAncillaryFDs(ancillary[:ancillarySize])
		return packet, -1, errors.New("truncated CUDA shim packet")
	}
	messages, err := unix.ParseSocketControlMessage(
		ancillary[:ancillarySize],
	)
	if err != nil {
		closeAncillaryFDs(ancillary[:ancillarySize])
		return packet, -1, fmt.Errorf("parse ancillary data: %w", err)
	}
	receivedFD := -1
	for _, message := range messages {
		fds, rightsErr := unix.ParseUnixRights(&message)
		if rightsErr != nil || len(fds) != 1 || receivedFD >= 0 {
			for _, fd := range fds {
				closeFD(fd)
			}
			closeFD(receivedFD)
			return packet, -1, errors.New(
				"CUDA shim packet must carry at most one SCM_RIGHTS FD",
			)
		}
		receivedFD = fds[0]
	}
	if err := binary.Read(
		strings.NewReader(string(buffer)),
		binary.LittleEndian,
		&packet,
	); err != nil {
		closeFD(receivedFD)
		return packet, -1, err
	}
	return packet, receivedFD, nil
}

func closeAncillaryFDs(ancillary []byte) {
	messages, err := unix.ParseSocketControlMessage(ancillary)
	if err != nil {
		return
	}
	for _, message := range messages {
		fds, err := unix.ParseUnixRights(&message)
		if err != nil {
			continue
		}
		for _, fd := range fds {
			closeFD(fd)
		}
	}
}

func closeFD(fd int) {
	if fd >= 0 {
		_ = unix.Close(fd)
	}
}

type fixedWriter struct {
	buffer []byte
	offset int
}

func newFixedWriter(buffer []byte) *fixedWriter {
	return &fixedWriter{buffer: buffer}
}

func (writer *fixedWriter) Write(content []byte) (int, error) {
	if len(content) > len(writer.buffer)-writer.offset {
		return 0, io.ErrShortBuffer
	}
	copy(writer.buffer[writer.offset:], content)
	writer.offset += len(content)
	return len(content), nil
}
