package executor

import (
	"context"
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/go-logr/logr"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/cuda"
)

func TestPreparePeerMappingsRunsBeforeCUDALock(t *testing.T) {
	var order []string
	_, err := prepareAndCheckpointCUDA(
		context.Background(),
		true,
		nil,
		1,
		[]int{10},
		logr.Discard(),
		func(
			context.Context,
			[]cuda.PeerMappingProcess,
			uint64,
			logr.Logger,
		) error {
			order = append(order, "prepare")
			return nil
		},
		func() error {
			order = append(order, "validate")
			return nil
		},
		func(
			[]cuda.PeerMappingProcess,
			uint64,
			logr.Logger,
		) error {
			order = append(order, "abort")
			return nil
		},
		func(
			context.Context,
			[]int,
			logr.Logger,
		) (cuda.CheckpointPhaseTimings, error) {
			order = append(order, "lock-checkpoint")
			return cuda.CheckpointPhaseTimings{}, nil
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(
		order,
		[]string{"prepare", "validate", "lock-checkpoint"},
	) {
		t.Fatalf("CUDA checkpoint order = %v", order)
	}
}

func TestDisabledInterposerKeepsDefaultCUDAOrdering(t *testing.T) {
	prepareCalled := false
	validateCalled := false
	checkpointCalled := false
	_, err := prepareAndCheckpointCUDA(
		context.Background(),
		false,
		nil,
		0,
		nil,
		logr.Discard(),
		func(
			context.Context,
			[]cuda.PeerMappingProcess,
			uint64,
			logr.Logger,
		) error {
			prepareCalled = true
			return nil
		},
		func() error {
			validateCalled = true
			return nil
		},
		func(
			[]cuda.PeerMappingProcess,
			uint64,
			logr.Logger,
		) error {
			t.Fatal("disabled interposer called abort")
			return nil
		},
		func(
			context.Context,
			[]int,
			logr.Logger,
		) (cuda.CheckpointPhaseTimings, error) {
			checkpointCalled = true
			return cuda.CheckpointPhaseTimings{}, nil
		},
	)
	if err != nil || prepareCalled || validateCalled || !checkpointCalled {
		t.Fatalf(
			"error=%v prepare=%t validate=%t checkpoint=%t",
			err, prepareCalled, validateCalled, checkpointCalled,
		)
	}

	discoverCalled := false
	peerCalled := false
	restoreCalled := false
	_, err = restoreCUDAAndPeerMappings(
		context.Background(),
		false,
		0,
		nil,
		"",
		logr.Discard(),
		func(
			context.Context,
			[]int,
			string,
			logr.Logger,
		) (cuda.RestorePhaseTimings, error) {
			restoreCalled = true
			return cuda.RestorePhaseTimings{}, nil
		},
		func() ([]cuda.PeerMappingProcess, error) {
			discoverCalled = true
			return nil, nil
		},
		func(
			context.Context,
			[]cuda.PeerMappingProcess,
			uint64,
			logr.Logger,
		) error {
			peerCalled = true
			return nil
		},
	)
	if err != nil || !restoreCalled || discoverCalled || peerCalled {
		t.Fatalf(
			"error=%v restore=%t discover=%t peers=%t",
			err, restoreCalled, discoverCalled, peerCalled,
		)
	}
}

func TestPrepareFailureSuppressesCUDALock(t *testing.T) {
	validateCalled := false
	lockCalled := false
	_, err := prepareAndCheckpointCUDA(
		context.Background(),
		true,
		nil,
		1,
		nil,
		logr.Discard(),
		func(
			context.Context,
			[]cuda.PeerMappingProcess,
			uint64,
			logr.Logger,
		) error {
			return errors.New("prepare failed")
		},
		func() error {
			validateCalled = true
			return nil
		},
		func(
			[]cuda.PeerMappingProcess,
			uint64,
			logr.Logger,
		) error {
			t.Fatal("prepare failure called duplicate abort")
			return nil
		},
		func(
			context.Context,
			[]int,
			logr.Logger,
		) (cuda.CheckpointPhaseTimings, error) {
			lockCalled = true
			return cuda.CheckpointPhaseTimings{}, nil
		},
	)
	if err == nil || validateCalled || lockCalled {
		t.Fatalf(
			"error=%v validateCalled=%t lockCalled=%t",
			err, validateCalled, lockCalled,
		)
	}
}

func TestRecoveredImportedHandleReleaseFailureNeverCallsCUDALock(
	t *testing.T,
) {
	validateCalled := false
	lockCalls := 0
	_, err := prepareAndCheckpointCUDA(
		context.Background(),
		true,
		nil,
		71,
		[]int{10},
		logr.Discard(),
		func(
			context.Context,
			[]cuda.PeerMappingProcess,
			uint64,
			logr.Logger,
		) error {
			return errors.New(
				"failed to release imported real handle; " +
					"local peer rollback restored active mappings",
			)
		},
		func() error {
			validateCalled = true
			return nil
		},
		func(
			[]cuda.PeerMappingProcess,
			uint64,
			logr.Logger,
		) error {
			t.Fatal("prepare owns release-failure rollback")
			return nil
		},
		func(
			context.Context,
			[]int,
			logr.Logger,
		) (cuda.CheckpointPhaseTimings, error) {
			lockCalls++
			return cuda.CheckpointPhaseTimings{}, nil
		},
	)
	if err == nil ||
		!strings.Contains(
			err.Error(), "failed to release imported real handle",
		) ||
		validateCalled ||
		lockCalls != 0 {
		t.Fatalf(
			"error=%v validateCalled=%t lockCalls=%d",
			err,
			validateCalled,
			lockCalls,
		)
	}
}

func TestPostPrepareProcessDriftSuppressesCUDALock(t *testing.T) {
	const generation = 17
	processes := []cuda.PeerMappingProcess{{ObservedPID: 11}}
	peerActive := false
	agentGeneration := generation
	abortCalled := false
	lockCalled := false
	_, err := prepareAndCheckpointCUDA(
		context.Background(),
		true,
		processes,
		generation,
		nil,
		logr.Discard(),
		func(
			context.Context,
			[]cuda.PeerMappingProcess,
			uint64,
			logr.Logger,
		) error {
			peerActive = false
			return nil
		},
		func() error {
			return errors.New("CUDA process set drift")
		},
		func(
			gotProcesses []cuda.PeerMappingProcess,
			gotGeneration uint64,
			_ logr.Logger,
		) error {
			abortCalled = true
			if !slices.Equal(gotProcesses, processes) ||
				gotGeneration != generation {
				t.Fatalf(
					"abort processes/generation = %v/%d, want %v/%d",
					gotProcesses,
					gotGeneration,
					processes,
					generation,
				)
			}
			peerActive = true
			agentGeneration = 0
			return nil
		},
		func(
			context.Context,
			[]int,
			logr.Logger,
		) (cuda.CheckpointPhaseTimings, error) {
			lockCalled = true
			return cuda.CheckpointPhaseTimings{}, nil
		},
	)
	if err == nil || !abortCalled || !peerActive ||
		agentGeneration != 0 || lockCalled {
		t.Fatalf(
			"error=%v abort=%t active=%t generation=%d lock=%t",
			err,
			abortCalled,
			peerActive,
			agentGeneration,
			lockCalled,
		)
	}
}

func TestPostPrepareProcessDriftIncludesAbortFailure(t *testing.T) {
	lockCalled := false
	_, err := prepareAndCheckpointCUDA(
		context.Background(),
		true,
		nil,
		1,
		nil,
		logr.Discard(),
		func(
			context.Context,
			[]cuda.PeerMappingProcess,
			uint64,
			logr.Logger,
		) error {
			return nil
		},
		func() error {
			return errors.New("CUDA process set drift")
		},
		func(
			[]cuda.PeerMappingProcess,
			uint64,
			logr.Logger,
		) error {
			return errors.New("peer remap failed")
		},
		func(
			context.Context,
			[]int,
			logr.Logger,
		) (cuda.CheckpointPhaseTimings, error) {
			lockCalled = true
			return cuda.CheckpointPhaseTimings{}, nil
		},
	)
	if err == nil ||
		!strings.Contains(err.Error(), "CUDA process set drift") ||
		!strings.Contains(err.Error(), "peer remap failed") ||
		lockCalled {
		t.Fatalf("error=%v lockCalled=%t", err, lockCalled)
	}
}

func TestPeerMappingsRestoreOnlyAfterCUDAUnlock(t *testing.T) {
	var order []string
	_, err := restoreCUDAAndPeerMappings(
		context.Background(),
		true,
		1,
		[]int{10},
		"",
		logr.Discard(),
		func(
			context.Context,
			[]int,
			string,
			logr.Logger,
		) (cuda.RestorePhaseTimings, error) {
			order = append(order, "restore-unlock")
			return cuda.RestorePhaseTimings{}, nil
		},
		func() ([]cuda.PeerMappingProcess, error) {
			order = append(order, "discover")
			return nil, nil
		},
		func(
			context.Context,
			[]cuda.PeerMappingProcess,
			uint64,
			logr.Logger,
		) error {
			order = append(order, "peer-reattach")
			return nil
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(
		order,
		[]string{"restore-unlock", "discover", "peer-reattach"},
	) {
		t.Fatalf("CUDA restore order = %v", order)
	}
}

func TestPostUnlockDiscoveryFailureSuppressesPeerRestore(t *testing.T) {
	peerCalled := false
	_, err := restoreCUDAAndPeerMappings(
		context.Background(),
		true,
		1,
		nil,
		"",
		logr.Discard(),
		func(
			context.Context,
			[]int,
			string,
			logr.Logger,
		) (cuda.RestorePhaseTimings, error) {
			return cuda.RestorePhaseTimings{}, nil
		},
		func() ([]cuda.PeerMappingProcess, error) {
			return nil, errors.New("restored CUDA process set drift")
		},
		func(
			context.Context,
			[]cuda.PeerMappingProcess,
			uint64,
			logr.Logger,
		) error {
			peerCalled = true
			return nil
		},
	)
	if err == nil || peerCalled {
		t.Fatalf("error=%v peerCalled=%t", err, peerCalled)
	}
}

func TestCUDAUnlockFailureSuppressesPeerRestore(t *testing.T) {
	discoverCalled := false
	peerCalled := false
	_, err := restoreCUDAAndPeerMappings(
		context.Background(),
		true,
		1,
		nil,
		"",
		logr.Discard(),
		func(
			context.Context,
			[]int,
			string,
			logr.Logger,
		) (cuda.RestorePhaseTimings, error) {
			return cuda.RestorePhaseTimings{}, errors.New("unlock failed")
		},
		func() ([]cuda.PeerMappingProcess, error) {
			discoverCalled = true
			return nil, nil
		},
		func(
			context.Context,
			[]cuda.PeerMappingProcess,
			uint64,
			logr.Logger,
		) error {
			peerCalled = true
			return nil
		},
	)
	if err == nil || discoverCalled || peerCalled {
		t.Fatalf(
			"error=%v discoverCalled=%t peerCalled=%t",
			err, discoverCalled, peerCalled,
		)
	}
}
