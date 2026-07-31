<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Snapshot CUDA POSIX VMM interposer

This opt-in proof of concept lets Dynamo Snapshot detach and restore CUDA VMM
peer-sharing edges on one eight-GPU NVLink node. CUDA checkpoint remains
responsible for native owner allocations, native owner mappings, ordinary
allocations, and all allocation contents.

## Lifecycle

```mermaid
sequenceDiagram
    participant W as Quiesced workload
    participant S as Dynamo Snapshot
    participant I as Importer shim
    participant C as CUDA checkpoint
    participant O as Owner shim

    W->>S: ready-for-snapshot
    S->>I: PreparePeerMappings
    I->>I: sync contexts; unmap peer VAs; release imported handles
    S->>C: lock; checkpoint
    S->>S: CRIU dump / restore
    S->>C: restore; unlock all CUDA PIDs
    S->>O: fresh POSIX export
    O-->>S: SCM_RIGHTS FD
    S->>I: fresh FD + original object identity
    I->>I: import; exact-VA/offset map; access replay; sync
    S->>W: restore-complete
```

Owner handles stay native and application-visible. A successful POSIX import
receives a collision-resistant tagged logical handle. Supported imported-handle
consumers translate that logical handle to the current real driver handle.
After restore, the logical handle and mapped pointer stay unchanged while the
real imported handle may differ.

`cuMemSetAccess` is still issued once for the application's complete range.
Only intersections with tracked imported mappings are journaled and replayed;
native owner and unrelated regions are never lifecycle-managed by the shim.

## Image integration

`deploy/snapshot/Dockerfile` builds the DSO in the CUDA-devel helper stage and
installs it in the source/restore workload image at:

```text
/usr/local/lib/dynamo/libdynamo_snapshot_cuda_vmm.so
```

The same path is listed in `/etc/ld.so.preload`, so normal dynamically linked
vLLM spawn/exec workers load it and forked children inherit it. The constructor
does not initialize CUDA or bind a socket. Unless
`DYN_SNAPSHOT_CUDA_VMM_INTERPOSE=1` is set, wrappers are transparent. When
enabled, the private Snapshot control endpoint starts lazily on the first
relevant CUDA call.

Static executables, setuid/secure-execution binaries that ignore preload
configuration, non-glibc loaders, and non-base `dlmopen` namespaces are not
supported. Standard dynamically linked vLLM workers using spawn/exec are the
target. Fork before CUDA use may continue; CUDA/VMM state inherited across fork
must exec before further CUDA use.

### CI-only image delivery

Do not build container images locally or in a Kubernetes cluster. After source
review and local native/Go tests:

1. create focused DCO-signed commits and push the feature branch/PR;
2. wait for repository CI to publish the matching ACR artifacts;
3. record the exact tag and immutable digest for both artifacts:

   | CI artifact | Dockerfile target | Exact ACR reference |
   | --- | --- | --- |
   | Snapshot agent, PR job `Snapshot Agent` | `deploy/snapshot/Dockerfile:agent` | `${ACR}/ai-dynamo/dynamo:${GITHUB_SHA}-snapshot-agent@sha256:<recorded-digest>` |
   | vLLM source/restore image, PR job `vLLM Snapshot Placeholder` | `deploy/snapshot/Dockerfile:placeholder` | `${ACR}/ai-dynamo/dynamo:<runtime-version>-ci-${GITHUB_SHA}-vllm-placeholder@sha256:<recorded-digest>` |

4. reference those CI-built ACR digests directly in the Snapshot agent and TP8
   workload/restore manifests.

Use the exact commit SHA from the reviewed feature branch, not a moving branch
tag. Read `<runtime-version>` from the `snapshot-placeholder-vllm` job output
and resolve both immutable digests from ACR after those jobs succeed. Record all
four values—commit, runtime version, agent digest, and placeholder digest—in
the TP8 test report before deploying.

The two artifacts must come from the same reviewed revision. Target clusters
are expected to have ACR pull credentials. Do not use Docker, Podman/Buildah,
BuildKit, Kaniko, or an in-cluster image build. Local DSO compilation and unit
tests are source-level checks, not image builds.

### Real-CUDA preflight

The vLLM placeholder CI artifact contains:

```text
/usr/local/lib/dynamo/libdynamo_snapshot_cuda_vmm.so
/usr/local/bin/dynamo-snapshot-cuda-vmm-pair
```

Before the TP8 checkpoint test, replace
`ACR_VLLM_PLACEHOLDER_IMAGE_AT_DIGEST` in
`cmd/cuda-vmm-interpose/real/pod.yaml` with the recorded immutable vLLM
placeholder reference and deploy that manifest on a two-GPU node. Do not
compile inside the pod. The harness exec-spawns an owner and importer, releases
the application's native owner handle while preserving its owner mapping,
detaches only the imported peer, re-exports/imports with `SCM_RIGHTS`, remaps
the exact peer VA at a nonzero physical offset, replays access, performs a
peer DtoD write, verifies owner bytes, and cleans up.

A passing result ends with one JSON record whose `status` is `pass`. Preserve
the complete JSONL output and shim stderr alongside the TP8 report.

## Enablement

Set both variables in the source workload and restore placeholder:

```bash
export DYN_SNAPSHOT_CUDA_VMM_INTERPOSE=1
export DYN_SNAPSHOT_CUDA_VMM_FORCE_POSIX=1
export TORCH_SYMM_MEM_DISABLE_MULTICAST=1
```

The current PyTorch symmetric-memory allocator first asks for FABRIC
granularity, then attempts FABRIC creation. The granularity query reaches the
real driver; force-POSIX rejects the create capability trial so PyTorch selects
POSIX. This rejected trial is not a live unsupported resource and does not
poison Snapshot prepare. A FABRIC allocation that somehow becomes live remains
a fail-closed audit error. `TORCH_SYMM_MEM_DISABLE_MULTICAST=1` is mandatory for
the target path; the shim does not implement multicast. Default Snapshot
behavior, including multicast direct calls and resolver-returned entry points,
is unchanged when interposition is not enabled.

The workload must already use the existing Snapshot sleep/wake lifecycle and be
quiesced before writing `ready-for-snapshot`. The DSO does not schedule
checkpoints or trigger framework lifecycle operations. Its Unix endpoint is a
private transport used only by Snapshot's `PreparePeerMappings` and
`RestorePeerMappings` operations.

The endpoint and CPU registry are created lazily in source workload processes
and are included in that process tree's CRIU image. Restore-standby placeholder
processes remain inert and do not bind competing endpoints. After CRIU restore,
Snapshot addresses each restored endpoint by its restored PID under the
restored `/snapshot-control` mount; it does not start an autonomous daemon.

## Scope and fail-fast behavior

Supported:

- one process tree on one node;
- eight GPUs connected through NVLink;
- `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`;
- direct CUDA symbols, CUDA driver resolvers, CUDA runtime resolvers, and
  explicit `dlopen(libcuda/libcudart)` plus `dlsym`/`dlvsym`;
- imported generic-handle map, release, properties, retain-by-address, exact VA
  remap, physical offset, and access replay.

Rejected:

- FABRIC/IMEX and multi-node sharing;
- multicast/NVLS and MNNVL;
- legacy CUDA IPC;
- NCCL VMM/NVLS/IB data paths;
- array mappings backed by managed imported handles;
- CUDA state inherited across a post-CUDA fork;
- process registration or owner/importer graph drift.

The PoC uses a bounded, single-node `SOCK_SEQPACKET` transport and sends fresh
descriptors only through `SCM_RIGHTS`. It validates packet ABI, peer
credentials, PID, generation, exact ancillary multiplicity/truncation, and
fresh `fstat` identity. A prepare failure aborts the checkpoint and attempts a
best-effort peer remap. Any failure after CUDA lock begins fails restore and
prevents `restore-complete`.

## Qwen TP8 validation manifest

Target:

| Setting | Required value |
| --- | --- |
| Cluster | `nscale-dev` |
| Node | one non-DRA 8xB200 node |
| Model | `Qwen/Qwen3-0.6B` |
| Tensor parallelism | `8` |
| FlashInfer backend | TRT-LLM |
| vLLM custom allreduce | disabled |
| CUDA graphs | enabled for the final proof |

Environment:

```bash
export DYN_SNAPSHOT_CUDA_VMM_INTERPOSE=1
export DYN_SNAPSHOT_CUDA_VMM_FORCE_POSIX=1
export TORCH_SYMM_MEM_DISABLE_MULTICAST=1
export VLLM_ALLREDUCE_USE_FLASHINFER=1
export VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_USE_NCCL_SYMM_MEM=0
export NCCL_CUMEM_ENABLE=0
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_NVLS_ENABLE=0
export NCCL_MNNVL_ENABLE=0
export NCCL_IB_DISABLE=1
```

For the reviewed target path, explicit `backend=trtllm` avoids the MNNVL
multicast workspace, `VLLM_ALLREDUCE_USE_SYMM_MEM=0` disables vLLM's PyTorch
symmetric-memory backend, force-POSIX makes PyTorch select the allocator's
POSIX fallback, and `TORCH_SYMM_MEM_DISABLE_MULTICAST=1` disables PyTorch's
current multicast path. Any multicast call in enabled Phase 1 mode is rejected
immediately and counted, but that shim-generated rejection is not itself proof
that an unsupported object became live and does not poison a later audit.

Before any real-CUDA pair or TP8 run, inspect the exact CI image and save this
preflight output with the report:

```bash
test "${TORCH_SYMM_MEM_DISABLE_MULTICAST:?must be set}" = 1
python - <<'PY'
from pathlib import Path
import torch

root = Path(torch.__file__).resolve().parent
matches = [
    path for path in root.rglob("*")
    if path.is_file()
    and b"TORCH_SYMM_MEM_DISABLE_MULTICAST" in path.read_bytes()
]
assert matches, "installed PyTorch lacks the required multicast control"
print("torch_version", torch.__version__)
print("torch_root", root)
print("multicast_control_files", *matches, sep="\n")
PY
```

Capture shim stderr separately. The target POSIX run must have no
`event=multicast_rejected` line, and every final per-process commit line must
contain `multicast_attempts=0 unsupported=0x0`:

```bash
! grep -q 'event=multicast_rejected' /evidence/shim.stderr
test "$(grep -c 'event=commit.*multicast_attempts=0 unsupported=0x0' \
  /evidence/shim.stderr)" -eq 8
```

Pass:

```text
--tensor-parallel-size 8 --disable-custom-all-reduce
```

Do not use the `auto` FlashInfer backend. Do not use eager mode for the final
CUDA Graph stability proof; eager mode is acceptable only as a separately
labelled diagnostic.

Evidence to collect:

1. shim `owner_export` and `peer_import` records show only POSIX identities;
2. each `peer_detach` has a matching `peer_reattach` with the same VA, size,
   and offset and a fresh real handle;
3. all eight commit records show
   `multicast_attempts=0 unsupported=0x0`;
4. FlashInfer logs prove TRT-LLM allreduce before and after restore;
5. logs contain no backend `checkpoint_prepare`/`checkpoint_restore`, FABRIC,
   multicast/NVLS/MNNVL, legacy IPC, NCCL model allreduce, or IB path;
6. one checkpoint/restore cycle reaches `restore-complete`;
7. several deterministic temperature-0 prompts match baseline and produce
   coherent responses after restore;
8. teardown leaves no shim socket, imported mapping, or transferred FD leak.

Repeated generations and multi-node TP16/MNNVL/FABRIC work are separate future
phases and are not acceptance criteria for this PoC.
