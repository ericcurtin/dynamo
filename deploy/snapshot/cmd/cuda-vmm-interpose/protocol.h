/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DYN_SNAPSHOT_CUDA_VMM_PROTOCOL_H
#define DYN_SNAPSHOT_CUDA_VMM_PROTOCOL_H

#include <stdint.h>

#define DYN_VMM_MAGIC 0x44564d4dU
#define DYN_VMM_VERSION 3U
#define DYN_VMM_SOCKET_PREFIX "cuda-vmm-"

enum dyn_vmm_operation {
  DYN_VMM_OP_AUDIT = 1,
  DYN_VMM_OP_SYNC_ALL_CONTEXTS = 2,
  DYN_VMM_OP_DETACH_IMPORTS = 3,
  DYN_VMM_OP_VERIFY_LOCK_READY = 4,
  DYN_VMM_OP_EXPORT = 5,
  DYN_VMM_OP_IMPORT = 6,
  DYN_VMM_OP_VERIFY_ACTIVE = 7,
  DYN_VMM_OP_COMMIT = 8,
  DYN_VMM_OP_ABORT = 9,
};

enum dyn_vmm_role {
  DYN_VMM_ROLE_OWNER = 1,
  DYN_VMM_ROLE_IMPORTER = 2,
};

enum dyn_vmm_flags {
  DYN_VMM_UNSUPPORTED_FABRIC = 1U << 0,
  DYN_VMM_UNSUPPORTED_MULTICAST = 1U << 1,
  DYN_VMM_UNSUPPORTED_LEGACY_IPC = 1U << 2,
  DYN_VMM_UNSUPPORTED_UNKNOWN_HANDLE = 1U << 3,
  DYN_VMM_UNSUPPORTED_ACCESS_RANGE = 1U << 4,
  DYN_VMM_UNSUPPORTED_FORK = 1U << 5,
  DYN_VMM_UNSUPPORTED_RESOLVER_ABI = 1U << 6,
  DYN_VMM_UNSUPPORTED_LOADER_NAMESPACE = 1U << 7,
  DYN_VMM_UNSUPPORTED_UNREDIRECTABLE_LOOKUP = 1U << 8,
  DYN_VMM_UNSUPPORTED_BOOKKEEPING = 1U << 9,
  DYN_VMM_STATE_DETACHED = 1U << 16,
  DYN_VMM_STATE_POISONED = 1U << 17,
};

enum dyn_vmm_phase {
  DYN_VMM_PHASE_ACTIVE = 0,
  DYN_VMM_PHASE_AUDITED = 1,
  DYN_VMM_PHASE_CONTEXTS_SYNCED = 2,
  DYN_VMM_PHASE_LOCK_READY = 3,
  DYN_VMM_PHASE_REATTACHING = 4,
  DYN_VMM_PHASE_ACTIVE_VERIFIED = 5,
  DYN_VMM_PHASE_POISONED = 6,
};

/*
 * Fixed-size packets keep the preload library independent of a serializer.
 * All current deployment targets are little-endian Linux/amd64. The Go peer
 * encodes fields explicitly and both implementations reject other versions.
 */
struct dyn_vmm_packet {
  uint32_t magic;
  uint16_t version;
  uint16_t operation;
  int32_t status;
  uint32_t flags;
  uint64_t generation;
  uint64_t revision;
  uint64_t object_dev;
  uint64_t object_ino;
  uint64_t fresh_dev;
  uint64_t fresh_ino;
  uint64_t address;
  uint64_t size;
  uint64_t offset;
  uint64_t logical_handle;
  uint32_t pid;
  uint32_t count;
  uint32_t role;
  uint32_t mapping_count;
  char message[112];
  uint32_t phase;
  uint32_t outcome;
  uint8_t reserved[24];
};

_Static_assert(sizeof(struct dyn_vmm_packet) == 256, "Dynamo VMM protocol packet size changed");

#endif
