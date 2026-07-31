/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define _GNU_SOURCE

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include "../protocol.h"

#undef cuGetProcAddress

#define SYNTHETIC_MASK 0xffff000000000000ULL
#define SYNTHETIC_TAG 0xd95a000000000000ULL

CUresult CUDAAPI cuGetProcAddress(const char*, void**, int, cuuint64_t);
CUresult CUDAAPI cuGetProcAddress_v2(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);

void fake_cuda_reset(void);
void fake_cuda_fail_after(const char*, long);
unsigned int fake_cuda_create_calls(void);
unsigned int fake_cuda_granularity_calls(void);
unsigned int fake_cuda_release_calls(void);
unsigned int fake_cuda_export_calls(void);
unsigned int fake_cuda_import_calls(void);
unsigned int fake_cuda_map_calls(void);
unsigned int fake_cuda_unmap_calls(void);
unsigned int fake_cuda_resolver4_calls(void);
unsigned int fake_cuda_resolver5_calls(void);
unsigned int fake_cuda_runtime_resolver_calls(void);
unsigned int fake_cuda_sync_calls(void);
unsigned int fake_cuda_multicast_calls(void);
unsigned int fake_cuda_multicast_operation(void);
uint64_t fake_cuda_multicast_argument(unsigned int);
unsigned int fake_cuda_live_objects(void);
unsigned int fake_cuda_live_handles(void);
unsigned int fake_cuda_live_maps(void);
unsigned int fake_cuda_live_reservations(void);
unsigned int fake_cuda_live_references(void);
size_t fake_cuda_map_offset(CUdeviceptr);
CUmemGenericAllocationHandle fake_cuda_last_export_handle(void);
int fake_cuda_last_import_fd(void);
void* fake_cuda_last_resolved_entry(void);

static void
require(bool condition, const char* message)
{
  if (!condition) {
    fprintf(stderr, "FAIL: %s\n", message);
    exit(1);
  }
}

static bool
synthetic(CUmemGenericAllocationHandle handle)
{
  return (handle & SYNTHETIC_MASK) == SYNTHETIC_TAG;
}

typedef CUresult(CUDAAPI* multicast_create_type)(CUmemGenericAllocationHandle*, const CUmulticastObjectProp*);
typedef CUresult(CUDAAPI* multicast_add_device_type)(CUmemGenericAllocationHandle, CUdevice);
typedef CUresult(CUDAAPI* multicast_bind_mem_type)(
    CUmemGenericAllocationHandle, size_t, CUmemGenericAllocationHandle, size_t, size_t, unsigned long long);
typedef CUresult(CUDAAPI* multicast_bind_mem_v2_type)(
    CUmemGenericAllocationHandle, CUdevice, size_t, CUmemGenericAllocationHandle, size_t, size_t, unsigned long long);
typedef CUresult(CUDAAPI* multicast_bind_addr_type)(
    CUmemGenericAllocationHandle, size_t, CUdeviceptr, size_t, unsigned long long);
typedef CUresult(CUDAAPI* multicast_bind_addr_v2_type)(
    CUmemGenericAllocationHandle, CUdevice, size_t, CUdeviceptr, size_t, unsigned long long);
typedef CUresult(CUDAAPI* multicast_unbind_type)(CUmemGenericAllocationHandle, CUdevice, size_t, size_t);
typedef CUresult(CUDAAPI* multicast_granularity_type)(
    size_t*, const CUmulticastObjectProp*, CUmulticastGranularity_flags);

struct multicast_functions {
  multicast_create_type create;
  multicast_add_device_type add_device;
  multicast_bind_mem_type bind_mem;
  multicast_bind_mem_v2_type bind_mem_v2;
  multicast_bind_addr_type bind_addr;
  multicast_bind_addr_v2_type bind_addr_v2;
  multicast_unbind_type unbind;
  multicast_granularity_type granularity;
};

static void
require_multicast_call(unsigned int operation, const uint64_t arguments[7], const char* message)
{
  unsigned int index;

  require(fake_cuda_multicast_operation() == operation, message);
  for (index = 0; index < 7; index++) require(fake_cuda_multicast_argument(index) == arguments[index], message);
}

static void
exercise_dormant_multicast(const struct multicast_functions* functions)
{
  CUmulticastObjectProp properties = {0};
  CUmemGenericAllocationHandle output = 0;
  size_t granularity = 0;
  unsigned int initial_calls = fake_cuda_multicast_calls();
  uint64_t arguments[7];

  require(
      functions->create(&output, &properties) == CUDA_ERROR_UNKNOWN && output == UINT64_C(0xabcdef),
      "dormant multicast create result/output changed");
  memset(arguments, 0, sizeof(arguments));
  arguments[0] = (uint64_t)(uintptr_t)&output;
  arguments[1] = (uint64_t)(uintptr_t)&properties;
  require_multicast_call(1, arguments, "dormant multicast create arguments changed");

  require(
      functions->add_device(UINT64_C(0x11), 2) == CUDA_ERROR_UNKNOWN, "dormant multicast add-device result changed");
  memset(arguments, 0, sizeof(arguments));
  arguments[0] = UINT64_C(0x11);
  arguments[1] = 2;
  require_multicast_call(2, arguments, "dormant multicast add-device arguments changed");

  require(
      functions->bind_mem(UINT64_C(0x12), 0x13, UINT64_C(0x14), 0x15, 0x16, UINT64_C(0x17)) == CUDA_ERROR_UNKNOWN,
      "dormant multicast bind-mem result changed");
  {
    const uint64_t expected[7] = {
        UINT64_C(0x12), UINT64_C(0x13), UINT64_C(0x14), UINT64_C(0x15), UINT64_C(0x16), UINT64_C(0x17), 0,
    };
    require_multicast_call(3, expected, "dormant multicast bind-mem arguments changed");
  }

  require(
      functions->bind_mem_v2(UINT64_C(0x21), 3, 0x22, UINT64_C(0x23), 0x24, 0x25, UINT64_C(0x26)) == CUDA_ERROR_UNKNOWN,
      "dormant multicast bind-mem-v2 result changed");
  {
    const uint64_t expected[7] = {
        UINT64_C(0x21), 3, UINT64_C(0x22), UINT64_C(0x23), UINT64_C(0x24), UINT64_C(0x25), UINT64_C(0x26),
    };
    require_multicast_call(4, expected, "dormant multicast bind-mem-v2 arguments changed");
  }

  require(
      functions->bind_addr(UINT64_C(0x31), 0x32, UINT64_C(0x33), 0x34, UINT64_C(0x35)) == CUDA_ERROR_UNKNOWN,
      "dormant multicast bind-addr result changed");
  {
    const uint64_t expected[7] = {
        UINT64_C(0x31), UINT64_C(0x32), UINT64_C(0x33), UINT64_C(0x34), UINT64_C(0x35), 0, 0,
    };
    require_multicast_call(5, expected, "dormant multicast bind-addr arguments changed");
  }

  require(
      functions->bind_addr_v2(UINT64_C(0x41), 4, 0x42, UINT64_C(0x43), 0x44, UINT64_C(0x45)) == CUDA_ERROR_UNKNOWN,
      "dormant multicast bind-addr-v2 result changed");
  {
    const uint64_t expected[7] = {
        UINT64_C(0x41), 4, UINT64_C(0x42), UINT64_C(0x43), UINT64_C(0x44), UINT64_C(0x45), 0,
    };
    require_multicast_call(6, expected, "dormant multicast bind-addr-v2 arguments changed");
  }

  require(
      functions->unbind(UINT64_C(0x51), 5, 0x52, 0x53) == CUDA_ERROR_UNKNOWN,
      "dormant multicast unbind result changed");
  {
    const uint64_t expected[7] = {
        UINT64_C(0x51), 5, UINT64_C(0x52), UINT64_C(0x53), 0, 0, 0,
    };
    require_multicast_call(7, expected, "dormant multicast unbind arguments changed");
  }

  require(
      functions->granularity(&granularity, &properties, CU_MULTICAST_GRANULARITY_MINIMUM) == CUDA_ERROR_UNKNOWN &&
          granularity == 32768,
      "dormant multicast granularity result/output changed");
  memset(arguments, 0, sizeof(arguments));
  arguments[0] = (uint64_t)(uintptr_t)&granularity;
  arguments[1] = (uint64_t)(uintptr_t)&properties;
  arguments[2] = CU_MULTICAST_GRANULARITY_MINIMUM;
  require_multicast_call(8, arguments, "dormant multicast granularity arguments changed");
  require(fake_cuda_multicast_calls() == initial_calls + 8, "dormant multicast call count changed");
}

static int
connect_agent(void)
{
  struct sockaddr_un address = {
      .sun_family = AF_UNIX,
  };
  const char* directory = getenv("DYN_SNAPSHOT_CONTROL_DIR");
  unsigned int attempt;

  require(directory != NULL, "control directory is unset");
  require(
      snprintf(address.sun_path, sizeof(address.sun_path), "%s/%s%d.sock", directory, DYN_VMM_SOCKET_PREFIX, getpid()) <
          (int)sizeof(address.sun_path),
      "control socket path is too long");
  for (attempt = 0; attempt < 200; attempt++) {
    int fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    require(fd >= 0, "create control socket");
    if (connect(fd, (struct sockaddr*)&address, sizeof(address)) == 0)
      return fd;
    close(fd);
    usleep(10000);
  }
  require(false, "private control endpoint did not become ready");
  return -1;
}

static int
send_packet(int socket_fd, const struct dyn_vmm_packet* packet, int passed_fd)
{
  struct iovec iov = {
      .iov_base = (void*)packet,
      .iov_len = sizeof(*packet),
  };
  char control[CMSG_SPACE(sizeof(int))] = {0};
  struct msghdr message = {
      .msg_iov = &iov,
      .msg_iovlen = 1,
  };

  if (passed_fd >= 0) {
    struct cmsghdr* header;
    message.msg_control = control;
    message.msg_controllen = sizeof(control);
    header = CMSG_FIRSTHDR(&message);
    header->cmsg_level = SOL_SOCKET;
    header->cmsg_type = SCM_RIGHTS;
    header->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(header), &passed_fd, sizeof(passed_fd));
  }
  return sendmsg(socket_fd, &message, MSG_NOSIGNAL) == (ssize_t)sizeof(*packet) ? 0 : -1;
}

static int
receive_packet(int socket_fd, struct dyn_vmm_packet* packet, int* received_fd)
{
  struct iovec iov = {
      .iov_base = packet,
      .iov_len = sizeof(*packet),
  };
  char control[CMSG_SPACE(sizeof(int))] = {0};
  struct msghdr message = {
      .msg_iov = &iov,
      .msg_iovlen = 1,
      .msg_control = control,
      .msg_controllen = sizeof(control),
  };
  struct cmsghdr* header;

  *received_fd = -1;
  if (recvmsg(socket_fd, &message, MSG_CMSG_CLOEXEC) != (ssize_t)sizeof(*packet) ||
      (message.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) != 0)
    return -1;
  header = CMSG_FIRSTHDR(&message);
  if (header != NULL) {
    if (header->cmsg_level != SOL_SOCKET || header->cmsg_type != SCM_RIGHTS ||
        header->cmsg_len != CMSG_LEN(sizeof(int)))
      return -1;
    memcpy(received_fd, CMSG_DATA(header), sizeof(*received_fd));
  }
  return 0;
}

static struct dyn_vmm_packet
control(
    uint16_t operation, uint64_t generation, const struct dyn_vmm_packet* fields, int passed_fd, int* received_fd,
    bool success)
{
  struct dyn_vmm_packet request = {
      .magic = DYN_VMM_MAGIC,
      .version = DYN_VMM_VERSION,
      .operation = operation,
      .generation = generation,
      .pid = (uint32_t)getpid(),
  };
  struct dyn_vmm_packet response;
  int ignored = -1;
  int socket_fd;

  if (fields != NULL) {
    request.object_dev = fields->object_dev;
    request.object_ino = fields->object_ino;
    request.fresh_dev = fields->fresh_dev;
    request.fresh_ino = fields->fresh_ino;
  }
  socket_fd = connect_agent();
  require(send_packet(socket_fd, &request, passed_fd) == 0, "send control packet");
  require(
      receive_packet(socket_fd, &response, received_fd != NULL ? received_fd : &ignored) == 0,
      "receive control packet");
  close(socket_fd);
  if (ignored >= 0)
    close(ignored);
  require(
      response.magic == DYN_VMM_MAGIC && response.version == DYN_VMM_VERSION && response.operation == operation &&
          response.pid == (uint32_t)getpid(),
      "control response envelope mismatch");
  require((response.status == 0) == success, success ? response.message : "control operation unexpectedly succeeded");
  return response;
}

static size_t
audit(uint64_t generation, struct dyn_vmm_packet* records, size_t capacity, struct dyn_vmm_packet* summary)
{
  struct dyn_vmm_packet request = {
      .magic = DYN_VMM_MAGIC,
      .version = DYN_VMM_VERSION,
      .operation = DYN_VMM_OP_AUDIT,
      .generation = generation,
      .pid = (uint32_t)getpid(),
  };
  int socket_fd = connect_agent();
  int received_fd;
  size_t index;

  require(send_packet(socket_fd, &request, -1) == 0, "send audit");
  require(
      receive_packet(socket_fd, summary, &received_fd) == 0 && received_fd < 0 && summary->status == 0,
      "receive audit summary");
  require(summary->count <= capacity, "audit capacity exceeded");
  for (index = 0; index < summary->count; index++) {
    require(
        receive_packet(socket_fd, &records[index], &received_fd) == 0 && received_fd < 0 && records[index].status == 0,
        "receive audit record");
  }
  close(socket_fd);
  return summary->count;
}

static void
require_audit_failure(uint64_t generation)
{
  struct dyn_vmm_packet request = {
      .magic = DYN_VMM_MAGIC,
      .version = DYN_VMM_VERSION,
      .operation = DYN_VMM_OP_AUDIT,
      .generation = generation,
      .pid = (uint32_t)getpid(),
  };
  struct dyn_vmm_packet response;
  int socket_fd = connect_agent();
  int received_fd = -1;

  require(send_packet(socket_fd, &request, -1) == 0, "send audit");
  require(
      receive_packet(socket_fd, &response, &received_fd) == 0 && received_fd < 0 && response.status != 0,
      "unsupported live resource did not fail audit");
  close(socket_fd);
}

static void
test_dormant(void)
{
  struct multicast_functions direct = {
      .create = cuMulticastCreate,
      .add_device = cuMulticastAddDevice,
      .bind_mem = cuMulticastBindMem,
      .bind_mem_v2 = cuMulticastBindMem_v2,
      .bind_addr = cuMulticastBindAddr,
      .bind_addr_v2 = cuMulticastBindAddr_v2,
      .unbind = cuMulticastUnbind,
      .granularity = cuMulticastGetGranularity,
  };
  struct multicast_functions resolver_functions;
  CUmemAllocationProp properties = {0};
  CUmemGenericAllocationHandle handle;
  void* entry = NULL;

  fake_cuda_reset();
  require(cuGetProcAddress("cuMemCreate", &entry, CUDA_VERSION, 0) == CUDA_SUCCESS, "dormant resolver call");
  require(entry == fake_cuda_last_resolved_entry(), "dormant resolver substituted a wrapper");
  require(
      cuMemCreate(&handle, 4096, &properties, 0) == CUDA_SUCCESS && !synthetic(handle),
      "dormant create changed native handle");
  exercise_dormant_multicast(&direct);
#define RESOLVE_MULTICAST(field, type, name)                                                                   \
  do {                                                                                                         \
    entry = NULL;                                                                                              \
    require(                                                                                                   \
        cuGetProcAddress(#name, &entry, 13010, 0) == CUDA_SUCCESS && entry == fake_cuda_last_resolved_entry(), \
        "dormant resolver substituted multicast " #name);                                                      \
    resolver_functions.field = (type)entry;                                                                    \
  } while (0)
  RESOLVE_MULTICAST(create, multicast_create_type, cuMulticastCreate);
  RESOLVE_MULTICAST(add_device, multicast_add_device_type, cuMulticastAddDevice);
  RESOLVE_MULTICAST(bind_mem, multicast_bind_mem_type, cuMulticastBindMem);
  RESOLVE_MULTICAST(bind_mem_v2, multicast_bind_mem_v2_type, cuMulticastBindMem_v2);
  RESOLVE_MULTICAST(bind_addr, multicast_bind_addr_type, cuMulticastBindAddr);
  RESOLVE_MULTICAST(bind_addr_v2, multicast_bind_addr_v2_type, cuMulticastBindAddr_v2);
  RESOLVE_MULTICAST(unbind, multicast_unbind_type, cuMulticastUnbind);
  RESOLVE_MULTICAST(granularity, multicast_granularity_type, cuMulticastGetGranularity);
#undef RESOLVE_MULTICAST
  exercise_dormant_multicast(&resolver_functions);
}

static void
test_resolvers(void)
{
  CUmemAllocationProp properties = {0};
  CUmemGenericAllocationHandle handle;
  CUdriverProcAddressQueryResult driver_status;
  enum cudaDriverEntryPointQueryResult runtime_status;
  void* resolved = NULL;

  fake_cuda_reset();
  require(
      cuMemCreate(&handle, 4096, &properties, 0) == CUDA_SUCCESS && !synthetic(handle),
      "enabled owner handle was not native");
  require(fake_cuda_create_calls() == 1, "real create not called once");
  require(
      cuGetProcAddress("cuMemImportFromShareableHandle", &resolved, CUDA_VERSION, 0) == CUDA_SUCCESS &&
          resolved != fake_cuda_last_resolved_entry() && fake_cuda_resolver4_calls() == 1,
      "4-argument driver resolver bypassed wrapper");
  resolved = NULL;
  require(
      cuGetProcAddress_v2("cuMemMap", &resolved, 12000, 0, &driver_status) == CUDA_SUCCESS &&
          driver_status == CU_GET_PROC_ADDRESS_SUCCESS && resolved != fake_cuda_last_resolved_entry() &&
          fake_cuda_resolver5_calls() == 1,
      "5-argument driver resolver bypassed wrapper");
  resolved = NULL;
  require(
      cudaGetDriverEntryPointByVersion("cuMemRetainAllocationHandle", &resolved, 12000, 0, &runtime_status) ==
              cudaSuccess &&
          runtime_status == cudaDriverEntryPointSuccess && resolved != fake_cuda_last_resolved_entry() &&
          fake_cuda_runtime_resolver_calls() == 1,
      "versioned runtime resolver bypassed wrapper");
  resolved = NULL;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  require(
      cudaGetDriverEntryPoint("cuMemRelease", &resolved, 0, &runtime_status) == cudaSuccess &&
          runtime_status == cudaDriverEntryPointSuccess && resolved != fake_cuda_last_resolved_entry() &&
          fake_cuda_runtime_resolver_calls() == 2,
      "runtime resolver bypassed wrapper");
#pragma GCC diagnostic pop
  require(cuMemRelease(handle) == CUDA_SUCCESS, "owner cleanup");
}

static void
test_explicit_loader(void)
{
  typedef CUresult(CUDAAPI * create_type)(
      CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long);
  CUmemAllocationProp properties = {0};
  CUmemGenericAllocationHandle handle;
  void* library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  create_type create;

  require(library != NULL, "explicit dlopen(libcuda)");
  create = (create_type)dlsym(library, "cuMemCreate");
  require(
      create != NULL && create(&handle, 4096, &properties, 0) == CUDA_SUCCESS && !synthetic(handle),
      "explicit dlsym did not route native owner create");
  create = (create_type)dlvsym(library, "cuMemCreate", "FAKE_CUDA_1.0");
  require(create != NULL, "explicit dlvsym did not resolve wrapper");
  require(cuMemRelease(handle) == CUDA_SUCCESS, "explicit cleanup");
  dlclose(library);
}

static void
test_ordinary_map_contract(void)
{
  CUmemAllocationProp properties = {
      .type = CU_MEM_ALLOCATION_TYPE_PINNED,
      .location =
          {
              .type = CU_MEM_LOCATION_TYPE_DEVICE,
              .id = 0,
          },
      .requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
  };
  CUmemGenericAllocationHandle owner;
  CUmemGenericAllocationHandle imported;
  CUdeviceptr reservation;
  CUdeviceptr owner_va;
  CUdeviceptr peer_va;
  unsigned int maps_before;
  int export_fd;

  fake_cuda_reset();
  require(cuMemAddressReserve(&reservation, 16384, 0, 0, 0) == CUDA_SUCCESS, "map contract reservation");
  owner_va = reservation;
  peer_va = reservation + 8192;
  require(
      cuMemCreate(&owner, 8192, &properties, 0) == CUDA_SUCCESS &&
          cuMemMap(owner_va, 8192, 0, owner, 0) == CUDA_SUCCESS &&
          cuMemExportToShareableHandle(&export_fd, owner, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0) ==
              CUDA_SUCCESS &&
          cuMemImportFromShareableHandle(
              &imported, (void*)(uintptr_t)export_fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) == CUDA_SUCCESS &&
          synthetic(imported),
      "map contract fixture");
  close(export_fd);
  maps_before = fake_cuda_map_calls();
  require(
      cuMemMap(peer_va, 8192, 4096, imported, 0) == CUDA_ERROR_NOT_SUPPORTED,
      "ordinary nonzero-offset map was not rejected");
  require(cuMemMap(peer_va, 4096, 0, imported, 0) == CUDA_ERROR_NOT_SUPPORTED, "ordinary partial map was not rejected");
  require(
      fake_cuda_map_calls() == maps_before && fake_cuda_live_objects() == 1 && fake_cuda_live_handles() == 2 &&
          fake_cuda_live_references() == 2 && fake_cuda_live_maps() == 1 && fake_cuda_live_reservations() == 1 &&
          fake_cuda_map_offset(peer_va) == SIZE_MAX,
      "rejected ordinary maps changed fake CUDA counters or state");
  require(
      cuMemMap(peer_va, 8192, 0, imported, 0) == CUDA_SUCCESS && fake_cuda_map_offset(peer_va) == 0,
      "whole-object zero-offset ordinary map failed");
  require(
      cuMemUnmap(peer_va, 8192) == CUDA_SUCCESS && cuMemRelease(imported) == CUDA_SUCCESS &&
          cuMemUnmap(owner_va, 8192) == CUDA_SUCCESS && cuMemRelease(owner) == CUDA_SUCCESS &&
          cuMemAddressFree(reservation, 16384) == CUDA_SUCCESS,
      "map contract cleanup");
  require(
      fake_cuda_live_objects() == 0 && fake_cuda_live_handles() == 0 && fake_cuda_live_references() == 0 &&
          fake_cuda_live_maps() == 0 && fake_cuda_live_reservations() == 0 &&
          fake_cuda_map_calls() == fake_cuda_unmap_calls(),
      "map contract cleanup leaked fake CUDA state");
}

static void
test_peer_lifecycle(bool reset)
{
  CUmemAllocationProp properties = {
      .type = CU_MEM_ALLOCATION_TYPE_PINNED,
      .location =
          {
              .type = CU_MEM_LOCATION_TYPE_DEVICE,
              .id = 0,
          },
      .requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
  };
  CUmemAccessDesc peer_access = {
      .location =
          {
              .type = CU_MEM_LOCATION_TYPE_DEVICE,
              .id = 1,
          },
      .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
  };
  CUmemGenericAllocationHandle owner;
  CUmemGenericAllocationHandle imported;
  CUmemGenericAllocationHandle retained;
  CUmemAllocationProp imported_properties;
  CUdeviceptr shared_reservation;
  CUdeviceptr owner_va;
  CUdeviceptr peer_va;
  unsigned char pattern[8192];
  unsigned long long access = 0;
  struct dyn_vmm_packet summary;
  struct dyn_vmm_packet records[4];
  struct dyn_vmm_packet owner_record = {0};
  struct dyn_vmm_packet export_response;
  int original_fd;
  int fresh_fd = -1;
  size_t index;

  if (reset)
    fake_cuda_reset();
  memset(pattern, 0x5a, sizeof(pattern));
  require(cuMemAddressReserve(&shared_reservation, 16384, 0, 0, 0) == CUDA_SUCCESS, "shared owner/peer reservation");
  owner_va = shared_reservation;
  peer_va = shared_reservation + 8192;
  require(
      cuMemCreate(&owner, 8192, &properties, 0) == CUDA_SUCCESS && !synthetic(owner) &&
          cuMemMap(owner_va, 8192, 0, owner, 0) == CUDA_SUCCESS &&
          cuMemExportToShareableHandle(&original_fd, owner, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0) ==
              CUDA_SUCCESS &&
          cuMemRelease(owner) == CUDA_SUCCESS,
      "native owner setup");
  memcpy((void*)(uintptr_t)owner_va, pattern, sizeof(pattern));
  require(
      cuMemImportFromShareableHandle(
          &imported, (void*)(uintptr_t)original_fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) == CUDA_SUCCESS &&
          synthetic(imported),
      "peer import did not return a synthetic token");
  require(
      cuMemMap(peer_va, 8192, 0, imported, 0) == CUDA_SUCCESS &&
          cuMemSetAccess(owner_va, 16384, &peer_access, 1) == CUDA_SUCCESS,
      "mixed owner/peer access setup");
  require(
      cuMemGetAllocationPropertiesFromHandle(&imported_properties, imported) == CUDA_SUCCESS &&
          cuMemRetainAllocationHandle(&retained, (void*)(uintptr_t)peer_va) == CUDA_SUCCESS && synthetic(retained),
      "imported handle consumers did not translate");
  close(original_fd);

  require(
      cuMemRelease(SYNTHETIC_TAG | UINT64_C(0x1234)) == CUDA_ERROR_INVALID_HANDLE,
      "unknown synthetic token reached fake CUDA");
  require(audit(55, records, 4, &summary) == 2 && summary.phase == DYN_VMM_PHASE_AUDITED, "resource graph audit");
  for (index = 0; index < summary.count; index++) {
    if (records[index].role == DYN_VMM_ROLE_OWNER)
      owner_record = records[index];
  }
  require(owner_record.object_dev != 0 && owner_record.object_ino != 0, "owner identity missing");
  (void)control(DYN_VMM_OP_SYNC_ALL_CONTEXTS, 55, NULL, -1, NULL, true);
  require(fake_cuda_sync_calls() >= 1, "participating contexts were not synchronized");
  (void)control(DYN_VMM_OP_DETACH_IMPORTS, 55, NULL, -1, NULL, true);
  (void)control(DYN_VMM_OP_VERIFY_LOCK_READY, 55, NULL, -1, NULL, true);
  require(
      fake_cuda_live_maps() == 1 && fake_cuda_live_reservations() == 1 && fake_cuda_live_objects() == 1 &&
          fake_cuda_live_handles() == 0 && memcmp((void*)(uintptr_t)owner_va, pattern, sizeof(pattern)) == 0,
      "detach changed native owner or peer VA reservation");
  require(cuMemRelease(imported) == CUDA_ERROR_INVALID_HANDLE, "detached synthetic token reached fake CUDA");

  export_response = control(DYN_VMM_OP_EXPORT, 55, &owner_record, -1, &fresh_fd, true);
  require(fresh_fd >= 0, "fresh owner export missing SCM_RIGHTS FD");
  (void)control(DYN_VMM_OP_IMPORT, 55, &export_response, fresh_fd, NULL, true);
  close(fresh_fd);
  (void)control(DYN_VMM_OP_VERIFY_ACTIVE, 55, NULL, -1, NULL, true);
  (void)control(DYN_VMM_OP_COMMIT, 55, NULL, -1, NULL, true);
  require(
      fake_cuda_live_maps() == 2 && fake_cuda_live_reservations() == 1 && fake_cuda_live_objects() == 1 &&
          fake_cuda_live_handles() == 2 && cuMemGetAccess(&access, &peer_access.location, peer_va) == CUDA_SUCCESS &&
          access == CU_MEM_ACCESS_FLAGS_PROT_READWRITE &&
          memcmp((void*)(uintptr_t)owner_va, pattern, sizeof(pattern)) == 0 && fake_cuda_import_calls() == 3,
      "fresh import did not restore exact VA/access/logical references");

  require(
      cuMemUnmap(peer_va, 8192) == CUDA_SUCCESS && cuMemRelease(imported) == CUDA_SUCCESS &&
          cuMemRelease(retained) == CUDA_SUCCESS && cuMemUnmap(owner_va, 8192) == CUDA_SUCCESS &&
          cuMemAddressFree(shared_reservation, 16384) == CUDA_SUCCESS,
      "lifecycle cleanup failed");
  require(
      fake_cuda_live_maps() == 0 && fake_cuda_live_reservations() == 0 && fake_cuda_live_handles() == 0 &&
          fake_cuda_live_objects() == 0,
      "lifecycle cleanup leaked native or imported CUDA state");
}

static void
test_release_failure_rollback(void)
{
  CUmemAllocationProp properties = {
      .type = CU_MEM_ALLOCATION_TYPE_PINNED,
      .location =
          {
              .type = CU_MEM_LOCATION_TYPE_DEVICE,
              .id = 0,
          },
      .requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
  };
  CUmemAccessDesc peer_access = {
      .location =
          {
              .type = CU_MEM_LOCATION_TYPE_DEVICE,
              .id = 1,
          },
      .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
  };
  CUmemGenericAllocationHandle owner;
  CUmemGenericAllocationHandle imported;
  CUmemGenericAllocationHandle retained;
  CUdeviceptr reservation;
  CUdeviceptr owner_va;
  CUdeviceptr peer_va;
  unsigned char pattern[8192];
  unsigned long long access = 0;
  struct dyn_vmm_packet summary;
  struct dyn_vmm_packet records[4];
  struct dyn_vmm_packet response;
  int export_fd;

  fake_cuda_reset();
  memset(pattern, 0x6b, sizeof(pattern));
  require(cuMemAddressReserve(&reservation, 16384, 0, 0, 0) == CUDA_SUCCESS, "release rollback reservation");
  owner_va = reservation;
  peer_va = reservation + 8192;
  require(
      cuMemCreate(&owner, 8192, &properties, 0) == CUDA_SUCCESS &&
          cuMemMap(owner_va, 8192, 0, owner, 0) == CUDA_SUCCESS &&
          cuMemExportToShareableHandle(&export_fd, owner, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0) == CUDA_SUCCESS,
      "release rollback native owner fixture");
  require(
      cuMemcpyHtoD_v2(owner_va, pattern, sizeof(pattern)) == CUDA_SUCCESS,
      "release rollback owner pattern initialization");
  require(
      cuMemImportFromShareableHandle(
          &imported, (void*)(uintptr_t)export_fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) == CUDA_SUCCESS &&
          cuMemMap(peer_va, 8192, 0, imported, 0) == CUDA_SUCCESS &&
          cuMemSetAccess(peer_va, 8192, &peer_access, 1) == CUDA_SUCCESS &&
          cuMemRetainAllocationHandle(&retained, (void*)(uintptr_t)peer_va) == CUDA_SUCCESS,
      "release rollback imported peer fixture");
  close(export_fd);
  require(audit(71, records, 4, &summary) == 2 && summary.phase == DYN_VMM_PHASE_AUDITED, "release rollback audit");
  (void)control(DYN_VMM_OP_SYNC_ALL_CONTEXTS, 71, NULL, -1, NULL, true);
  fake_cuda_fail_after("release", 1);
  response = control(DYN_VMM_OP_DETACH_IMPORTS, 71, NULL, -1, NULL, false);
  require(
      response.phase == DYN_VMM_PHASE_CONTEXTS_SYNCED && response.generation == 71 &&
          strstr(response.message, "local peer rollback restored active mappings") != NULL &&
          fake_cuda_live_objects() == 1 && fake_cuda_live_maps() == 2 && fake_cuda_live_handles() == 3 &&
          fake_cuda_live_references() == 3 && fake_cuda_map_calls() == 3 && fake_cuda_unmap_calls() == 1 &&
          cuMemGetAccess(&access, &peer_access.location, peer_va) == CUDA_SUCCESS &&
          access == CU_MEM_ACCESS_FLAGS_PROT_READWRITE &&
          memcmp((void*)(uintptr_t)owner_va, pattern, sizeof(pattern)) == 0,
      "release failure did not restore peer and preserve native owner");
  require(
      audit(71, records, 4, &summary) == 2 && summary.generation == 71 &&
          summary.phase == DYN_VMM_PHASE_CONTEXTS_SYNCED,
      "release rollback state/object graph is not queryable");
  response = control(DYN_VMM_OP_ABORT, 71, NULL, -1, NULL, true);
  require(
      response.generation == 0 && response.phase == DYN_VMM_PHASE_ACTIVE, "release rollback did not reset generation");
  require(
      cuMemUnmap(peer_va, 8192) == CUDA_SUCCESS && cuMemRelease(imported) == CUDA_SUCCESS &&
          cuMemRelease(retained) == CUDA_SUCCESS && cuMemUnmap(owner_va, 8192) == CUDA_SUCCESS &&
          cuMemRelease(owner) == CUDA_SUCCESS && cuMemAddressFree(reservation, 16384) == CUDA_SUCCESS,
      "release rollback cleanup");
  require(
      fake_cuda_live_objects() == 0 && fake_cuda_live_handles() == 0 && fake_cuda_live_references() == 0 &&
          fake_cuda_live_maps() == 0 && fake_cuda_live_reservations() == 0 &&
          fake_cuda_map_calls() == fake_cuda_unmap_calls(),
      "release rollback leaked fake CUDA state");
}

static void
test_multi_object_replay_retry(void)
{
  CUmemAllocationProp properties = {
      .type = CU_MEM_ALLOCATION_TYPE_PINNED,
      .location =
          {
              .type = CU_MEM_LOCATION_TYPE_DEVICE,
              .id = 0,
          },
      .requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
  };
  CUmemAccessDesc peer_access = {
      .location =
          {
              .type = CU_MEM_LOCATION_TYPE_DEVICE,
              .id = 1,
          },
      .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
  };
  CUmemGenericAllocationHandle owners[2];
  CUmemGenericAllocationHandle imports[2];
  CUmemGenericAllocationHandle retained;
  CUdeviceptr reservations[2];
  CUdeviceptr owner_vas[2];
  CUdeviceptr peer_vas[2];
  struct dyn_vmm_packet summary;
  struct dyn_vmm_packet records[8];
  struct dyn_vmm_packet owner_records[2] = {{0}};
  struct dyn_vmm_packet exports[2];
  struct dyn_vmm_packet response;
  struct stat status;
  int original_fds[2];
  int fresh_fds[2] = {-1, -1};
  unsigned int imports_before_retry;
  unsigned int index;

  fake_cuda_reset();
  for (index = 0; index < 2; index++) {
    require(cuMemAddressReserve(&reservations[index], 16384, 0, 0, 0) == CUDA_SUCCESS, "multi-object reservation");
    owner_vas[index] = reservations[index];
    peer_vas[index] = reservations[index] + 8192;
    require(
        cuMemCreate(&owners[index], 8192, &properties, 0) == CUDA_SUCCESS &&
            cuMemMap(owner_vas[index], 8192, 0, owners[index], 0) == CUDA_SUCCESS &&
            cuMemExportToShareableHandle(
                &original_fds[index], owners[index], CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0) == CUDA_SUCCESS &&
            cuMemImportFromShareableHandle(
                &imports[index], (void*)(uintptr_t)original_fds[index], CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) ==
                CUDA_SUCCESS &&
            cuMemMap(peer_vas[index], 8192, 0, imports[index], 0) == CUDA_SUCCESS &&
            cuMemSetAccess(peer_vas[index], 8192, &peer_access, 1) == CUDA_SUCCESS,
        "multi-object fixture");
    require(synthetic(imports[index]), "multi-object import did not return stable synthetic handle");
    require(fstat(original_fds[index], &status) == 0, "multi-object original FD identity");
    owner_records[index].object_dev = (uint64_t)status.st_dev;
    owner_records[index].object_ino = (uint64_t)status.st_ino;
    owner_records[index].role = DYN_VMM_ROLE_OWNER;
    close(original_fds[index]);
  }
  require(
      cuMemRetainAllocationHandle(&retained, (void*)(uintptr_t)peer_vas[1]) == CUDA_SUCCESS,
      "multi-object second logical reference");
  require(audit(73, records, 8, &summary) == 4, "multi-object audit");
  (void)control(DYN_VMM_OP_SYNC_ALL_CONTEXTS, 73, NULL, -1, NULL, true);
  (void)control(DYN_VMM_OP_DETACH_IMPORTS, 73, NULL, -1, NULL, true);
  (void)control(DYN_VMM_OP_VERIFY_LOCK_READY, 73, NULL, -1, NULL, true);

  for (index = 0; index < 2; index++) {
    exports[index] = control(DYN_VMM_OP_EXPORT, 73, &owner_records[index], -1, &fresh_fds[index], true);
    require(fresh_fds[index] >= 0, "multi-object fresh export");
  }
  (void)control(DYN_VMM_OP_IMPORT, 73, &exports[0], fresh_fds[0], NULL, true);
  fake_cuda_fail_after("import", 1);
  (void)control(DYN_VMM_OP_IMPORT, 73, &exports[1], fresh_fds[1], NULL, false);
  imports_before_retry = fake_cuda_import_calls();
  (void)control(DYN_VMM_OP_IMPORT, 73, &exports[0], fresh_fds[0], NULL, true);
  require(fake_cuda_import_calls() == imports_before_retry, "retry re-imported an already-active object");
  (void)control(DYN_VMM_OP_IMPORT, 73, &exports[1], fresh_fds[1], NULL, true);
  close(fresh_fds[0]);
  close(fresh_fds[1]);
  (void)control(DYN_VMM_OP_VERIFY_ACTIVE, 73, NULL, -1, NULL, true);
  response = control(DYN_VMM_OP_ABORT, 73, NULL, -1, NULL, true);
  require(
      response.generation == 0 && response.phase == DYN_VMM_PHASE_ACTIVE && fake_cuda_live_objects() == 2 &&
          fake_cuda_live_maps() == 4 && fake_cuda_live_handles() == 5 && fake_cuda_live_references() == 5 &&
          fake_cuda_import_calls() == 7,
      "multi-object retry did not converge with balanced references");
  for (index = 0; index < 2; index++) {
    unsigned long long access = 0;
    require(
        cuMemGetAccess(&access, &peer_access.location, peer_vas[index]) == CUDA_SUCCESS &&
            access == CU_MEM_ACCESS_FLAGS_PROT_READWRITE && fake_cuda_map_offset(peer_vas[index]) == 0 &&
            synthetic(imports[index]),
        "multi-object retry did not preserve handle/VA/offset/access");
  }
  require(
      cuMemUnmap(peer_vas[0], 8192) == CUDA_SUCCESS && cuMemRelease(imports[0]) == CUDA_SUCCESS &&
          cuMemUnmap(peer_vas[1], 8192) == CUDA_SUCCESS && cuMemRelease(imports[1]) == CUDA_SUCCESS &&
          cuMemRelease(retained) == CUDA_SUCCESS,
      "multi-object peer cleanup");
  for (index = 0; index < 2; index++) {
    require(
        cuMemUnmap(owner_vas[index], 8192) == CUDA_SUCCESS && cuMemRelease(owners[index]) == CUDA_SUCCESS &&
            cuMemAddressFree(reservations[index], 16384) == CUDA_SUCCESS,
        "multi-object owner cleanup");
  }
  require(
      fake_cuda_live_objects() == 0 && fake_cuda_live_handles() == 0 && fake_cuda_live_references() == 0 &&
          fake_cuda_live_maps() == 0 && fake_cuda_live_reservations() == 0 &&
          fake_cuda_map_calls() == fake_cuda_unmap_calls(),
      "multi-object retry leaked fake CUDA state");
}

static void
test_force_posix_negotiation(void)
{
  multicast_create_type multicast_create;
  CUmemAllocationProp fabric = {
      .type = CU_MEM_ALLOCATION_TYPE_PINNED,
      .location =
          {
              .type = CU_MEM_LOCATION_TYPE_DEVICE,
              .id = 0,
          },
      .requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC,
  };
  CUmulticastObjectProp multicast = {0};
  CUmemGenericAllocationHandle handle = 0;
  size_t granularity = 0;
  void* entry = NULL;

  fake_cuda_reset();
  require(
      cuMemGetAllocationGranularity(&granularity, &fabric, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) == CUDA_SUCCESS &&
          granularity == 4096 && fake_cuda_granularity_calls() == 1,
      "PyTorch FABRIC granularity trial did not reach real driver");
  require(
      cuMemCreate(&handle, granularity, &fabric, 0) == CUDA_ERROR_NOT_SUPPORTED && fake_cuda_create_calls() == 0 &&
          fake_cuda_live_objects() == 0,
      "force-POSIX FABRIC create did not reject before allocation");
  require(
      cuGetProcAddress("cuMulticastCreate", &entry, 12010, 0) == CUDA_SUCCESS &&
          entry != fake_cuda_last_resolved_entry(),
      "enabled resolver bypassed multicast rejection");
  multicast_create = (multicast_create_type)entry;
  require(
      multicast_create(&handle, &multicast) == CUDA_ERROR_NOT_SUPPORTED && fake_cuda_multicast_calls() == 0,
      "resolver multicast probe reached the real driver");
  test_peer_lifecycle(false);
}

static void
test_live_fabric_fails_closed(void)
{
  CUmemAllocationProp fabric = {
      .type = CU_MEM_ALLOCATION_TYPE_PINNED,
      .location =
          {
              .type = CU_MEM_LOCATION_TYPE_DEVICE,
              .id = 0,
          },
      .requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC,
  };
  CUmemGenericAllocationHandle handle;

  fake_cuda_reset();
  require(
      cuMemCreate(&handle, 4096, &fabric, 0) == CUDA_SUCCESS && fake_cuda_create_calls() == 1 &&
          fake_cuda_live_objects() == 1,
      "live FABRIC fixture was not created");
  require_audit_failure(91);
  require(cuMemRelease(handle) == CUDA_SUCCESS, "live FABRIC fixture cleanup failed");
}

static void
test_unsupported(void)
{
  typedef CUresult(CUDAAPI * ipc_type)(CUipcMemHandle*, CUdeviceptr);
  typedef CUresult(CUDAAPI * multicast_type)(CUmemGenericAllocationHandle*, const CUmulticastObjectProp*);
  CUmemAllocationProp properties = {
      .type = CU_MEM_ALLOCATION_TYPE_PINNED,
      .location =
          {
              .type = CU_MEM_LOCATION_TYPE_DEVICE,
              .id = 0,
          },
      .requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC,
  };
  CUmemGenericAllocationHandle handle;
  CUipcMemHandle ipc = {{0}};
  CUdeviceptr ptr = 0;
  ipc_type ipc_function = (ipc_type)dlsym(RTLD_DEFAULT, "cuIpcGetMemHandle");
  multicast_type multicast_function = (multicast_type)dlsym(RTLD_DEFAULT, "cuMulticastCreate");

  require(cuMemCreate(&handle, 4096, &properties, 0) == CUDA_ERROR_NOT_SUPPORTED, "FABRIC create was not rejected");
  require(ipc_function != NULL && ipc_function(&ipc, ptr) != CUDA_SUCCESS, "legacy IPC was not rejected");
  require(
      multicast_function != NULL && multicast_function(NULL, NULL) == CUDA_ERROR_NOT_SUPPORTED,
      "multicast was not rejected");
}

static void
test_fork_exec_contract(void)
{
  CUmemAllocationProp properties = {0};
  CUmemGenericAllocationHandle owner;
  pid_t child;
  int status;

  require(cuMemCreate(&owner, 4096, &properties, 0) == CUDA_SUCCESS, "fork fixture create");
  child = fork();
  require(child >= 0, "fork");
  if (child == 0) {
    CUmemGenericAllocationHandle child_owner;
    CUresult result = cuMemCreate(&child_owner, 4096, &properties, 0);
    _exit(result == CUDA_ERROR_NOT_SUPPORTED ? 0 : 1);
  }
  require(waitpid(child, &status, 0) == child, "waitpid");
  require(WIFEXITED(status) && WEXITSTATUS(status) == 0, "CUDA-after-fork did not fail clearly");
  require(cuMemRelease(owner) == CUDA_SUCCESS, "fork fixture cleanup");
}

static void
test_pre_cuda_fork_contract(void)
{
  CUmemAllocationProp properties = {0};
  pid_t child;
  int status;

  child = fork();
  require(child >= 0, "pre-CUDA fork");
  if (child == 0) {
    CUmemGenericAllocationHandle owner;
    CUresult result = cuMemCreate(&owner, 4096, &properties, 0);
    if (result != CUDA_SUCCESS)
      _exit(1);
    _exit(cuMemRelease(owner) == CUDA_SUCCESS ? 0 : 1);
  }
  require(waitpid(child, &status, 0) == child, "pre-CUDA waitpid");
  require(WIFEXITED(status) && WEXITSTATUS(status) == 0, "pre-CUDA fork child could not lazily activate");
}

int
main(int argc, char** argv)
{
  require(argc == 2, "one test mode is required");
  if (strcmp(argv[1], "dormant") == 0)
    test_dormant();
  else if (strcmp(argv[1], "resolvers") == 0)
    test_resolvers();
  else if (strcmp(argv[1], "explicit-loader") == 0)
    test_explicit_loader();
  else if (strcmp(argv[1], "ordinary-map-contract") == 0)
    test_ordinary_map_contract();
  else if (strcmp(argv[1], "peer-lifecycle") == 0)
    test_peer_lifecycle(true);
  else if (strcmp(argv[1], "release-failure-rollback") == 0)
    test_release_failure_rollback();
  else if (strcmp(argv[1], "multi-object-retry") == 0)
    test_multi_object_replay_retry();
  else if (strcmp(argv[1], "force-posix-negotiation") == 0)
    test_force_posix_negotiation();
  else if (strcmp(argv[1], "live-fabric") == 0)
    test_live_fabric_fails_closed();
  else if (strcmp(argv[1], "unsupported") == 0)
    test_unsupported();
  else if (strcmp(argv[1], "fork") == 0)
    test_fork_exec_contract();
  else if (strcmp(argv[1], "pre-cuda-fork") == 0)
    test_pre_cuda_fork_contract();
  else
    require(false, "unknown test mode");
  return 0;
}
