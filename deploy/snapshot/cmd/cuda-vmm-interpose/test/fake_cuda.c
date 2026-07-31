/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define _GNU_SOURCE

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#undef cuGetProcAddress

#define MAX_OBJECTS 64
#define MAX_HANDLES 256
#define MAX_MAPS 256
#define MAX_RESERVATIONS 128
#define MAX_EXPORTS 128
#define MAX_ACCESS 16

struct fake_object {
    bool used;
    unsigned char *bytes;
    size_t size;
    unsigned int refs;
    unsigned int maps;
};

struct fake_handle {
    bool used;
    CUmemGenericAllocationHandle value;
    struct fake_object *object;
};

struct fake_access {
    bool used;
    CUmemLocation location;
    unsigned long long flags;
};

struct fake_map {
    bool used;
    CUdeviceptr ptr;
    size_t size;
    size_t offset;
    CUcontext context;
    struct fake_object *object;
    struct fake_access access[MAX_ACCESS];
};

struct fake_reservation {
    bool used;
    CUdeviceptr ptr;
    size_t size;
};

struct fake_export {
    bool used;
    uint64_t dev;
    uint64_t ino;
    struct fake_object *object;
};

struct failpoint {
    const char *operation;
    long remaining;
};

static struct fake_object objects[MAX_OBJECTS];
static struct fake_handle handles[MAX_HANDLES];
static struct fake_map maps[MAX_MAPS];
static struct fake_reservation reservations[MAX_RESERVATIONS];
static struct fake_export exports[MAX_EXPORTS];
static struct failpoint failpoints[16];
static unsigned int create_calls;
static unsigned int granularity_calls;
static unsigned int release_calls;
static unsigned int export_calls;
static unsigned int import_calls;
static unsigned int map_calls;
static unsigned int unmap_calls;
static unsigned int resolver4_calls;
static unsigned int resolver5_calls;
static unsigned int runtime_resolver_calls;
static unsigned int sync_calls;
static unsigned int multicast_calls;
static unsigned int multicast_operation;
static uint64_t multicast_arguments[7];
static CUmemGenericAllocationHandle next_handle = 0x100;
static CUmemGenericAllocationHandle last_export_handle;
static int last_import_fd = -1;
static void *last_resolved_entry;
static _Thread_local CUcontext current_context = (CUcontext)(uintptr_t)1;

static bool should_fail(const char *operation)
{
    size_t index;

    for (index = 0; index < sizeof(failpoints) / sizeof(failpoints[0]);
         index++) {
        if (failpoints[index].operation == NULL ||
            strcmp(failpoints[index].operation, operation) != 0)
            continue;
        if (failpoints[index].remaining == 0) {
            failpoints[index].operation = NULL;
            return true;
        }
        failpoints[index].remaining--;
        return false;
    }
    return false;
}

void fake_cuda_fail_after(const char *operation, long successes)
{
    size_t index;

    for (index = 0; index < sizeof(failpoints) / sizeof(failpoints[0]);
         index++) {
        if (failpoints[index].operation == NULL) {
            failpoints[index].operation = operation;
            failpoints[index].remaining = successes;
            return;
        }
    }
    abort();
}

static void maybe_destroy(struct fake_object *object)
{
    if (object->refs == 0 && object->maps == 0) {
        free(object->bytes);
        memset(object, 0, sizeof(*object));
    }
}

static struct fake_object *new_object(size_t size)
{
    size_t index;

    for (index = 0; index < MAX_OBJECTS; index++) {
        if (objects[index].used)
            continue;
        objects[index].bytes = calloc(1, size);
        if (objects[index].bytes == NULL)
            return NULL;
        objects[index].used = true;
        objects[index].size = size;
        return &objects[index];
    }
    return NULL;
}

static struct fake_handle *new_handle(struct fake_object *object)
{
    size_t index;

    for (index = 0; index < MAX_HANDLES; index++) {
        if (handles[index].used)
            continue;
        handles[index].used = true;
        handles[index].value = next_handle++;
        handles[index].object = object;
        object->refs++;
        return &handles[index];
    }
    return NULL;
}

static struct fake_handle *find_handle(CUmemGenericAllocationHandle value)
{
    size_t index;

    for (index = 0; index < MAX_HANDLES; index++) {
        if (handles[index].used && handles[index].value == value)
            return &handles[index];
    }
    return NULL;
}

static struct fake_map *find_map(CUdeviceptr ptr)
{
    size_t index;

    for (index = 0; index < MAX_MAPS; index++) {
        if (maps[index].used && ptr >= maps[index].ptr &&
            ptr < maps[index].ptr + maps[index].size)
            return &maps[index];
    }
    return NULL;
}

static void mirror_object(struct fake_object *object)
{
    size_t index;

    for (index = 0; index < MAX_MAPS; index++) {
        if (!maps[index].used || maps[index].object != object)
            continue;
        memcpy((void *)(uintptr_t)maps[index].ptr,
               object->bytes + maps[index].offset, maps[index].size);
    }
}

static struct fake_object *object_for_fd(int fd)
{
    struct stat status;
    size_t index;

    if (fstat(fd, &status) != 0)
        return NULL;
    for (index = 0; index < MAX_EXPORTS; index++) {
        if (exports[index].used &&
            exports[index].dev == (uint64_t)status.st_dev &&
            exports[index].ino == (uint64_t)status.st_ino)
            return exports[index].object;
    }
    return NULL;
}

static struct fake_reservation *reservation_covering(
    CUdeviceptr ptr, size_t size)
{
    size_t index;

    for (index = 0; index < MAX_RESERVATIONS; index++) {
        if (reservations[index].used &&
            ptr >= reservations[index].ptr &&
            ptr + size <=
                reservations[index].ptr + reservations[index].size)
            return &reservations[index];
    }
    return NULL;
}

void fake_cuda_reset(void)
{
    size_t index;

    for (index = 0; index < MAX_RESERVATIONS; index++) {
        if (reservations[index].used)
            munmap((void *)(uintptr_t)reservations[index].ptr,
                   reservations[index].size);
    }
    for (index = 0; index < MAX_OBJECTS; index++)
        free(objects[index].bytes);
    memset(objects, 0, sizeof(objects));
    memset(handles, 0, sizeof(handles));
    memset(maps, 0, sizeof(maps));
    memset(reservations, 0, sizeof(reservations));
    memset(exports, 0, sizeof(exports));
    memset(failpoints, 0, sizeof(failpoints));
    create_calls = 0;
    granularity_calls = 0;
    release_calls = 0;
    export_calls = 0;
    import_calls = 0;
    map_calls = 0;
    unmap_calls = 0;
    resolver4_calls = 0;
    resolver5_calls = 0;
    runtime_resolver_calls = 0;
    sync_calls = 0;
    multicast_calls = 0;
    multicast_operation = 0;
    memset(multicast_arguments, 0, sizeof(multicast_arguments));
    last_export_handle = 0;
    last_import_fd = -1;
    last_resolved_entry = NULL;
    current_context = (CUcontext)(uintptr_t)1;
}

unsigned int fake_cuda_create_calls(void) { return create_calls; }
unsigned int fake_cuda_granularity_calls(void)
{
    return granularity_calls;
}
unsigned int fake_cuda_release_calls(void) { return release_calls; }
unsigned int fake_cuda_export_calls(void) { return export_calls; }
unsigned int fake_cuda_import_calls(void) { return import_calls; }
unsigned int fake_cuda_map_calls(void) { return map_calls; }
unsigned int fake_cuda_unmap_calls(void) { return unmap_calls; }
unsigned int fake_cuda_resolver4_calls(void) { return resolver4_calls; }
unsigned int fake_cuda_resolver5_calls(void) { return resolver5_calls; }
unsigned int fake_cuda_runtime_resolver_calls(void)
{
    return runtime_resolver_calls;
}
unsigned int fake_cuda_sync_calls(void) { return sync_calls; }
unsigned int fake_cuda_multicast_calls(void)
{
    return multicast_calls;
}
unsigned int fake_cuda_multicast_operation(void)
{
    return multicast_operation;
}
uint64_t fake_cuda_multicast_argument(unsigned int index)
{
    return index < 7 ? multicast_arguments[index] : 0;
}
CUmemGenericAllocationHandle fake_cuda_last_export_handle(void)
{
    return last_export_handle;
}
int fake_cuda_last_import_fd(void) { return last_import_fd; }
void *fake_cuda_last_resolved_entry(void) { return last_resolved_entry; }

unsigned int fake_cuda_live_objects(void)
{
    unsigned int result = 0;
    size_t index;
    for (index = 0; index < MAX_OBJECTS; index++)
        result += objects[index].used;
    return result;
}

size_t fake_cuda_map_offset(CUdeviceptr ptr)
{
    struct fake_map *mapping = find_map(ptr);

    return mapping != NULL ? mapping->offset : SIZE_MAX;
}

unsigned int fake_cuda_live_references(void)
{
    unsigned int result = 0;
    size_t index;
    for (index = 0; index < MAX_OBJECTS; index++) {
        if (objects[index].used)
            result += objects[index].refs;
    }
    return result;
}

unsigned int fake_cuda_live_handles(void)
{
    unsigned int result = 0;
    size_t index;
    for (index = 0; index < MAX_HANDLES; index++)
        result += handles[index].used;
    return result;
}

unsigned int fake_cuda_live_maps(void)
{
    unsigned int result = 0;
    size_t index;
    for (index = 0; index < MAX_MAPS; index++)
        result += maps[index].used;
    return result;
}

unsigned int fake_cuda_live_reservations(void)
{
    unsigned int result = 0;
    size_t index;
    for (index = 0; index < MAX_RESERVATIONS; index++)
        result += reservations[index].used;
    return result;
}

CUresult CUDAAPI cuCtxGetCurrent(CUcontext *context)
{
    *context = current_context;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxSetCurrent(CUcontext context)
{
    current_context = context;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxSynchronize(void)
{
    sync_calls++;
    return should_fail("sync") ? CUDA_ERROR_UNKNOWN : CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemAddressReserve(
    CUdeviceptr *ptr, size_t size, size_t alignment, CUdeviceptr address,
    unsigned long long flags)
{
    void *memory;
    size_t index;

    (void)alignment;
    (void)flags;
    if (should_fail("reserve"))
        return CUDA_ERROR_OUT_OF_MEMORY;
    memory = mmap((void *)(uintptr_t)address, size,
                  PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS |
                      (address != 0 ? MAP_FIXED_NOREPLACE : 0),
                  -1, 0);
    if (memory == MAP_FAILED)
        return CUDA_ERROR_OUT_OF_MEMORY;
    for (index = 0; index < MAX_RESERVATIONS; index++) {
        if (!reservations[index].used) {
            reservations[index].used = true;
            reservations[index].ptr = (CUdeviceptr)(uintptr_t)memory;
            reservations[index].size = size;
            *ptr = reservations[index].ptr;
            return CUDA_SUCCESS;
        }
    }
    munmap(memory, size);
    return CUDA_ERROR_OUT_OF_MEMORY;
}

CUresult CUDAAPI cuMemAddressFree(CUdeviceptr ptr, size_t size)
{
    size_t index;

    for (index = 0; index < MAX_RESERVATIONS; index++) {
        if (reservations[index].used &&
            reservations[index].ptr == ptr &&
            reservations[index].size == size) {
            if (munmap((void *)(uintptr_t)ptr, size) != 0)
                return CUDA_ERROR_UNKNOWN;
            memset(&reservations[index], 0,
                   sizeof(reservations[index]));
            return CUDA_SUCCESS;
        }
    }
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult CUDAAPI cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size,
                             const CUmemAllocationProp *properties,
                             unsigned long long flags)
{
    struct fake_object *object;
    struct fake_handle *reference;

    (void)properties;
    (void)flags;
    create_calls++;
    if (should_fail("create"))
        return CUDA_ERROR_OUT_OF_MEMORY;
    object = new_object(size);
    reference = object != NULL ? new_handle(object) : NULL;
    if (reference == NULL) {
        if (object != NULL)
            maybe_destroy(object);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    *handle = reference->value;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemRelease(CUmemGenericAllocationHandle handle)
{
    struct fake_handle *reference = find_handle(handle);
    struct fake_object *object;

    release_calls++;
    if (should_fail("release"))
        return CUDA_ERROR_UNKNOWN;
    if (reference == NULL)
        return CUDA_ERROR_INVALID_HANDLE;
    object = reference->object;
    memset(reference, 0, sizeof(*reference));
    object->refs--;
    maybe_destroy(object);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                          CUmemGenericAllocationHandle handle,
                          unsigned long long flags)
{
    struct fake_handle *reference = find_handle(handle);
    size_t index;

    (void)flags;
    map_calls++;
    if (should_fail("map"))
        return CUDA_ERROR_UNKNOWN;
    if (reference == NULL ||
        offset + size > reference->object->size ||
        reservation_covering(ptr, size) == NULL)
        return CUDA_ERROR_INVALID_VALUE;
    for (index = 0; index < MAX_MAPS; index++) {
        if (!maps[index].used) {
            maps[index].used = true;
            maps[index].ptr = ptr;
            maps[index].size = size;
            maps[index].offset = offset;
            maps[index].context = current_context;
            maps[index].object = reference->object;
            maps[index].object->maps++;
            memcpy((void *)(uintptr_t)ptr,
                   maps[index].object->bytes + offset, size);
            return CUDA_SUCCESS;
        }
    }
    return CUDA_ERROR_OUT_OF_MEMORY;
}

CUresult CUDAAPI cuMemUnmap(CUdeviceptr ptr, size_t size)
{
    size_t index;

    unmap_calls++;
    if (should_fail("unmap"))
        return CUDA_ERROR_UNKNOWN;
    for (index = 0; index < MAX_MAPS; index++) {
        struct fake_object *object;

        if (!maps[index].used || maps[index].ptr != ptr ||
            maps[index].size != size)
            continue;
        object = maps[index].object;
        memcpy(object->bytes + maps[index].offset,
               (void *)(uintptr_t)ptr, size);
        memset((void *)(uintptr_t)ptr, 0, size);
        memset(&maps[index], 0, sizeof(maps[index]));
        object->maps--;
        maybe_destroy(object);
        return CUDA_SUCCESS;
    }
    return CUDA_ERROR_INVALID_VALUE;
}

static struct fake_access *access_for(struct fake_map *mapping,
                                      const CUmemLocation *location,
                                      bool create)
{
    size_t index;

    for (index = 0; index < MAX_ACCESS; index++) {
        if (mapping->access[index].used &&
            mapping->access[index].location.type == location->type &&
            mapping->access[index].location.id == location->id)
            return &mapping->access[index];
    }
    if (!create)
        return NULL;
    for (index = 0; index < MAX_ACCESS; index++) {
        if (!mapping->access[index].used) {
            mapping->access[index].used = true;
            mapping->access[index].location = *location;
            return &mapping->access[index];
        }
    }
    return NULL;
}

CUresult CUDAAPI cuMemSetAccess(CUdeviceptr ptr, size_t size,
                                const CUmemAccessDesc *descriptors,
                                size_t count)
{
    CUdeviceptr end = ptr + size;
    size_t map_index;

    if (should_fail("access"))
        return CUDA_ERROR_UNKNOWN;
    for (map_index = 0; map_index < MAX_MAPS; map_index++) {
        CUdeviceptr intersection_start;
        CUdeviceptr intersection_end;
        size_t descriptor;

        if (!maps[map_index].used)
            continue;
        intersection_start =
            ptr > maps[map_index].ptr ? ptr : maps[map_index].ptr;
        intersection_end =
            end < maps[map_index].ptr + maps[map_index].size
                ? end
                : maps[map_index].ptr + maps[map_index].size;
        if (intersection_start >= intersection_end)
            continue;
        for (descriptor = 0; descriptor < count; descriptor++) {
            struct fake_access *access = access_for(
                &maps[map_index],
                &descriptors[descriptor].location, true);
            if (access == NULL)
                return CUDA_ERROR_OUT_OF_MEMORY;
            access->flags = descriptors[descriptor].flags;
        }
    }
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemGetAccess(unsigned long long *flags,
                                const CUmemLocation *location,
                                CUdeviceptr ptr)
{
    struct fake_map *mapping = find_map(ptr);
    struct fake_access *access;

    if (mapping == NULL)
        return CUDA_ERROR_INVALID_VALUE;
    access = access_for(mapping, location, false);
    *flags = access != NULL ? access->flags
                            : CU_MEM_ACCESS_FLAGS_PROT_NONE;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyHtoD_v2(CUdeviceptr destination,
                                 const void *source, size_t size)
{
    struct fake_map *mapping = find_map(destination);
    size_t offset;

    if (should_fail("copy"))
        return CUDA_ERROR_UNKNOWN;
    if (mapping == NULL ||
        destination + size > mapping->ptr + mapping->size)
        return CUDA_ERROR_INVALID_VALUE;
    offset = mapping->offset + (size_t)(destination - mapping->ptr);
    memcpy(mapping->object->bytes + offset, source, size);
    mirror_object(mapping->object);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyDtoH_v2(void *destination, CUdeviceptr source,
                                 size_t size)
{
    struct fake_map *mapping = find_map(source);

    if (should_fail("copy"))
        return CUDA_ERROR_UNKNOWN;
    if (mapping == NULL ||
        source + size > mapping->ptr + mapping->size)
        return CUDA_ERROR_INVALID_VALUE;
    memcpy(destination, (void *)(uintptr_t)source, size);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemExportToShareableHandle(
    void *shareable, CUmemGenericAllocationHandle handle,
    CUmemAllocationHandleType type, unsigned long long flags)
{
    struct fake_handle *reference = find_handle(handle);
    struct stat status;
    int fd;
    size_t index;

    (void)flags;
    if (type != CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR ||
        reference == NULL)
        return CUDA_ERROR_NOT_SUPPORTED;
    if (should_fail("export"))
        return CUDA_ERROR_UNKNOWN;
    fd = memfd_create("fake-cuda-vmm", MFD_CLOEXEC);
    if (fd < 0 || fstat(fd, &status) != 0) {
        if (fd >= 0)
            close(fd);
        return CUDA_ERROR_UNKNOWN;
    }
    for (index = 0; index < MAX_EXPORTS; index++) {
        if (!exports[index].used) {
            exports[index].used = true;
            exports[index].dev = (uint64_t)status.st_dev;
            exports[index].ino = (uint64_t)status.st_ino;
            exports[index].object = reference->object;
            export_calls++;
            last_export_handle = handle;
            *(int *)shareable = fd;
            return CUDA_SUCCESS;
        }
    }
    close(fd);
    return CUDA_ERROR_OUT_OF_MEMORY;
}

CUresult CUDAAPI cuMemImportFromShareableHandle(
    CUmemGenericAllocationHandle *handle, void *os_handle,
    CUmemAllocationHandleType type)
{
    int fd = (int)(uintptr_t)os_handle;
    struct fake_object *object;
    struct fake_handle *reference;

    if (type != CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)
        return CUDA_ERROR_NOT_SUPPORTED;
    import_calls++;
    last_import_fd = fd;
    if (should_fail("import"))
        return CUDA_ERROR_UNKNOWN;
    object = object_for_fd(fd);
    reference = object != NULL ? new_handle(object) : NULL;
    if (reference == NULL)
        return CUDA_ERROR_INVALID_HANDLE;
    *handle = reference->value;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemRetainAllocationHandle(
    CUmemGenericAllocationHandle *handle, void *address)
{
    struct fake_map *mapping =
        find_map((CUdeviceptr)(uintptr_t)address);
    struct fake_handle *reference;

    if (should_fail("retain"))
        return CUDA_ERROR_UNKNOWN;
    reference = mapping != NULL ? new_handle(mapping->object) : NULL;
    if (reference == NULL)
        return CUDA_ERROR_INVALID_VALUE;
    *handle = reference->value;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemGetAllocationPropertiesFromHandle(
    CUmemAllocationProp *properties,
    CUmemGenericAllocationHandle handle)
{
    if (find_handle(handle) == NULL)
        return CUDA_ERROR_INVALID_HANDLE;
    memset(properties, 0, sizeof(*properties));
    properties->type = CU_MEM_ALLOCATION_TYPE_PINNED;
    properties->location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    properties->requestedHandleTypes =
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemGetAllocationGranularity(
    size_t *granularity, const CUmemAllocationProp *properties,
    CUmemAllocationGranularity_flags option)
{
    (void)properties;
    (void)option;
    granularity_calls++;
    *granularity = 4096;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetAttribute(int *value,
                                      CUdevice_attribute attribute,
                                      CUdevice device)
{
    (void)attribute;
    (void)device;
    *value = 1;
    return CUDA_SUCCESS;
}

static CUresult record_multicast(
    unsigned int operation, uint64_t argument0,
    uint64_t argument1, uint64_t argument2,
    uint64_t argument3, uint64_t argument4,
    uint64_t argument5, uint64_t argument6)
{
    multicast_calls++;
    multicast_operation = operation;
    multicast_arguments[0] = argument0;
    multicast_arguments[1] = argument1;
    multicast_arguments[2] = argument2;
    multicast_arguments[3] = argument3;
    multicast_arguments[4] = argument4;
    multicast_arguments[5] = argument5;
    multicast_arguments[6] = argument6;
    return CUDA_ERROR_UNKNOWN;
}

CUresult CUDAAPI cuMulticastCreate(
    CUmemGenericAllocationHandle *output,
    const CUmulticastObjectProp *properties)
{
    if (output != NULL)
        *output = UINT64_C(0xabcdef);
    return record_multicast(
        1, (uint64_t)(uintptr_t)output,
        (uint64_t)(uintptr_t)properties, 0, 0, 0, 0, 0);
}

CUresult CUDAAPI cuMulticastAddDevice(
    CUmemGenericAllocationHandle handle, CUdevice device)
{
    return record_multicast(
        2, handle, (uint64_t)(unsigned int)device,
        0, 0, 0, 0, 0);
}

CUresult CUDAAPI cuMulticastBindMem(
    CUmemGenericAllocationHandle multicast,
    size_t multicast_offset,
    CUmemGenericAllocationHandle memory,
    size_t memory_offset, size_t size,
    unsigned long long flags)
{
    return record_multicast(
        3, multicast, multicast_offset, memory,
        memory_offset, size, flags, 0);
}

CUresult CUDAAPI cuMulticastBindMem_v2(
    CUmemGenericAllocationHandle multicast, CUdevice device,
    size_t multicast_offset,
    CUmemGenericAllocationHandle memory,
    size_t memory_offset, size_t size,
    unsigned long long flags)
{
    return record_multicast(
        4, multicast, (uint64_t)(unsigned int)device,
        multicast_offset, memory, memory_offset, size, flags);
}

CUresult CUDAAPI cuMulticastBindAddr(
    CUmemGenericAllocationHandle multicast,
    size_t multicast_offset, CUdeviceptr ptr,
    size_t size, unsigned long long flags)
{
    return record_multicast(
        5, multicast, multicast_offset, ptr, size, flags, 0, 0);
}

CUresult CUDAAPI cuMulticastBindAddr_v2(
    CUmemGenericAllocationHandle multicast, CUdevice device,
    size_t multicast_offset, CUdeviceptr ptr,
    size_t size, unsigned long long flags)
{
    return record_multicast(
        6, multicast, (uint64_t)(unsigned int)device,
        multicast_offset, ptr, size, flags, 0);
}

CUresult CUDAAPI cuMulticastUnbind(
    CUmemGenericAllocationHandle multicast,
    CUdevice device, size_t multicast_offset, size_t size)
{
    return record_multicast(
        7, multicast, (uint64_t)(unsigned int)device,
        multicast_offset, size, 0, 0, 0);
}

CUresult CUDAAPI cuMulticastGetGranularity(
    size_t *granularity,
    const CUmulticastObjectProp *properties,
    CUmulticastGranularity_flags option)
{
    if (granularity != NULL)
        *granularity = 32768;
    return record_multicast(
        8, (uint64_t)(uintptr_t)granularity,
        (uint64_t)(uintptr_t)properties,
        (uint64_t)(unsigned int)option, 0, 0, 0, 0);
}

static void *fake_entry_point(const char *symbol)
{
#define ENTRY(name)                                                            \
    if (strcmp(symbol, #name) == 0)                                            \
        return (void *)&name
    ENTRY(cuDeviceGetAttribute);
    ENTRY(cuMemAddressReserve);
    ENTRY(cuMemAddressFree);
    ENTRY(cuMemCreate);
    ENTRY(cuMemRelease);
    ENTRY(cuMemMap);
    ENTRY(cuMemUnmap);
    ENTRY(cuMemSetAccess);
    ENTRY(cuMemGetAccess);
    ENTRY(cuMemExportToShareableHandle);
    ENTRY(cuMemImportFromShareableHandle);
    ENTRY(cuMemGetAllocationGranularity);
    ENTRY(cuMemGetAllocationPropertiesFromHandle);
    ENTRY(cuMemRetainAllocationHandle);
    ENTRY(cuMulticastCreate);
    ENTRY(cuMulticastAddDevice);
    ENTRY(cuMulticastBindMem);
    ENTRY(cuMulticastBindMem_v2);
    ENTRY(cuMulticastBindAddr);
    ENTRY(cuMulticastBindAddr_v2);
    ENTRY(cuMulticastUnbind);
    ENTRY(cuMulticastGetGranularity);
#undef ENTRY
    return NULL;
}

CUresult CUDAAPI cuGetProcAddress(const char *symbol, void **function,
                                  int cuda_version, cuuint64_t flags)
{
    (void)cuda_version;
    (void)flags;
    resolver4_calls++;
    *function = fake_entry_point(symbol);
    last_resolved_entry = *function;
    return *function != NULL ? CUDA_SUCCESS : CUDA_ERROR_NOT_FOUND;
}

CUresult CUDAAPI cuGetProcAddress_v2(
    const char *symbol, void **function, int cuda_version,
    cuuint64_t flags, CUdriverProcAddressQueryResult *status)
{
    (void)cuda_version;
    (void)flags;
    resolver5_calls++;
    *function = fake_entry_point(symbol);
    last_resolved_entry = *function;
    if (status != NULL)
        *status = *function != NULL
                      ? CU_GET_PROC_ADDRESS_SUCCESS
                      : CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    return *function != NULL ? CUDA_SUCCESS : CUDA_ERROR_NOT_FOUND;
}

CUresult CUDAAPI cuGetProcAddress_v2_ptsz(
    const char *symbol, void **function, int cuda_version,
    cuuint64_t flags, CUdriverProcAddressQueryResult *status)
{
    return cuGetProcAddress_v2(
        symbol, function, cuda_version, flags, status);
}

static cudaError_t fake_runtime_entry_point(
    const char *symbol, void **function, unsigned long long flags,
    enum cudaDriverEntryPointQueryResult *status)
{
    (void)flags;
    runtime_resolver_calls++;
    *function = fake_entry_point(symbol);
    last_resolved_entry = *function;
    if (status != NULL)
        *status = *function != NULL
                      ? cudaDriverEntryPointSuccess
                      : cudaDriverEntryPointSymbolNotFound;
    return *function != NULL ? cudaSuccess : cudaErrorNotSupported;
}

cudaError_t CUDARTAPI cudaGetDriverEntryPoint(
    const char *symbol, void **function, unsigned long long flags,
    enum cudaDriverEntryPointQueryResult *status)
{
    return fake_runtime_entry_point(symbol, function, flags, status);
}

cudaError_t CUDARTAPI cudaGetDriverEntryPointByVersion(
    const char *symbol, void **function, unsigned int cuda_version,
    unsigned long long flags,
    enum cudaDriverEntryPointQueryResult *status)
{
    (void)cuda_version;
    return fake_runtime_entry_point(symbol, function, flags, status);
}
