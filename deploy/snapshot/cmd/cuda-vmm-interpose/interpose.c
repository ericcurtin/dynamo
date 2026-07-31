/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define _GNU_SOURCE

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <dlfcn.h>
#include <elf.h>
#include <errno.h>
#include <link.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

#include "protocol.h"

/* Modern CUDA headers alias these source names to versioned ELF symbols. */
#undef cuGetProcAddress
#undef cuIpcOpenMemHandle
#undef cudaGetDriverEntryPoint
#undef cudaGetDriverEntryPointByVersion

#define SYNTHETIC_TAG 0xd95a000000000000ULL
#define SYNTHETIC_MASK 0xffff000000000000ULL
#define DEFAULT_CONTROL_DIR "/snapshot-control"

CUresult CUDAAPI cuGetProcAddress(const char *, void **, int, cuuint64_t);
CUresult CUDAAPI cuGetProcAddress_v2(
    const char *, void **, int, cuuint64_t,
    CUdriverProcAddressQueryResult *);
CUresult CUDAAPI cuGetProcAddress_v2_ptsz(
    const char *, void **, int, cuuint64_t,
    CUdriverProcAddressQueryResult *);
cudaError_t CUDARTAPI cudaGetDriverEntryPoint_ptsz(
    const char *, void **, unsigned long long,
    enum cudaDriverEntryPointQueryResult *);
cudaError_t CUDARTAPI cudaGetDriverEntryPointByVersion_ptsz(
    const char *, void **, unsigned int, unsigned long long,
    enum cudaDriverEntryPointQueryResult *);

struct imported_allocation;

struct logical_handle {
    CUmemGenericAllocationHandle logical;
    CUmemGenericAllocationHandle real;
    struct imported_allocation *allocation;
    struct logical_handle *next;
};

struct access_record {
    CUdeviceptr ptr;
    size_t size;
    CUmemAccessDesc *descriptors;
    size_t descriptor_count;
    struct access_record *next;
};

struct reservation {
    CUdeviceptr ptr;
    size_t size;
    struct reservation *next;
};

struct imported_mapping {
    CUdeviceptr ptr;
    size_t size;
    size_t offset;
    CUcontext context;
    CUdeviceptr reservation_ptr;
    size_t reservation_size;
    bool detached;
    struct imported_allocation *allocation;
    struct access_record *access;
    struct imported_mapping *next;
};

struct imported_allocation {
    uint64_t object_dev;
    uint64_t object_ino;
    struct imported_allocation *next;
};

/* Owners remain native. This record never owns their mappings, handles, or bytes. */
struct owner_allocation {
    CUmemGenericAllocationHandle application_real;
    CUcontext context;
    CUdeviceptr mapped_ptr;
    size_t mapped_size;
    uint64_t object_dev;
    uint64_t object_ino;
    bool exported;
    bool application_released;
    struct owner_allocation *next;
};

struct context_scope {
    CUcontext previous;
    bool changed;
};

struct counters {
    uint64_t owner_exports;
    uint64_t peer_imports;
    uint64_t peer_detached;
    uint64_t peer_reattached;
    uint64_t multicast_attempts;
};

typedef CUresult(CUDAAPI * map_type)(
    CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle,
    unsigned long long);
typedef CUresult(CUDAAPI * unmap_type)(CUdeviceptr, size_t);
typedef CUresult(CUDAAPI * release_type)(CUmemGenericAllocationHandle);
typedef CUresult(CUDAAPI * import_type)(
    CUmemGenericAllocationHandle *, void *, CUmemAllocationHandleType);
typedef CUresult(CUDAAPI * retain_type)(
    CUmemGenericAllocationHandle *, void *);
typedef CUresult(CUDAAPI * set_access_type)(
    CUdeviceptr, size_t, const CUmemAccessDesc *, size_t);

static pthread_mutex_t state_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t explicit_loader_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t loader_once = PTHREAD_ONCE_INIT;
static pthread_once_t agent_once = PTHREAD_ONCE_INIT;
static bool enabled;
static bool force_posix;
static bool fork_child;
static atomic_bool cuda_observed = ATOMIC_VAR_INIT(false);
static enum dyn_vmm_phase phase = DYN_VMM_PHASE_ACTIVE;
static uint64_t generation;
static uint64_t revision;
static uint64_t next_logical = 1;
static uint32_t unsupported;
static char poison_reason[112];
static struct logical_handle *handles;
static struct imported_allocation *imports;
static struct imported_mapping *mappings;
static struct owner_allocation *owners;
static struct reservation *reservations;
static struct counters stats;
static int listener_fd = -1;
static char socket_path[sizeof(((struct sockaddr_un *)0)->sun_path)];
static void *(*real_dlsym)(void *, const char *);
static void *(*real_dlvsym)(void *, const char *, const char *);
static void *(*real_dlmopen)(Lmid_t, const char *, int);
static void *explicit_libcuda;
static void *explicit_libcudart;

static void *replacement_for_symbol(const char *symbol, int cuda_version);
static void *resolve_next(const char *symbol);
static void ensure_agent_started(void);
static void poison(const char *reason);
static int enter_context(
    CUcontext context, struct context_scope *scope);
static int leave_context(struct context_scope *scope);

static bool env_enabled(const char *name)
{
    const char *value = getenv(name);
    return value != NULL && strcmp(value, "1") == 0;
}

static void maybe_remove_owner(struct owner_allocation *owner)
{
    struct owner_allocation **cursor;

    if (owner == NULL || !owner->application_released ||
        owner->mapped_ptr != 0)
        return;
    cursor = &owners;
    while (*cursor != NULL) {
        if (*cursor == owner) {
            *cursor = owner->next;
            free(owner);
            return;
        }
        cursor = &(*cursor)->next;
    }
}

static void log_message(const char *format, ...)
{
    va_list args;

    if (!enabled)
        return;
    fprintf(stderr, "dynamo-snapshot-cuda-vmm[%d]: ", getpid());
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
    fputc('\n', stderr);
}

static void mark_unsupported(uint32_t flag, const char *reason)
{
    unsupported |= flag;
    if (reason != NULL && poison_reason[0] == '\0')
        snprintf(poison_reason, sizeof(poison_reason), "%s", reason);
    revision++;
}

static void poison(const char *reason)
{
    mark_unsupported(DYN_VMM_UNSUPPORTED_BOOKKEEPING, reason);
    phase = DYN_VMM_PHASE_POISONED;
}

static int fd_identity(int fd, uint64_t *dev, uint64_t *ino)
{
    struct stat status;

    if (fd < 0 || fstat(fd, &status) != 0)
        return -1;
    *dev = (uint64_t)status.st_dev;
    *ino = (uint64_t)status.st_ino;
    return 0;
}

struct loader_symbols {
    const char *wanted[3];
    void *found[3];
};

static int find_loader_symbols(
    struct dl_phdr_info *info, size_t size, void *data)
{
    struct loader_symbols *symbols = data;
    const ElfW(Phdr) *header;
    ElfW(Dyn) *dynamic = NULL;
    ElfW(Sym) *symbol_table = NULL;
    const char *string_table = NULL;
    uint32_t *hash = NULL;
    size_t index;

    (void)size;
    if (info->dlpi_name == NULL ||
        (strstr(info->dlpi_name, "/libc.so.") == NULL &&
         strstr(info->dlpi_name, "/libdl.so.") == NULL))
        return 0;
    for (index = 0; index < info->dlpi_phnum; index++) {
        header = &info->dlpi_phdr[index];
        if (header->p_type == PT_DYNAMIC) {
            dynamic = (ElfW(Dyn) *)(info->dlpi_addr + header->p_vaddr);
            break;
        }
    }
    if (dynamic == NULL)
        return 0;
    for (; dynamic->d_tag != DT_NULL; dynamic++) {
        switch (dynamic->d_tag) {
        case DT_SYMTAB:
            symbol_table = (ElfW(Sym) *)dynamic->d_un.d_ptr;
            break;
        case DT_STRTAB:
            string_table = (const char *)dynamic->d_un.d_ptr;
            break;
        case DT_HASH:
            hash = (uint32_t *)dynamic->d_un.d_ptr;
            break;
        default:
            break;
        }
    }
    if (symbol_table == NULL || string_table == NULL || hash == NULL)
        return 0;
    for (index = 0; index < hash[1]; index++) {
        const ElfW(Sym) *symbol = &symbol_table[index];
        size_t wanted;

        if (symbol->st_shndx == SHN_UNDEF ||
            ELF64_ST_BIND(symbol->st_info) == STB_LOCAL)
            continue;
        for (wanted = 0; wanted < 3; wanted++) {
            if (symbols->found[wanted] == NULL &&
                strcmp(string_table + symbol->st_name,
                       symbols->wanted[wanted]) == 0)
                symbols->found[wanted] =
                    (void *)(info->dlpi_addr + symbol->st_value);
        }
    }
    return symbols->found[0] != NULL && symbols->found[1] != NULL &&
                   symbols->found[2] != NULL
               ? 1
               : 0;
}

static void initialize_loader(void)
{
    struct loader_symbols symbols = {
        .wanted = {"dlsym", "dlvsym", "dlmopen"},
    };

    (void)dl_iterate_phdr(find_loader_symbols, &symbols);
    real_dlsym = (void *(*)(void *, const char *))symbols.found[0];
    real_dlvsym =
        (void *(*)(void *, const char *, const char *))symbols.found[1];
    real_dlmopen =
        (void *(*)(Lmid_t, const char *, int))symbols.found[2];
}

enum explicit_cuda_loader {
    EXPLICIT_CUDA_NONE,
    EXPLICIT_CUDA_DEFAULT,
    EXPLICIT_CUDA_DRIVER,
    EXPLICIT_CUDA_RUNTIME,
};

static enum explicit_cuda_loader classify_explicit_cuda_handle(
    void *handle, const char **path)
{
    struct link_map *map = NULL;

    *path = NULL;
    if (handle == RTLD_DEFAULT)
        return EXPLICIT_CUDA_DEFAULT;
    if (handle == NULL || handle == RTLD_NEXT ||
        dlinfo(handle, RTLD_DI_LINKMAP, &map) != 0 || map == NULL ||
        map->l_name == NULL)
        return EXPLICIT_CUDA_NONE;
    *path = map->l_name;
    if (strstr(map->l_name, "libcuda.so") != NULL)
        return EXPLICIT_CUDA_DRIVER;
    if (strstr(map->l_name, "libcudart.so") != NULL)
        return EXPLICIT_CUDA_RUNTIME;
    return EXPLICIT_CUDA_NONE;
}

static bool retain_explicit_cuda_handle(
    enum explicit_cuda_loader loader, const char *path)
{
    void **slot;
    void *retained;

    if (loader == EXPLICIT_CUDA_DEFAULT)
        return true;
    slot = loader == EXPLICIT_CUDA_DRIVER
               ? &explicit_libcuda
               : &explicit_libcudart;
    pthread_mutex_lock(&explicit_loader_lock);
    if (*slot != NULL) {
        pthread_mutex_unlock(&explicit_loader_lock);
        return true;
    }
    retained = dlopen(path, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
    if (retained != NULL)
        *slot = retained;
    pthread_mutex_unlock(&explicit_loader_lock);
    if (retained != NULL)
        return true;
    pthread_mutex_lock(&state_lock);
    mark_unsupported(
        DYN_VMM_UNSUPPORTED_UNREDIRECTABLE_LOOKUP,
        "explicit CUDA handle could not be retained");
    pthread_mutex_unlock(&state_lock);
    return false;
}

static void *resolve_next(const char *symbol)
{
    void *result;
    void *driver;
    void *runtime;

    pthread_once(&loader_once, initialize_loader);
    result = real_dlsym != NULL ? real_dlsym(RTLD_NEXT, symbol) : NULL;
    if (result == NULL && real_dlsym != NULL) {
        pthread_mutex_lock(&explicit_loader_lock);
        driver = explicit_libcuda;
        runtime = explicit_libcudart;
        pthread_mutex_unlock(&explicit_loader_lock);
        if (driver != NULL)
            result = real_dlsym(driver, symbol);
        if (result == NULL && runtime != NULL)
            result = real_dlsym(runtime, symbol);
    }
    return result;
}

struct loader_lookup {
    void *value;
    uintptr_t handled;
};

__attribute__((used, noinline))
static struct loader_lookup intercept_dlsym(
    void *handle, const char *symbol)
{
    struct loader_lookup lookup = {0};
    void *result;
    void *replacement;
    const char *path;
    enum explicit_cuda_loader loader;

    pthread_once(&loader_once, initialize_loader);
    if (real_dlsym == NULL) {
        lookup.handled = 1;
        return lookup;
    }
    if (!enabled)
        return lookup;
    if (handle == RTLD_NEXT) {
        if (replacement_for_symbol(symbol, CUDA_VERSION) != NULL) {
            pthread_mutex_lock(&state_lock);
            mark_unsupported(
                DYN_VMM_UNSUPPORTED_UNREDIRECTABLE_LOOKUP,
                "managed RTLD_NEXT lookup cannot preserve caller scope");
            pthread_mutex_unlock(&state_lock);
        }
        return lookup;
    }
    replacement = replacement_for_symbol(symbol, CUDA_VERSION);
    loader = classify_explicit_cuda_handle(handle, &path);
    if (loader == EXPLICIT_CUDA_NONE || replacement == NULL)
        return lookup;
    result = real_dlsym(handle, symbol);
    lookup.value = result != NULL &&
                           retain_explicit_cuda_handle(loader, path)
                       ? replacement
                       : result;
    lookup.handled = 1;
    return lookup;
}

__attribute__((used, noinline))
static struct loader_lookup intercept_dlvsym(
    void *handle, const char *symbol, const char *version)
{
    struct loader_lookup lookup = {0};
    void *result;
    void *replacement;
    const char *path;
    enum explicit_cuda_loader loader;

    pthread_once(&loader_once, initialize_loader);
    if (real_dlvsym == NULL) {
        lookup.handled = 1;
        return lookup;
    }
    if (!enabled)
        return lookup;
    if (handle == RTLD_NEXT) {
        if (replacement_for_symbol(symbol, CUDA_VERSION) != NULL) {
            pthread_mutex_lock(&state_lock);
            mark_unsupported(
                DYN_VMM_UNSUPPORTED_UNREDIRECTABLE_LOOKUP,
                "managed RTLD_NEXT lookup cannot preserve caller scope");
            pthread_mutex_unlock(&state_lock);
        }
        return lookup;
    }
    replacement = replacement_for_symbol(symbol, CUDA_VERSION);
    loader = classify_explicit_cuda_handle(handle, &path);
    if (loader == EXPLICIT_CUDA_NONE || replacement == NULL)
        return lookup;
    result = real_dlvsym(handle, symbol, version);
    lookup.value = result != NULL &&
                           retain_explicit_cuda_handle(loader, path)
                       ? replacement
                       : result;
    lookup.handled = 1;
    return lookup;
}

#if !defined(__x86_64__)
#error "CUDA VMM loader interposition currently requires Linux/amd64"
#endif

__attribute__((naked, visibility("default")))
void *dlsym(void *, const char *)
{
    __asm__ volatile(
        "push %rdi\n"
        "push %rsi\n"
        "sub $8, %rsp\n"
        "call intercept_dlsym\n"
        "add $8, %rsp\n"
        "pop %rsi\n"
        "pop %rdi\n"
        "test %rdx, %rdx\n"
        "jz 1f\n"
        "ret\n"
        "1: jmp *real_dlsym(%rip)\n");
}

__attribute__((naked, visibility("default")))
void *dlvsym(void *, const char *, const char *)
{
    __asm__ volatile(
        "push %rdi\n"
        "push %rsi\n"
        "push %rdx\n"
        "call intercept_dlvsym\n"
        "mov %rdx, %r11\n"
        "pop %rdx\n"
        "pop %rsi\n"
        "pop %rdi\n"
        "test %r11, %r11\n"
        "jz 1f\n"
        "ret\n"
        "1: jmp *real_dlvsym(%rip)\n");
}

void *dlmopen(Lmid_t namespace_id, const char *filename, int flags)
{
    pthread_once(&loader_once, initialize_loader);
    if (enabled && namespace_id != LM_ID_BASE) {
        pthread_mutex_lock(&state_lock);
        mark_unsupported(
            DYN_VMM_UNSUPPORTED_LOADER_NAMESPACE,
            "non-base loader namespaces are unsupported");
        pthread_mutex_unlock(&state_lock);
    }
    return real_dlmopen != NULL
               ? real_dlmopen(namespace_id, filename, flags)
               : NULL;
}

static CUresult unavailable(void)
{
    return CUDA_ERROR_NOT_SUPPORTED;
}

static bool is_synthetic(CUmemGenericAllocationHandle handle)
{
    return (handle & SYNTHETIC_MASK) == SYNTHETIC_TAG;
}

static struct logical_handle *find_handle(
    CUmemGenericAllocationHandle logical)
{
    struct logical_handle *current;

    for (current = handles; current != NULL; current = current->next) {
        if (current->logical == logical)
            return current;
    }
    return NULL;
}

static struct owner_allocation *find_owner(
    CUmemGenericAllocationHandle real)
{
    struct owner_allocation *owner;

    for (owner = owners; owner != NULL; owner = owner->next) {
        if (!owner->application_released &&
            owner->application_real == real)
            return owner;
    }
    return NULL;
}

static struct owner_allocation *find_owner_by_id(
    uint64_t dev, uint64_t ino)
{
    struct owner_allocation *owner;

    for (owner = owners; owner != NULL; owner = owner->next) {
        if (owner->exported && owner->object_dev == dev &&
            owner->object_ino == ino)
            return owner;
    }
    return NULL;
}

static struct imported_allocation *find_import(
    uint64_t dev, uint64_t ino)
{
    struct imported_allocation *allocation;

    for (allocation = imports; allocation != NULL;
         allocation = allocation->next) {
        if (allocation->object_dev == dev &&
            allocation->object_ino == ino)
            return allocation;
    }
    return NULL;
}

static struct reservation *find_reservation(
    CUdeviceptr ptr, size_t size)
{
    struct reservation *reservation;

    if (ptr > UINT64_MAX - size)
        return NULL;
    for (reservation = reservations; reservation != NULL;
         reservation = reservation->next) {
        if (ptr >= reservation->ptr &&
            ptr + size <= reservation->ptr + reservation->size)
            return reservation;
    }
    return NULL;
}

static struct imported_mapping *find_mapping_covering(
    CUdeviceptr ptr)
{
    struct imported_mapping *mapping;

    for (mapping = mappings; mapping != NULL; mapping = mapping->next) {
        if (ptr >= mapping->ptr && ptr < mapping->ptr + mapping->size)
            return mapping;
    }
    return NULL;
}

static CUmemGenericAllocationHandle new_logical_handle(void)
{
    CUmemGenericAllocationHandle candidate;

    do {
        candidate =
            SYNTHETIC_TAG | (next_logical++ & ~SYNTHETIC_MASK);
    } while ((candidate & ~SYNTHETIC_MASK) == 0 ||
             find_handle(candidate) != NULL ||
             find_owner(candidate) != NULL);
    return candidate;
}

static CUresult translate_import(
    CUmemGenericAllocationHandle logical,
    CUmemGenericAllocationHandle *real,
    struct imported_allocation **allocation)
{
    struct logical_handle *handle;

    if (!is_synthetic(logical)) {
        *real = logical;
        if (allocation != NULL)
            *allocation = NULL;
        return CUDA_SUCCESS;
    }
    handle = find_handle(logical);
    if (handle == NULL || handle->real == 0)
        return CUDA_ERROR_INVALID_HANDLE;
    *real = handle->real;
    if (allocation != NULL)
        *allocation = handle->allocation;
    return CUDA_SUCCESS;
}

static void free_access(struct access_record *access)
{
    while (access != NULL) {
        struct access_record *next = access->next;
        free(access->descriptors);
        free(access);
        access = next;
    }
}

static bool import_is_referenced(
    struct imported_allocation *allocation)
{
    struct logical_handle *handle;
    struct imported_mapping *mapping;

    for (handle = handles; handle != NULL; handle = handle->next) {
        if (handle->allocation == allocation)
            return true;
    }
    for (mapping = mappings; mapping != NULL; mapping = mapping->next) {
        if (mapping->allocation == allocation)
            return true;
    }
    return false;
}

static void maybe_remove_import(
    struct imported_allocation *allocation)
{
    struct imported_allocation **cursor;

    if (allocation == NULL || import_is_referenced(allocation))
        return;
    cursor = &imports;
    while (*cursor != NULL) {
        if (*cursor == allocation) {
            *cursor = allocation->next;
            free(allocation);
            return;
        }
        cursor = &(*cursor)->next;
    }
}

static int enter_context(CUcontext context, struct context_scope *scope)
{
    typedef CUresult(CUDAAPI * get_type)(CUcontext *);
    typedef CUresult(CUDAAPI * set_type)(CUcontext);
    get_type get_function = (get_type)resolve_next("cuCtxGetCurrent");
    set_type set_function = (set_type)resolve_next("cuCtxSetCurrent");

    memset(scope, 0, sizeof(*scope));
    if (context == NULL || get_function == NULL ||
        get_function(&scope->previous) != CUDA_SUCCESS)
        return -1;
    if (scope->previous == context)
        return 0;
    if (set_function == NULL ||
        set_function(context) != CUDA_SUCCESS)
        return -1;
    scope->changed = true;
    return 0;
}

static int leave_context(struct context_scope *scope)
{
    typedef CUresult(CUDAAPI * set_type)(CUcontext);
    set_type set_function = (set_type)resolve_next("cuCtxSetCurrent");

    if (!scope->changed)
        return 0;
    return set_function != NULL &&
                   set_function(scope->previous) == CUDA_SUCCESS
               ? 0
               : -1;
}

static int current_context(CUcontext *context)
{
    typedef CUresult(CUDAAPI * function_type)(CUcontext *);
    function_type function =
        (function_type)resolve_next("cuCtxGetCurrent");

    return function != NULL &&
                   function(context) == CUDA_SUCCESS &&
                   *context != NULL
               ? 0
               : -1;
}

static int synchronize_context(CUcontext context)
{
    typedef CUresult(CUDAAPI * function_type)(void);
    function_type function =
        (function_type)resolve_next("cuCtxSynchronize");
    struct context_scope scope;
    int result;

    if (function == NULL || enter_context(context, &scope) != 0)
        return -1;
    result = function() == CUDA_SUCCESS ? 0 : -1;
    if (leave_context(&scope) != 0)
        return -1;
    return result;
}

static int scoped_map(
    map_type function, struct imported_mapping *mapping,
    CUmemGenericAllocationHandle real)
{
    struct context_scope scope;
    int result;

    if (function == NULL ||
        enter_context(mapping->context, &scope) != 0)
        return -1;
    result = function(
                 mapping->ptr, mapping->size, mapping->offset,
                 real, 0) == CUDA_SUCCESS
                 ? 0
                 : -1;
    if (leave_context(&scope) != 0)
        return -1;
    return result;
}

static int scoped_unmap(
    unmap_type function, struct imported_mapping *mapping)
{
    struct context_scope scope;
    int result;

    if (function == NULL ||
        enter_context(mapping->context, &scope) != 0)
        return -1;
    result = function(mapping->ptr, mapping->size) == CUDA_SUCCESS
                 ? 0
                 : -1;
    if (leave_context(&scope) != 0)
        return -1;
    return result;
}

static int restore_access(struct imported_mapping *mapping)
{
    set_access_type function =
        (set_access_type)resolve_next("cuMemSetAccess");
    struct access_record *access;
    struct context_scope scope;
    int result = 0;

    if (mapping->access == NULL)
        return 0;
    if (function == NULL ||
        enter_context(mapping->context, &scope) != 0)
        return -1;
    for (access = mapping->access; access != NULL;
         access = access->next) {
        if (function(
                access->ptr, access->size, access->descriptors,
                access->descriptor_count) != CUDA_SUCCESS) {
            result = -1;
            break;
        }
    }
    if (leave_context(&scope) != 0)
        return -1;
    return result;
}

static unsigned long long effective_access(
    struct imported_mapping *mapping, CUdeviceptr ptr,
    const CUmemLocation *location)
{
    struct access_record *access;
    unsigned long long result = CU_MEM_ACCESS_FLAGS_PROT_NONE;

    for (access = mapping->access; access != NULL;
         access = access->next) {
        size_t index;

        if (ptr < access->ptr || ptr >= access->ptr + access->size)
            continue;
        for (index = 0; index < access->descriptor_count; index++) {
            if (access->descriptors[index].location.type ==
                    location->type &&
                access->descriptors[index].location.id == location->id)
                result = access->descriptors[index].flags;
        }
    }
    return result;
}

static int verify_access(struct imported_mapping *mapping)
{
    typedef CUresult(CUDAAPI * function_type)(
        unsigned long long *, const CUmemLocation *, CUdeviceptr);
    function_type function =
        (function_type)resolve_next("cuMemGetAccess");
    struct access_record *access;
    struct context_scope scope;
    int result = 0;

    if (mapping->access == NULL)
        return 0;
    if (function == NULL ||
        enter_context(mapping->context, &scope) != 0)
        return -1;
    for (access = mapping->access; access != NULL;
         access = access->next) {
        size_t index;
        CUdeviceptr last = access->ptr + access->size - 1;

        for (index = 0; index < access->descriptor_count; index++) {
            unsigned long long first_flags = 0;
            unsigned long long last_flags = 0;
            CUmemLocation *location =
                &access->descriptors[index].location;
            if (function(&first_flags, location, access->ptr) !=
                    CUDA_SUCCESS ||
                function(&last_flags, location, last) != CUDA_SUCCESS ||
                first_flags !=
                    effective_access(mapping, access->ptr, location) ||
                last_flags !=
                    effective_access(mapping, last, location)) {
                result = -1;
                goto done;
            }
        }
    }
done:
    if (leave_context(&scope) != 0)
        return -1;
    return result;
}

static void packet_error(
    struct dyn_vmm_packet *packet, const char *format, ...)
{
    va_list args;

    packet->status = -1;
    va_start(args, format);
    vsnprintf(packet->message, sizeof(packet->message), format, args);
    va_end(args);
}

static uint32_t mapping_count(
    struct imported_allocation *allocation)
{
    struct imported_mapping *mapping;
    uint32_t count = 0;

    for (mapping = mappings; mapping != NULL; mapping = mapping->next) {
        if (mapping->allocation == allocation)
            count++;
    }
    return count;
}

static void set_packet_state(struct dyn_vmm_packet *packet)
{
    packet->generation = generation;
    packet->revision = revision;
    packet->pid = (uint32_t)getpid();
    packet->phase = phase;
    packet->flags |= unsupported;
    if (phase == DYN_VMM_PHASE_LOCK_READY ||
        phase == DYN_VMM_PHASE_REATTACHING)
        packet->flags |= DYN_VMM_STATE_DETACHED;
    if (phase == DYN_VMM_PHASE_POISONED)
        packet->flags |= DYN_VMM_STATE_POISONED;
}

static int send_packet(
    int fd, const struct dyn_vmm_packet *packet, int passed_fd)
{
    struct iovec iov = {
        .iov_base = (void *)packet,
        .iov_len = sizeof(*packet),
    };
    char control[CMSG_SPACE(sizeof(int))] = {0};
    struct msghdr message = {
        .msg_iov = &iov,
        .msg_iovlen = 1,
    };

    if (passed_fd >= 0) {
        struct cmsghdr *header;
        message.msg_control = control;
        message.msg_controllen = sizeof(control);
        header = CMSG_FIRSTHDR(&message);
        header->cmsg_level = SOL_SOCKET;
        header->cmsg_type = SCM_RIGHTS;
        header->cmsg_len = CMSG_LEN(sizeof(int));
        memcpy(CMSG_DATA(header), &passed_fd, sizeof(passed_fd));
    }
    return sendmsg(fd, &message, MSG_NOSIGNAL) ==
                   (ssize_t)sizeof(*packet)
               ? 0
               : -1;
}

static int receive_packet(
    int fd, struct dyn_vmm_packet *packet, int *received_fd)
{
    struct iovec iov = {
        .iov_base = packet,
        .iov_len = sizeof(*packet),
    };
    char control[CMSG_SPACE(4 * sizeof(int))] = {0};
    struct msghdr message = {
        .msg_iov = &iov,
        .msg_iovlen = 1,
        .msg_control = control,
        .msg_controllen = sizeof(control),
    };
    struct cmsghdr *header;
    ssize_t size;
    bool invalid = false;

    *received_fd = -1;
    size = recvmsg(fd, &message, MSG_CMSG_CLOEXEC);
    for (header = CMSG_FIRSTHDR(&message); header != NULL;
         header = CMSG_NXTHDR(&message, header)) {
        size_t payload;
        size_t count;
        size_t index;
        int *fds;

        if (header->cmsg_len < CMSG_LEN(0) ||
            header->cmsg_len > message.msg_controllen) {
            invalid = true;
            continue;
        }
        payload = header->cmsg_len - CMSG_LEN(0);
        count = payload / sizeof(int);
        fds = (int *)CMSG_DATA(header);
        if (header->cmsg_level != SOL_SOCKET ||
            header->cmsg_type != SCM_RIGHTS || payload == 0 ||
            payload % sizeof(int) != 0) {
            invalid = true;
            continue;
        }
        for (index = 0; index < count; index++) {
            if (*received_fd < 0 && count == 1 && !invalid)
                *received_fd = fds[index];
            else {
                close(fds[index]);
                invalid = true;
            }
        }
    }
    if (size != (ssize_t)sizeof(*packet) ||
        (message.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) != 0)
        invalid = true;
    if (!invalid)
        return 0;
    if (*received_fd >= 0) {
        close(*received_fd);
        *received_fd = -1;
    }
    return -1;
}

static int audit_local(struct dyn_vmm_packet *response)
{
    struct imported_allocation *allocation;
    struct owner_allocation *owner;

    if (fork_child)
        mark_unsupported(
            DYN_VMM_UNSUPPORTED_FORK,
            "CUDA VMM state was inherited across fork; exec is required");
    if (unsupported != 0 || phase == DYN_VMM_PHASE_POISONED) {
        packet_error(
            response, "%s%sunsupported CUDA state flags=0x%x",
            poison_reason[0] != '\0' ? poison_reason : "",
            poison_reason[0] != '\0' ? "; " : "", unsupported);
        return -1;
    }
    for (owner = owners; owner != NULL; owner = owner->next) {
        if (owner->exported) {
            if (owner->application_released && owner->mapped_ptr == 0) {
                packet_error(
                    response,
                    "released exported owner has no native mapping to retain");
                return -1;
            }
            response->count++;
        }
    }
    for (allocation = imports; allocation != NULL;
         allocation = allocation->next) {
        response->count++;
    }
    return 0;
}

static int handle_audit(
    const struct dyn_vmm_packet *request, int client,
    struct dyn_vmm_packet *response)
{
    struct imported_allocation *allocation;
    struct owner_allocation *owner;

    if (audit_local(response) != 0) {
        set_packet_state(response);
        return send_packet(client, response, -1);
    }
    if (phase == DYN_VMM_PHASE_ACTIVE && generation == 0 &&
        request->generation != 0) {
        generation = request->generation;
        phase = DYN_VMM_PHASE_AUDITED;
        revision++;
    }
    set_packet_state(response);
    if (send_packet(client, response, -1) != 0)
        return -1;
    for (owner = owners; owner != NULL; owner = owner->next) {
        struct dyn_vmm_packet record = {
            .magic = DYN_VMM_MAGIC,
            .version = DYN_VMM_VERSION,
            .operation = DYN_VMM_OP_AUDIT,
            .object_dev = owner->object_dev,
            .object_ino = owner->object_ino,
            .role = DYN_VMM_ROLE_OWNER,
        };
        if (!owner->exported)
            continue;
        set_packet_state(&record);
        if (send_packet(client, &record, -1) != 0)
            return -1;
    }
    for (allocation = imports; allocation != NULL;
         allocation = allocation->next) {
        struct dyn_vmm_packet record = {
            .magic = DYN_VMM_MAGIC,
            .version = DYN_VMM_VERSION,
            .operation = DYN_VMM_OP_AUDIT,
            .object_dev = allocation->object_dev,
            .object_ino = allocation->object_ino,
            .role = DYN_VMM_ROLE_IMPORTER,
            .mapping_count = mapping_count(allocation),
        };
        set_packet_state(&record);
        if (send_packet(client, &record, -1) != 0)
            return -1;
    }
    return 0;
}

static bool context_seen_before_owner(
    struct owner_allocation *owner)
{
    struct owner_allocation *earlier;

    for (earlier = owners; earlier != owner; earlier = earlier->next) {
        if (earlier->exported && earlier->context == owner->context)
            return true;
    }
    return false;
}

static bool context_seen_before_mapping(
    struct imported_mapping *mapping)
{
    struct owner_allocation *owner;
    struct imported_mapping *earlier;

    for (owner = owners; owner != NULL; owner = owner->next) {
        if (owner->exported && owner->context == mapping->context)
            return true;
    }
    for (earlier = mappings; earlier != mapping; earlier = earlier->next) {
        if (earlier->context == mapping->context)
            return true;
    }
    return false;
}

static int handle_sync(
    const struct dyn_vmm_packet *request,
    struct dyn_vmm_packet *response)
{
    struct owner_allocation *owner;
    struct imported_mapping *mapping;

    if (request->generation != generation ||
        (phase != DYN_VMM_PHASE_AUDITED &&
         phase != DYN_VMM_PHASE_CONTEXTS_SYNCED)) {
        packet_error(response, "synchronize generation/state mismatch");
        return -1;
    }
    if (phase == DYN_VMM_PHASE_CONTEXTS_SYNCED)
        return 0;
    for (owner = owners; owner != NULL; owner = owner->next) {
        if (!owner->exported ||
            context_seen_before_owner(owner))
            continue;
        if (synchronize_context(owner->context) != 0) {
            packet_error(response, "failed to synchronize owner context");
            return -1;
        }
    }
    for (mapping = mappings; mapping != NULL; mapping = mapping->next) {
        if (context_seen_before_mapping(mapping))
            continue;
        if (synchronize_context(mapping->context) != 0) {
            packet_error(response, "failed to synchronize peer context");
            return -1;
        }
    }
    phase = DYN_VMM_PHASE_CONTEXTS_SYNCED;
    revision++;
    return 0;
}

static CUmemGenericAllocationHandle allocation_real(
    struct imported_allocation *allocation)
{
    struct logical_handle *handle;

    for (handle = handles; handle != NULL; handle = handle->next) {
        if (handle->allocation == allocation && handle->real != 0)
            return handle->real;
    }
    return 0;
}

enum remap_result {
    REMAP_COMPLETE,
    REMAP_NEEDS_BROKER,
    REMAP_FAILED,
};

static struct imported_mapping *active_mapping(
    struct imported_allocation *allocation)
{
    struct imported_mapping *mapping;

    for (mapping = mappings; mapping != NULL; mapping = mapping->next) {
        if (mapping->allocation == allocation && !mapping->detached)
            return mapping;
    }
    return NULL;
}

static int scoped_retain(
    retain_type function, struct imported_mapping *mapping,
    CUmemGenericAllocationHandle *real)
{
    struct context_scope scope;
    int result;

    if (function == NULL ||
        enter_context(mapping->context, &scope) != 0)
        return -1;
    result = function(
                 real, (void *)(uintptr_t)mapping->ptr) ==
                 CUDA_SUCCESS
                 ? 0
                 : -1;
    if (leave_context(&scope) != 0)
        return -1;
    return result;
}

static enum remap_result best_effort_remap_detached(void)
{
    map_type map_function = (map_type)resolve_next("cuMemMap");
    retain_type retain_function =
        (retain_type)resolve_next("cuMemRetainAllocationHandle");
    struct imported_mapping *mapping;
    struct logical_handle *handle;
    bool needs_broker = false;

    for (mapping = mappings; mapping != NULL; mapping = mapping->next) {
        CUmemGenericAllocationHandle real;
        if (!mapping->detached)
            continue;
        real = allocation_real(mapping->allocation);
        if (real == 0) {
            needs_broker = true;
            continue;
        }
        if (scoped_map(map_function, mapping, real) != 0)
            return REMAP_FAILED;
        mapping->detached = false;
        if (restore_access(mapping) != 0)
            return REMAP_FAILED;
    }
    for (handle = handles; handle != NULL; handle = handle->next) {
        struct imported_mapping *source;
        CUmemGenericAllocationHandle retained = 0;

        if (handle->real != 0)
            continue;
        source = active_mapping(handle->allocation);
        if (source == NULL) {
            needs_broker = true;
            continue;
        }
        if (scoped_retain(
                retain_function, source, &retained) != 0)
            return REMAP_FAILED;
        handle->real = retained;
    }
    return needs_broker ? REMAP_NEEDS_BROKER : REMAP_COMPLETE;
}

static int handle_detach(
    const struct dyn_vmm_packet *request,
    struct dyn_vmm_packet *response)
{
    unmap_type unmap_function =
        (unmap_type)resolve_next("cuMemUnmap");
    release_type release_function =
        (release_type)resolve_next("cuMemRelease");
    struct imported_mapping *mapping;
    struct logical_handle *handle;

    if (request->generation != generation ||
        (phase != DYN_VMM_PHASE_CONTEXTS_SYNCED &&
         phase != DYN_VMM_PHASE_LOCK_READY)) {
        packet_error(response, "detach generation/state mismatch");
        return -1;
    }
    if (phase == DYN_VMM_PHASE_LOCK_READY)
        return 0;
    for (mapping = mappings; mapping != NULL; mapping = mapping->next) {
        if (mapping->detached)
            continue;
        if (scoped_unmap(unmap_function, mapping) != 0) {
            if (best_effort_remap_detached() != REMAP_COMPLETE)
                poison("peer detach rollback failed");
            packet_error(
                response, "failed to unmap peer VA 0x%llx",
                (unsigned long long)mapping->ptr);
            return -1;
        }
        mapping->detached = true;
        stats.peer_detached++;
        log_message(
            "event=peer_detach generation=%llu va=0x%llx size=%zu offset=%zu",
            (unsigned long long)generation,
            (unsigned long long)mapping->ptr,
            mapping->size, mapping->offset);
    }
    for (handle = handles; handle != NULL; handle = handle->next) {
        if (handle->real == 0)
            continue;
        if (release_function == NULL ||
            release_function(handle->real) != CUDA_SUCCESS) {
            enum remap_result rollback =
                best_effort_remap_detached();

            if (rollback == REMAP_FAILED) {
                poison("imported handle release and peer rollback failed");
                packet_error(
                    response,
                    "failed to release imported real handle; "
                    "peer rollback also failed");
            } else {
                if (rollback == REMAP_NEEDS_BROKER)
                    phase = DYN_VMM_PHASE_REATTACHING;
                revision++;
                packet_error(
                    response,
                    rollback == REMAP_COMPLETE
                        ? "failed to release imported real handle; "
                          "local peer rollback restored active mappings"
                        : "failed to release imported real handle; "
                          "Snapshot-brokered peer rollback required");
            }
            return -1;
        }
        handle->real = 0;
    }
    phase = DYN_VMM_PHASE_LOCK_READY;
    revision++;
    return 0;
}

static int handle_verify_lock_ready(
    const struct dyn_vmm_packet *request,
    struct dyn_vmm_packet *response)
{
    struct imported_mapping *mapping;
    struct logical_handle *handle;

    if (request->generation != generation ||
        phase != DYN_VMM_PHASE_LOCK_READY) {
        packet_error(response, "lock-ready generation/state mismatch");
        return -1;
    }
    for (mapping = mappings; mapping != NULL; mapping = mapping->next) {
        if (!mapping->detached ||
            find_reservation(mapping->ptr, mapping->size) == NULL ||
            mapping->reservation_ptr >
                mapping->ptr ||
            mapping->ptr + mapping->size >
                mapping->reservation_ptr +
                    mapping->reservation_size) {
            packet_error(
                response, "peer VA 0x%llx is not detached/reserved",
                (unsigned long long)mapping->ptr);
            return -1;
        }
    }
    for (handle = handles; handle != NULL; handle = handle->next) {
        if (handle->real != 0) {
            packet_error(response, "imported real handle remains live");
            return -1;
        }
    }
    return 0;
}

static int handle_export(
    const struct dyn_vmm_packet *request,
    struct dyn_vmm_packet *response, int *export_fd)
{
    typedef CUresult(CUDAAPI * function_type)(
        void *, CUmemGenericAllocationHandle,
        CUmemAllocationHandleType, unsigned long long);
    function_type function =
        (function_type)resolve_next("cuMemExportToShareableHandle");
    retain_type retain_function =
        (retain_type)resolve_next("cuMemRetainAllocationHandle");
    release_type release_function =
        (release_type)resolve_next("cuMemRelease");
    struct owner_allocation *owner;
    struct context_scope scope;
    CUmemGenericAllocationHandle real;
    bool temporary = false;
    bool failed = false;
    int fd = -1;

    if (request->generation != generation ||
        (phase != DYN_VMM_PHASE_AUDITED &&
         phase != DYN_VMM_PHASE_CONTEXTS_SYNCED &&
         phase != DYN_VMM_PHASE_LOCK_READY &&
         phase != DYN_VMM_PHASE_REATTACHING)) {
        packet_error(response, "fresh export generation/state mismatch");
        return -1;
    }
    owner = find_owner_by_id(
        request->object_dev, request->object_ino);
    if (owner == NULL || function == NULL ||
        enter_context(owner->context, &scope) != 0) {
        packet_error(response, "failed to access native owner for re-export");
        return -1;
    }
    real = owner->application_real;
    if (owner->application_released) {
        if (owner->mapped_ptr == 0 || retain_function == NULL ||
            retain_function(
                &real,
                (void *)(uintptr_t)owner->mapped_ptr) != CUDA_SUCCESS)
            failed = true;
        else
            temporary = true;
    }
    if (!failed &&
        (function(
             &fd, real, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
             0) != CUDA_SUCCESS ||
         fd_identity(
             fd, &response->fresh_dev,
             &response->fresh_ino) != 0))
        failed = true;
    if (temporary &&
        (release_function == NULL ||
         release_function(real) != CUDA_SUCCESS)) {
        poison("failed to release temporary native owner handle");
        failed = true;
    }
    if (leave_context(&scope) != 0)
        failed = true;
    if (failed) {
        if (fd >= 0)
            close(fd);
        packet_error(response, "failed to re-export native owner");
        return -1;
    }
    response->object_dev = owner->object_dev;
    response->object_ino = owner->object_ino;
    *export_fd = fd;
    return 0;
}

static int cleanup_import_replay(
    struct imported_allocation *allocation,
    CUmemGenericAllocationHandle temporary)
{
    unmap_type unmap_function =
        (unmap_type)resolve_next("cuMemUnmap");
    release_type release_function =
        (release_type)resolve_next("cuMemRelease");
    struct imported_mapping *mapping;
    struct logical_handle *handle;
    bool failed = false;

    for (mapping = mappings; mapping != NULL; mapping = mapping->next) {
        if (mapping->allocation != allocation || mapping->detached)
            continue;
        if (scoped_unmap(unmap_function, mapping) != 0)
            failed = true;
        else
            mapping->detached = true;
    }
    for (handle = handles; handle != NULL; handle = handle->next) {
        if (handle->allocation != allocation || handle->real == 0)
            continue;
        if (release_function == NULL ||
            release_function(handle->real) != CUDA_SUCCESS)
            failed = true;
        else
            handle->real = 0;
    }
    if (temporary != 0 &&
        (release_function == NULL ||
         release_function(temporary) != CUDA_SUCCESS))
        failed = true;
    if (failed) {
        poison("peer replay cleanup failed");
        return -1;
    }
    return 0;
}

enum import_replay_state {
    IMPORT_REPLAY_ACTIVE,
    IMPORT_REPLAY_DETACHED,
    IMPORT_REPLAY_INCONSISTENT,
};

static enum import_replay_state import_replay_state(
    struct imported_allocation *allocation)
{
    struct imported_mapping *mapping;
    struct logical_handle *handle;
    bool saw_mapping = false;
    bool active = false;
    bool detached = false;
    bool live_handle = false;
    bool released_handle = false;

    for (mapping = mappings; mapping != NULL; mapping = mapping->next) {
        if (mapping->allocation != allocation)
            continue;
        saw_mapping = true;
        if (mapping->detached)
            detached = true;
        else
            active = true;
    }
    for (handle = handles; handle != NULL; handle = handle->next) {
        if (handle->allocation != allocation)
            continue;
        if (handle->real == 0)
            released_handle = true;
        else
            live_handle = true;
    }
    if (saw_mapping && active && !detached && !released_handle)
        return IMPORT_REPLAY_ACTIVE;
    if (saw_mapping && detached && !active && !live_handle)
        return IMPORT_REPLAY_DETACHED;
    return IMPORT_REPLAY_INCONSISTENT;
}

static int handle_import(
    const struct dyn_vmm_packet *request,
    struct dyn_vmm_packet *response, int received_fd)
{
    import_type import_function =
        (import_type)resolve_next("cuMemImportFromShareableHandle");
    release_type release_function =
        (release_type)resolve_next("cuMemRelease");
    map_type map_function = (map_type)resolve_next("cuMemMap");
    struct imported_allocation *allocation;
    struct imported_mapping *mapping;
    struct logical_handle *handle;
    CUmemGenericAllocationHandle primary = 0;
    CUmemGenericAllocationHandle temporary = 0;
    CUmemGenericAllocationHandle imported;
    uint64_t fresh_dev;
    uint64_t fresh_ino;
    bool primary_assigned = false;

    if (request->generation != generation ||
        (phase != DYN_VMM_PHASE_LOCK_READY &&
         phase != DYN_VMM_PHASE_REATTACHING) ||
        fd_identity(received_fd, &fresh_dev, &fresh_ino) != 0 ||
        fresh_dev != request->fresh_dev ||
        fresh_ino != request->fresh_ino) {
        packet_error(response, "fresh import packet/FD mismatch");
        return -1;
    }
    allocation = find_import(
        request->object_dev, request->object_ino);
    if (allocation == NULL) {
        packet_error(response, "unknown imported peer identity");
        return -1;
    }
    switch (import_replay_state(allocation)) {
    case IMPORT_REPLAY_ACTIVE:
        phase = DYN_VMM_PHASE_REATTACHING;
        return 0;
    case IMPORT_REPLAY_DETACHED:
        if (import_function == NULL) {
            packet_error(response, "fresh POSIX peer import unavailable");
            return -1;
        }
        break;
    case IMPORT_REPLAY_INCONSISTENT:
        packet_error(
            response,
            "imported peer object is neither fully active nor detached");
        return -1;
    }
    phase = DYN_VMM_PHASE_REATTACHING;
    if (import_function(
            &primary, (void *)(uintptr_t)received_fd,
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) != CUDA_SUCCESS) {
        packet_error(response, "fresh POSIX peer import failed");
        return -1;
    }
    for (handle = handles; handle != NULL; handle = handle->next) {
        if (handle->allocation != allocation)
            continue;
        if (!primary_assigned) {
            handle->real = primary;
            primary_assigned = true;
            continue;
        }
        if (import_function(
                &imported, (void *)(uintptr_t)received_fd,
                CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) != CUDA_SUCCESS) {
            packet_error(response, "peer logical reference import failed");
            goto fail;
        }
        handle->real = imported;
    }
    if (!primary_assigned)
        temporary = primary;
    for (mapping = mappings; mapping != NULL; mapping = mapping->next) {
        if (mapping->allocation != allocation)
            continue;
        if (!mapping->detached ||
            scoped_map(map_function, mapping, primary) != 0) {
            packet_error(
                response, "exact peer replay failed at 0x%llx",
                (unsigned long long)mapping->ptr);
            goto fail;
        }
        mapping->detached = false;
        if (restore_access(mapping) != 0) {
            packet_error(
                response, "peer access replay failed at 0x%llx",
                (unsigned long long)mapping->ptr);
            goto fail;
        }
        stats.peer_reattached++;
        log_message(
            "event=peer_reattach generation=%llu va=0x%llx size=%zu offset=%zu fresh_real=0x%llx",
            (unsigned long long)generation,
            (unsigned long long)mapping->ptr,
            mapping->size, mapping->offset,
            (unsigned long long)primary);
    }
    if (temporary != 0) {
        if (release_function == NULL ||
            release_function(temporary) != CUDA_SUCCESS) {
            packet_error(response, "temporary peer handle release failed");
            goto fail;
        }
    }
    revision++;
    return 0;

fail:
    (void)cleanup_import_replay(allocation, temporary);
    return -1;
}

static int handle_verify_active(
    const struct dyn_vmm_packet *request,
    struct dyn_vmm_packet *response)
{
    struct imported_mapping *mapping;
    struct logical_handle *handle;

    if (request->generation != generation ||
        (phase != DYN_VMM_PHASE_REATTACHING &&
         phase != DYN_VMM_PHASE_LOCK_READY &&
         phase != DYN_VMM_PHASE_ACTIVE_VERIFIED)) {
        packet_error(response, "active verification generation/state mismatch");
        return -1;
    }
    if (phase == DYN_VMM_PHASE_ACTIVE_VERIFIED)
        return 0;
    for (mapping = mappings; mapping != NULL; mapping = mapping->next) {
        if (mapping->detached || verify_access(mapping) != 0) {
            packet_error(
                response, "peer mapping/access not active at 0x%llx",
                (unsigned long long)mapping->ptr);
            return -1;
        }
    }
    for (handle = handles; handle != NULL; handle = handle->next) {
        if (handle->real == 0) {
            packet_error(response, "logical peer handle has no real import");
            return -1;
        }
    }
    for (mapping = mappings; mapping != NULL; mapping = mapping->next) {
        if (synchronize_context(mapping->context) != 0) {
            packet_error(response, "peer context synchronization failed");
            return -1;
        }
    }
    phase = DYN_VMM_PHASE_ACTIVE_VERIFIED;
    revision++;
    return 0;
}

static int handle_commit(
    const struct dyn_vmm_packet *request,
    struct dyn_vmm_packet *response)
{
    if (request->generation != generation ||
        phase != DYN_VMM_PHASE_ACTIVE_VERIFIED) {
        packet_error(response, "commit generation/state mismatch");
        return -1;
    }
    phase = DYN_VMM_PHASE_ACTIVE;
    revision++;
    log_message(
        "event=commit generation=%llu owner_exports=%llu peer_imports=%llu peer_detached=%llu peer_reattached=%llu multicast_attempts=%llu unsupported=0x%x",
        (unsigned long long)generation,
        (unsigned long long)stats.owner_exports,
        (unsigned long long)stats.peer_imports,
        (unsigned long long)stats.peer_detached,
        (unsigned long long)stats.peer_reattached,
        (unsigned long long)stats.multicast_attempts,
        unsupported);
    return 0;
}

static int handle_abort(
    const struct dyn_vmm_packet *request,
    struct dyn_vmm_packet *response)
{
    if (phase == DYN_VMM_PHASE_ACTIVE && generation == 0)
        return 0;
    if (request->generation != generation ||
        phase == DYN_VMM_PHASE_POISONED) {
        packet_error(response, "abort generation/state mismatch");
        return -1;
    }
    if (phase <= DYN_VMM_PHASE_CONTEXTS_SYNCED ||
        phase == DYN_VMM_PHASE_ACTIVE_VERIFIED) {
        phase = DYN_VMM_PHASE_ACTIVE;
        generation = 0;
        revision++;
        return 0;
    }
    packet_error(
        response,
        "detached peers require Snapshot-brokered best-effort remap");
    return -1;
}

static void serve_client(int client)
{
    struct dyn_vmm_packet request = {0};
    struct dyn_vmm_packet response = {
        .magic = DYN_VMM_MAGIC,
        .version = DYN_VMM_VERSION,
        .pid = (uint32_t)getpid(),
    };
    struct ucred credentials;
    socklen_t credentials_size = sizeof(credentials);
    int received_fd = -1;
    int export_fd = -1;

    if (getsockopt(
            client, SOL_SOCKET, SO_PEERCRED, &credentials,
            &credentials_size) != 0 ||
        (credentials.uid != 0 && credentials.uid != geteuid())) {
        packet_error(&response, "control peer credentials rejected");
        (void)send_packet(client, &response, -1);
        return;
    }
    if (receive_packet(client, &request, &received_fd) != 0 ||
        request.magic != DYN_VMM_MAGIC ||
        request.version != DYN_VMM_VERSION ||
        request.pid != (uint32_t)getpid() ||
        (request.operation == DYN_VMM_OP_IMPORT
             ? received_fd < 0
             : received_fd >= 0)) {
        packet_error(&response, "invalid control packet");
        (void)send_packet(client, &response, -1);
        if (received_fd >= 0)
            close(received_fd);
        return;
    }
    response.operation = request.operation;
    pthread_mutex_lock(&state_lock);
    switch (request.operation) {
    case DYN_VMM_OP_AUDIT:
        (void)handle_audit(&request, client, &response);
        pthread_mutex_unlock(&state_lock);
        return;
    case DYN_VMM_OP_SYNC_ALL_CONTEXTS:
        (void)handle_sync(&request, &response);
        break;
    case DYN_VMM_OP_DETACH_IMPORTS:
        (void)handle_detach(&request, &response);
        break;
    case DYN_VMM_OP_VERIFY_LOCK_READY:
        (void)handle_verify_lock_ready(&request, &response);
        break;
    case DYN_VMM_OP_EXPORT:
        (void)handle_export(&request, &response, &export_fd);
        break;
    case DYN_VMM_OP_IMPORT:
        (void)handle_import(
            &request, &response, received_fd);
        break;
    case DYN_VMM_OP_VERIFY_ACTIVE:
        (void)handle_verify_active(&request, &response);
        break;
    case DYN_VMM_OP_COMMIT:
        (void)handle_commit(&request, &response);
        break;
    case DYN_VMM_OP_ABORT:
        (void)handle_abort(&request, &response);
        break;
    default:
        packet_error(
            &response, "unknown control operation %u",
            request.operation);
        break;
    }
    set_packet_state(&response);
    pthread_mutex_unlock(&state_lock);
    (void)send_packet(
        client, &response,
        response.status == 0 ? export_fd : -1);
    if (export_fd >= 0)
        close(export_fd);
    if (received_fd >= 0)
        close(received_fd);
}

static void *agent_main(void *unused)
{
    struct sockaddr_un address = {
        .sun_family = AF_UNIX,
    };
    char path[sizeof(address.sun_path)];
    const char *control_dir =
        getenv("DYN_SNAPSHOT_CONTROL_DIR");
    int fd;

    (void)unused;
    if (control_dir == NULL || control_dir[0] == '\0')
        control_dir = DEFAULT_CONTROL_DIR;
    if (snprintf(
            path, sizeof(path), "%s/%s%d.sock",
            control_dir, DYN_VMM_SOCKET_PREFIX, getpid()) >=
        (int)sizeof(path)) {
        log_message("control socket path is too long");
        return NULL;
    }
    pthread_mutex_lock(&state_lock);
    fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    if (fd < 0) {
        pthread_mutex_unlock(&state_lock);
        return NULL;
    }
    (void)unlink(path);
    snprintf(
        address.sun_path, sizeof(address.sun_path), "%s",
        path);
    if (bind(
            fd, (struct sockaddr *)&address,
            sizeof(address)) != 0 ||
        chmod(path, 0600) != 0 ||
        listen(fd, 8) != 0) {
        close(fd);
        (void)unlink(path);
        pthread_mutex_unlock(&state_lock);
        return NULL;
    }
    listener_fd = fd;
    snprintf(socket_path, sizeof(socket_path), "%s", path);
    pthread_mutex_unlock(&state_lock);
    for (;;) {
        int client = accept4(fd, NULL, NULL, SOCK_CLOEXEC);
        if (client < 0) {
            if (errno == EINTR)
                continue;
            break;
        }
        serve_client(client);
        close(client);
    }
    return NULL;
}

static void start_agent(void)
{
    pthread_t thread;

    if (pthread_create(&thread, NULL, agent_main, NULL) != 0) {
        pthread_mutex_lock(&state_lock);
        mark_unsupported(
            DYN_VMM_UNSUPPORTED_BOOKKEEPING,
            "private Snapshot control thread failed to start");
        pthread_mutex_unlock(&state_lock);
        return;
    }
    pthread_detach(thread);
}

static void ensure_agent_started(void)
{
    if (enabled && !fork_child)
        pthread_once(&agent_once, start_agent);
}

static void observe_cuda_call(void)
{
    if (!enabled)
        return;
    ensure_agent_started();
    atomic_store_explicit(
        &cuda_observed, true, memory_order_relaxed);
}

static bool reject_fork(void)
{
    if (fork_child) {
        mark_unsupported(
            DYN_VMM_UNSUPPORTED_FORK,
            "CUDA VMM call after stateful fork");
        return true;
    }
    return false;
}

static void atfork_prepare(void)
{
    if (enabled)
        pthread_mutex_lock(&state_lock);
}

static void atfork_parent(void)
{
    if (enabled)
        pthread_mutex_unlock(&state_lock);
}

static void atfork_child(void)
{
    if (!enabled)
        return;
    if (listener_fd >= 0)
        close(listener_fd);
    listener_fd = -1;
    socket_path[0] = '\0';
    agent_once = (pthread_once_t)PTHREAD_ONCE_INIT;
    if (atomic_load_explicit(
            &cuda_observed, memory_order_relaxed) ||
        owners != NULL || imports != NULL ||
        mappings != NULL) {
        fork_child = true;
        mark_unsupported(
            DYN_VMM_UNSUPPORTED_FORK,
            "CUDA VMM state inherited across fork");
    }
    pthread_mutex_unlock(&state_lock);
}

__attribute__((constructor)) static void initialize(void)
{
    enabled = env_enabled("DYN_SNAPSHOT_CUDA_VMM_INTERPOSE");
    if (!enabled)
        return;
    force_posix =
        env_enabled("DYN_SNAPSHOT_CUDA_VMM_FORCE_POSIX");
    if (pthread_atfork(
            atfork_prepare, atfork_parent, atfork_child) != 0)
        mark_unsupported(
            DYN_VMM_UNSUPPORTED_FORK,
            "could not install fork bookkeeping");
}

__attribute__((destructor)) static void finalize(void)
{
    char path[sizeof(socket_path)] = {0};
    int fd;

    if (!enabled || fork_child)
        return;
    pthread_mutex_lock(&state_lock);
    fd = listener_fd;
    listener_fd = -1;
    snprintf(path, sizeof(path), "%s", socket_path);
    socket_path[0] = '\0';
    pthread_mutex_unlock(&state_lock);
    if (fd >= 0)
        close(fd);
    if (path[0] != '\0')
        (void)unlink(path);
}

#define BEGIN_MANAGED_WRAPPER()                                                \
    do {                                                                       \
        observe_cuda_call();                                                   \
        pthread_mutex_lock(&state_lock);                                       \
    } while (0)
#define END_MANAGED_WRAPPER() pthread_mutex_unlock(&state_lock)

CUresult CUDAAPI cuDeviceGetAttribute(
    int *value, CUdevice_attribute attribute, CUdevice device)
{
    typedef CUresult(CUDAAPI * function_type)(
        int *, CUdevice_attribute, CUdevice);
    function_type function =
        (function_type)resolve_next("cuDeviceGetAttribute");

    if (enabled && force_posix &&
        attribute ==
            CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED) {
        observe_cuda_call();
        if (value == NULL)
            return CUDA_ERROR_INVALID_VALUE;
        *value = 0;
        return CUDA_SUCCESS;
    }
    return function != NULL
               ? function(value, attribute, device)
               : unavailable();
}

CUresult CUDAAPI cuMemAddressReserve(
    CUdeviceptr *ptr, size_t size, size_t alignment,
    CUdeviceptr address, unsigned long long flags)
{
    typedef CUresult(CUDAAPI * function_type)(
        CUdeviceptr *, size_t, size_t, CUdeviceptr,
        unsigned long long);
    function_type function =
        (function_type)resolve_next("cuMemAddressReserve");
    struct reservation *reservation;
    CUresult result;

    if (!enabled)
        return function != NULL
                   ? function(ptr, size, alignment, address, flags)
                   : unavailable();
    BEGIN_MANAGED_WRAPPER();
    if (reject_fork()) {
        END_MANAGED_WRAPPER();
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    result = function != NULL
                 ? function(ptr, size, alignment, address, flags)
                 : unavailable();
    if (result == CUDA_SUCCESS) {
        reservation = calloc(1, sizeof(*reservation));
        if (reservation == NULL) {
            mark_unsupported(
                DYN_VMM_UNSUPPORTED_BOOKKEEPING,
                "VA reservation could not be recorded");
        } else {
            reservation->ptr = *ptr;
            reservation->size = size;
            reservation->next = reservations;
            reservations = reservation;
            revision++;
        }
    }
    END_MANAGED_WRAPPER();
    return result;
}

CUresult CUDAAPI cuMemAddressFree(CUdeviceptr ptr, size_t size)
{
    typedef CUresult(CUDAAPI * function_type)(CUdeviceptr, size_t);
    function_type function =
        (function_type)resolve_next("cuMemAddressFree");
    struct reservation **cursor;
    struct imported_mapping *mapping;
    CUresult result;

    if (!enabled)
        return function != NULL ? function(ptr, size) : unavailable();
    BEGIN_MANAGED_WRAPPER();
    if (reject_fork()) {
        END_MANAGED_WRAPPER();
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    for (mapping = mappings; mapping != NULL; mapping = mapping->next) {
        if (mapping->reservation_ptr == ptr &&
            mapping->reservation_size == size) {
            END_MANAGED_WRAPPER();
            return CUDA_ERROR_INVALID_VALUE;
        }
    }
    result = function != NULL ? function(ptr, size) : unavailable();
    if (result == CUDA_SUCCESS) {
        cursor = &reservations;
        while (*cursor != NULL) {
            if ((*cursor)->ptr == ptr && (*cursor)->size == size) {
                struct reservation *removed = *cursor;
                *cursor = removed->next;
                free(removed);
                break;
            }
            cursor = &(*cursor)->next;
        }
        revision++;
    }
    END_MANAGED_WRAPPER();
    return result;
}

CUresult CUDAAPI cuMemCreate(
    CUmemGenericAllocationHandle *output, size_t size,
    const CUmemAllocationProp *properties,
    unsigned long long flags)
{
    typedef CUresult(CUDAAPI * function_type)(
        CUmemGenericAllocationHandle *, size_t,
        const CUmemAllocationProp *, unsigned long long);
    function_type function =
        (function_type)resolve_next("cuMemCreate");
    release_type release_function =
        (release_type)resolve_next("cuMemRelease");
    struct owner_allocation *owner;
    CUresult result;
    bool fabric =
        properties != NULL &&
        (properties->requestedHandleTypes &
         CU_MEM_HANDLE_TYPE_FABRIC) != 0;

    if (!enabled)
        return function != NULL
                   ? function(output, size, properties, flags)
                   : unavailable();
    BEGIN_MANAGED_WRAPPER();
    if (reject_fork()) {
        END_MANAGED_WRAPPER();
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (fabric && force_posix) {
        log_message(
            "event=fabric_create_rejected reason=force_posix");
        END_MANAGED_WRAPPER();
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    result = function != NULL
                 ? function(output, size, properties, flags)
                 : unavailable();
    if (result == CUDA_SUCCESS && is_synthetic(*output)) {
        if (release_function == NULL ||
            release_function(*output) != CUDA_SUCCESS)
            poison("native owner token collision cleanup failed");
        else
            mark_unsupported(
                DYN_VMM_UNSUPPORTED_BOOKKEEPING,
                "native owner handle collided with logical token namespace");
        END_MANAGED_WRAPPER();
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (result == CUDA_SUCCESS) {
        owner = calloc(1, sizeof(*owner));
        if (owner == NULL ||
            current_context(&owner->context) != 0) {
            free(owner);
            mark_unsupported(
                DYN_VMM_UNSUPPORTED_BOOKKEEPING,
                "native owner metadata could not be recorded");
        } else {
            owner->application_real = *output;
            owner->next = owners;
            owners = owner;
            if (fabric)
                mark_unsupported(
                    DYN_VMM_UNSUPPORTED_FABRIC,
                    "live FABRIC VMM allocation is outside POSIX PoC scope");
            revision++;
        }
    }
    END_MANAGED_WRAPPER();
    return result;
}

CUresult CUDAAPI cuMemRelease(
    CUmemGenericAllocationHandle logical)
{
    release_type function =
        (release_type)resolve_next("cuMemRelease");
    struct logical_handle **cursor;
    struct owner_allocation **owner_cursor;
    struct imported_allocation *allocation = NULL;
    CUmemGenericAllocationHandle real;
    CUresult result;

    if (!enabled)
        return function != NULL ? function(logical) : unavailable();
    BEGIN_MANAGED_WRAPPER();
    if (reject_fork()) {
        END_MANAGED_WRAPPER();
        return is_synthetic(logical)
                   ? CUDA_ERROR_INVALID_HANDLE
                   : CUDA_ERROR_NOT_SUPPORTED;
    }
    if (!is_synthetic(logical)) {
        struct owner_allocation *owner = find_owner(logical);
        result = function != NULL ? function(logical) : unavailable();
        if (result == CUDA_SUCCESS && owner != NULL) {
            owner->application_released = true;
            owner->application_real = 0;
            if (!owner->exported) {
                owner_cursor = &owners;
                while (*owner_cursor != NULL) {
                    if (*owner_cursor == owner) {
                        *owner_cursor = owner->next;
                        free(owner);
                        break;
                    }
                    owner_cursor = &(*owner_cursor)->next;
                }
            } else
                maybe_remove_owner(owner);
        }
        END_MANAGED_WRAPPER();
        return result;
    }
    result = translate_import(logical, &real, &allocation);
    if (result != CUDA_SUCCESS) {
        END_MANAGED_WRAPPER();
        return result;
    }
    result = function != NULL ? function(real) : unavailable();
    if (result == CUDA_SUCCESS) {
        cursor = &handles;
        while (*cursor != NULL) {
            if ((*cursor)->logical == logical) {
                struct logical_handle *removed = *cursor;
                *cursor = removed->next;
                free(removed);
                break;
            }
            cursor = &(*cursor)->next;
        }
        maybe_remove_import(allocation);
        revision++;
    }
    END_MANAGED_WRAPPER();
    return result;
}

CUresult CUDAAPI cuMemMap(
    CUdeviceptr ptr, size_t size, size_t offset,
    CUmemGenericAllocationHandle logical,
    unsigned long long flags)
{
    map_type function = (map_type)resolve_next("cuMemMap");
    struct imported_allocation *allocation = NULL;
    struct imported_mapping *mapping = NULL;
    struct reservation *reservation;
    CUmemGenericAllocationHandle real;
    CUresult result;

    if (!enabled)
        return function != NULL
                   ? function(ptr, size, offset, logical, flags)
                   : unavailable();
    BEGIN_MANAGED_WRAPPER();
    if (reject_fork()) {
        END_MANAGED_WRAPPER();
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    result = translate_import(logical, &real, &allocation);
    if (result != CUDA_SUCCESS) {
        END_MANAGED_WRAPPER();
        return result;
    }
    if (allocation == NULL) {
        result = function != NULL
                     ? function(ptr, size, offset, real, flags)
                     : unavailable();
        if (result == CUDA_SUCCESS) {
            struct owner_allocation *owner = find_owner(real);
            if (owner != NULL) {
                owner->mapped_ptr = ptr;
                owner->mapped_size = size;
                if (current_context(&owner->context) != 0)
                    mark_unsupported(
                        DYN_VMM_UNSUPPORTED_BOOKKEEPING,
                        "native owner map context unavailable");
            }
        }
        END_MANAGED_WRAPPER();
        return result;
    }
    reservation = find_reservation(ptr, size);
    if (reservation == NULL) {
        mark_unsupported(
            DYN_VMM_UNSUPPORTED_UNKNOWN_HANDLE,
            "peer map does not belong to a tracked VA reservation");
        END_MANAGED_WRAPPER();
        return CUDA_ERROR_INVALID_VALUE;
    }
    mapping = calloc(1, sizeof(*mapping));
    if (mapping == NULL ||
        current_context(&mapping->context) != 0) {
        free(mapping);
        END_MANAGED_WRAPPER();
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    mapping->ptr = ptr;
    mapping->size = size;
    mapping->offset = offset;
    mapping->reservation_ptr = reservation->ptr;
    mapping->reservation_size = reservation->size;
    mapping->allocation = allocation;
    result = function != NULL
                 ? function(ptr, size, offset, real, flags)
                 : unavailable();
    if (result == CUDA_SUCCESS) {
        mapping->next = mappings;
        mappings = mapping;
        revision++;
    } else {
        free(mapping);
    }
    END_MANAGED_WRAPPER();
    return result;
}

CUresult CUDAAPI cuMemUnmap(CUdeviceptr ptr, size_t size)
{
    unmap_type function =
        (unmap_type)resolve_next("cuMemUnmap");
    struct imported_mapping **cursor;
    CUresult result;

    if (!enabled)
        return function != NULL ? function(ptr, size) : unavailable();
    BEGIN_MANAGED_WRAPPER();
    if (reject_fork()) {
        END_MANAGED_WRAPPER();
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    result = function != NULL ? function(ptr, size) : unavailable();
    if (result == CUDA_SUCCESS) {
        struct owner_allocation *owner;
        cursor = &mappings;
        while (*cursor != NULL) {
            if ((*cursor)->ptr == ptr && (*cursor)->size == size) {
                struct imported_mapping *removed = *cursor;
                struct imported_allocation *allocation =
                    removed->allocation;
                *cursor = removed->next;
                free_access(removed->access);
                free(removed);
                maybe_remove_import(allocation);
                break;
            }
            cursor = &(*cursor)->next;
        }
        for (owner = owners; owner != NULL; owner = owner->next) {
            if (owner->mapped_ptr == ptr &&
                owner->mapped_size == size) {
                owner->mapped_ptr = 0;
                owner->mapped_size = 0;
                maybe_remove_owner(owner);
                break;
            }
        }
        revision++;
    }
    END_MANAGED_WRAPPER();
    return result;
}

CUresult CUDAAPI cuMemSetAccess(
    CUdeviceptr ptr, size_t size,
    const CUmemAccessDesc *descriptors, size_t count)
{
    set_access_type function =
        (set_access_type)resolve_next("cuMemSetAccess");
    struct pending {
        struct imported_mapping *mapping;
        struct access_record *record;
        struct pending *next;
    };
    struct pending *pending = NULL;
    struct pending **tail = &pending;
    struct imported_mapping *mapping;
    CUdeviceptr end;
    CUresult result;

    if (!enabled)
        return function != NULL
                   ? function(ptr, size, descriptors, count)
                   : unavailable();
    BEGIN_MANAGED_WRAPPER();
    if (reject_fork()) {
        END_MANAGED_WRAPPER();
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (size == 0 || ptr > UINT64_MAX - size ||
        (count != 0 && descriptors == NULL)) {
        END_MANAGED_WRAPPER();
        return CUDA_ERROR_INVALID_VALUE;
    }
    end = ptr + size;
    for (mapping = mappings; mapping != NULL; mapping = mapping->next) {
        CUdeviceptr start =
            ptr > mapping->ptr ? ptr : mapping->ptr;
        CUdeviceptr mapping_end = mapping->ptr + mapping->size;
        CUdeviceptr intersection_end =
            end < mapping_end ? end : mapping_end;
        struct pending *update;

        if (start >= intersection_end)
            continue;
        update = calloc(1, sizeof(*update));
        if (update != NULL)
            update->record = calloc(1, sizeof(*update->record));
        if (update == NULL || update->record == NULL) {
            free(update);
            result = CUDA_ERROR_OUT_OF_MEMORY;
            goto free_pending;
        }
        if (count != 0) {
            update->record->descriptors =
                calloc(count, sizeof(*descriptors));
            if (update->record->descriptors == NULL) {
                free(update->record);
                free(update);
                result = CUDA_ERROR_OUT_OF_MEMORY;
                goto free_pending;
            }
            memcpy(
                update->record->descriptors, descriptors,
                count * sizeof(*descriptors));
        }
        update->record->ptr = start;
        update->record->size = intersection_end - start;
        update->record->descriptor_count = count;
        update->mapping = mapping;
        *tail = update;
        tail = &update->next;
    }
    result = function != NULL
                 ? function(ptr, size, descriptors, count)
                 : unavailable();
    if (result == CUDA_SUCCESS) {
        while (pending != NULL) {
            struct pending *update = pending;
            struct access_record **access_tail =
                &update->mapping->access;
            while (*access_tail != NULL)
                access_tail = &(*access_tail)->next;
            *access_tail = update->record;
            pending = update->next;
            free(update);
        }
        revision++;
    }
free_pending:
    while (pending != NULL) {
        struct pending *next = pending->next;
        free_access(pending->record);
        free(pending);
        pending = next;
    }
    END_MANAGED_WRAPPER();
    return result;
}

CUresult CUDAAPI cuMemGetAccess(
    unsigned long long *flags, const CUmemLocation *location,
    CUdeviceptr ptr)
{
    typedef CUresult(CUDAAPI * function_type)(
        unsigned long long *, const CUmemLocation *, CUdeviceptr);
    function_type function =
        (function_type)resolve_next("cuMemGetAccess");
    return function != NULL
               ? function(flags, location, ptr)
               : unavailable();
}

CUresult CUDAAPI cuMemExportToShareableHandle(
    void *shareable, CUmemGenericAllocationHandle logical,
    CUmemAllocationHandleType handle_type,
    unsigned long long flags)
{
    typedef CUresult(CUDAAPI * function_type)(
        void *, CUmemGenericAllocationHandle,
        CUmemAllocationHandleType, unsigned long long);
    function_type function =
        (function_type)resolve_next("cuMemExportToShareableHandle");
    struct owner_allocation *owner;
    CUmemGenericAllocationHandle real;
    CUresult result;
    uint64_t dev;
    uint64_t ino;

    if (!enabled)
        return function != NULL
                   ? function(shareable, logical, handle_type, flags)
                   : unavailable();
    BEGIN_MANAGED_WRAPPER();
    if (reject_fork()) {
        END_MANAGED_WRAPPER();
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (handle_type !=
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
        log_message(
            "event=non_posix_export_rejected handle_type=0x%x",
            (unsigned int)handle_type);
        END_MANAGED_WRAPPER();
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (is_synthetic(logical)) {
        mark_unsupported(
            DYN_VMM_UNSUPPORTED_UNKNOWN_HANDLE,
            "re-export of an imported peer is unsupported");
        END_MANAGED_WRAPPER();
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    real = logical;
    result = function != NULL
                 ? function(shareable, real, handle_type, flags)
                 : unavailable();
    if (result == CUDA_SUCCESS) {
        owner = find_owner(real);
        if (owner == NULL ||
            fd_identity(*(int *)shareable, &dev, &ino) != 0) {
            mark_unsupported(
                DYN_VMM_UNSUPPORTED_UNKNOWN_HANDLE,
                "POSIX export did not match a native owner");
        } else if (owner->exported &&
                   (owner->object_dev != dev ||
                    owner->object_ino != ino)) {
            mark_unsupported(
                DYN_VMM_UNSUPPORTED_UNKNOWN_HANDLE,
                "native owner export identity changed");
        } else {
            owner->exported = true;
            owner->object_dev = dev;
            owner->object_ino = ino;
            if (current_context(&owner->context) != 0)
                mark_unsupported(
                    DYN_VMM_UNSUPPORTED_BOOKKEEPING,
                    "owner export context unavailable");
            stats.owner_exports++;
            log_message(
                "event=owner_export object=%llu:%llu real=0x%llx",
                (unsigned long long)dev,
                (unsigned long long)ino,
                (unsigned long long)real);
        }
        revision++;
    }
    END_MANAGED_WRAPPER();
    return result;
}

CUresult CUDAAPI cuMemImportFromShareableHandle(
    CUmemGenericAllocationHandle *output, void *os_handle,
    CUmemAllocationHandleType handle_type)
{
    import_type function =
        (import_type)resolve_next("cuMemImportFromShareableHandle");
    release_type release_function =
        (release_type)resolve_next("cuMemRelease");
    struct imported_allocation *allocation;
    struct imported_allocation *new_allocation = NULL;
    struct logical_handle *handle;
    CUmemGenericAllocationHandle real = 0;
    uint64_t dev;
    uint64_t ino;
    CUresult result;

    if (!enabled)
        return function != NULL
                   ? function(output, os_handle, handle_type)
                   : unavailable();
    BEGIN_MANAGED_WRAPPER();
    if (reject_fork()) {
        END_MANAGED_WRAPPER();
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (handle_type !=
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
        log_message(
            "event=non_posix_import_rejected handle_type=0x%x",
            (unsigned int)handle_type);
        END_MANAGED_WRAPPER();
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (fd_identity(
            (int)(uintptr_t)os_handle, &dev, &ino) != 0) {
        END_MANAGED_WRAPPER();
        return CUDA_ERROR_INVALID_HANDLE;
    }
    allocation = find_import(dev, ino);
    if (allocation == NULL) {
        new_allocation = calloc(1, sizeof(*new_allocation));
        allocation = new_allocation;
        if (allocation != NULL) {
            allocation->object_dev = dev;
            allocation->object_ino = ino;
        }
    }
    handle = calloc(1, sizeof(*handle));
    if (allocation == NULL || handle == NULL) {
        free(new_allocation);
        free(handle);
        END_MANAGED_WRAPPER();
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    result = function != NULL
                 ? function(&real, os_handle, handle_type)
                 : unavailable();
    if (result != CUDA_SUCCESS) {
        free(new_allocation);
        free(handle);
        END_MANAGED_WRAPPER();
        return result;
    }
    handle->logical = new_logical_handle();
    handle->real = real;
    handle->allocation = allocation;
    if (new_allocation != NULL) {
        new_allocation->next = imports;
        imports = new_allocation;
    }
    handle->next = handles;
    handles = handle;
    *output = handle->logical;
    stats.peer_imports++;
    revision++;
    log_message(
        "event=peer_import object=%llu:%llu logical=0x%llx real=0x%llx",
        (unsigned long long)dev, (unsigned long long)ino,
        (unsigned long long)handle->logical,
        (unsigned long long)real);
    (void)release_function;
    END_MANAGED_WRAPPER();
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemGetAllocationGranularity(
    size_t *granularity,
    const CUmemAllocationProp *properties,
    CUmemAllocationGranularity_flags option)
{
    typedef CUresult(CUDAAPI * function_type)(
        size_t *, const CUmemAllocationProp *,
        CUmemAllocationGranularity_flags);
    function_type function = (function_type)resolve_next(
        "cuMemGetAllocationGranularity");

    return function != NULL
               ? function(granularity, properties, option)
               : unavailable();
}

CUresult CUDAAPI cuMemGetAllocationPropertiesFromHandle(
    CUmemAllocationProp *properties,
    CUmemGenericAllocationHandle logical)
{
    typedef CUresult(CUDAAPI * function_type)(
        CUmemAllocationProp *,
        CUmemGenericAllocationHandle);
    function_type function = (function_type)resolve_next(
        "cuMemGetAllocationPropertiesFromHandle");
    CUmemGenericAllocationHandle real;
    CUresult result;

    if (!enabled)
        return function != NULL
                   ? function(properties, logical)
                   : unavailable();
    BEGIN_MANAGED_WRAPPER();
    result = translate_import(logical, &real, NULL);
    if (result == CUDA_SUCCESS)
        result = function != NULL
                     ? function(properties, real)
                     : unavailable();
    END_MANAGED_WRAPPER();
    return result;
}

CUresult CUDAAPI cuMemRetainAllocationHandle(
    CUmemGenericAllocationHandle *output, void *address)
{
    typedef CUresult(CUDAAPI * function_type)(
        CUmemGenericAllocationHandle *, void *);
    function_type function = (function_type)resolve_next(
        "cuMemRetainAllocationHandle");
    struct imported_mapping *mapping;
    struct logical_handle *handle;
    CUmemGenericAllocationHandle real;
    CUresult result;

    if (!enabled)
        return function != NULL
                   ? function(output, address)
                   : unavailable();
    BEGIN_MANAGED_WRAPPER();
    if (reject_fork()) {
        END_MANAGED_WRAPPER();
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    mapping = find_mapping_covering(
        (CUdeviceptr)(uintptr_t)address);
    if (mapping == NULL) {
        result = function != NULL
                     ? function(output, address)
                     : unavailable();
        END_MANAGED_WRAPPER();
        return result;
    }
    handle = calloc(1, sizeof(*handle));
    if (handle == NULL) {
        END_MANAGED_WRAPPER();
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    result = function != NULL
                 ? function(&real, address)
                 : unavailable();
    if (result != CUDA_SUCCESS) {
        free(handle);
        END_MANAGED_WRAPPER();
        return result;
    }
    handle->logical = new_logical_handle();
    handle->real = real;
    handle->allocation = mapping->allocation;
    handle->next = handles;
    handles = handle;
    *output = handle->logical;
    revision++;
    END_MANAGED_WRAPPER();
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemMapArrayAsync(
    CUarrayMapInfo *map_info, unsigned int count, CUstream stream)
{
    typedef CUresult(CUDAAPI * function_type)(
        CUarrayMapInfo *, unsigned int, CUstream);
    function_type function =
        (function_type)resolve_next("cuMemMapArrayAsync");
    unsigned int index;

    if (enabled) {
        for (index = 0; index < count; index++) {
            if (map_info[index].memOperationType ==
                    CU_MEM_OPERATION_TYPE_MAP &&
                map_info[index].memHandleType ==
                    CU_MEM_HANDLE_TYPE_GENERIC &&
                is_synthetic(
                    map_info[index].memHandle.memHandle)) {
                observe_cuda_call();
                pthread_mutex_lock(&state_lock);
                mark_unsupported(
                    DYN_VMM_UNSUPPORTED_UNKNOWN_HANDLE,
                    "array mapping of imported peers is unsupported");
                pthread_mutex_unlock(&state_lock);
                return CUDA_ERROR_NOT_SUPPORTED;
            }
        }
    }
    return function != NULL
               ? function(map_info, count, stream)
               : unavailable();
}

static CUresult legacy_ipc_result(CUresult result)
{
    if (enabled) {
        observe_cuda_call();
        pthread_mutex_lock(&state_lock);
        mark_unsupported(
            DYN_VMM_UNSUPPORTED_LEGACY_IPC,
            "legacy CUDA IPC observed");
        pthread_mutex_unlock(&state_lock);
    }
    return result;
}

CUresult CUDAAPI cuIpcGetMemHandle(
    CUipcMemHandle *handle, CUdeviceptr ptr)
{
    typedef CUresult(CUDAAPI * function_type)(
        CUipcMemHandle *, CUdeviceptr);
    function_type function =
        (function_type)resolve_next("cuIpcGetMemHandle");
    return legacy_ipc_result(
        function != NULL ? function(handle, ptr) : unavailable());
}

CUresult CUDAAPI cuIpcOpenMemHandle_v2(
    CUdeviceptr *ptr, CUipcMemHandle handle,
    unsigned int flags)
{
    typedef CUresult(CUDAAPI * function_type)(
        CUdeviceptr *, CUipcMemHandle, unsigned int);
    function_type function =
        (function_type)resolve_next("cuIpcOpenMemHandle_v2");
    return legacy_ipc_result(
        function != NULL
            ? function(ptr, handle, flags)
            : unavailable());
}

CUresult CUDAAPI cuIpcOpenMemHandle(
    CUdeviceptr *ptr, CUipcMemHandle handle,
    unsigned int flags)
{
    return cuIpcOpenMemHandle_v2(ptr, handle, flags);
}

CUresult CUDAAPI cuIpcCloseMemHandle(CUdeviceptr ptr)
{
    typedef CUresult(CUDAAPI * function_type)(CUdeviceptr);
    function_type function =
        (function_type)resolve_next("cuIpcCloseMemHandle");
    return legacy_ipc_result(
        function != NULL ? function(ptr) : unavailable());
}

static CUresult reject_multicast(const char *operation)
{
    observe_cuda_call();
    pthread_mutex_lock(&state_lock);
    stats.multicast_attempts++;
    pthread_mutex_unlock(&state_lock);
    log_message(
        "event=multicast_rejected operation=%s", operation);
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuMulticastCreate(
    CUmemGenericAllocationHandle *output,
    const CUmulticastObjectProp *properties)
{
    typedef CUresult(CUDAAPI * function_type)(
        CUmemGenericAllocationHandle *,
        const CUmulticastObjectProp *);
    function_type function =
        (function_type)resolve_next("cuMulticastCreate");

    if (!enabled)
        return function != NULL
                   ? function(output, properties)
                   : unavailable();
    return reject_multicast("cuMulticastCreate");
}

CUresult CUDAAPI cuMulticastAddDevice(
    CUmemGenericAllocationHandle handle, CUdevice device)
{
    typedef CUresult(CUDAAPI * function_type)(
        CUmemGenericAllocationHandle, CUdevice);
    function_type function =
        (function_type)resolve_next("cuMulticastAddDevice");

    if (!enabled)
        return function != NULL
                   ? function(handle, device)
                   : unavailable();
    return reject_multicast("cuMulticastAddDevice");
}

CUresult CUDAAPI cuMulticastBindMem(
    CUmemGenericAllocationHandle multicast,
    size_t multicast_offset,
    CUmemGenericAllocationHandle memory,
    size_t memory_offset, size_t size,
    unsigned long long flags)
{
    typedef CUresult(CUDAAPI * function_type)(
        CUmemGenericAllocationHandle, size_t,
        CUmemGenericAllocationHandle, size_t, size_t,
        unsigned long long);
    function_type function =
        (function_type)resolve_next("cuMulticastBindMem");

    if (!enabled)
        return function != NULL
                   ? function(
                         multicast, multicast_offset, memory,
                         memory_offset, size, flags)
                   : unavailable();
    return reject_multicast("cuMulticastBindMem");
}

CUresult CUDAAPI cuMulticastBindMem_v2(
    CUmemGenericAllocationHandle multicast, CUdevice device,
    size_t multicast_offset,
    CUmemGenericAllocationHandle memory,
    size_t memory_offset, size_t size,
    unsigned long long flags)
{
    typedef CUresult(CUDAAPI * function_type)(
        CUmemGenericAllocationHandle, CUdevice, size_t,
        CUmemGenericAllocationHandle, size_t, size_t,
        unsigned long long);
    function_type function =
        (function_type)resolve_next("cuMulticastBindMem_v2");

    if (!enabled)
        return function != NULL
                   ? function(
                         multicast, device, multicast_offset,
                         memory, memory_offset, size, flags)
                   : unavailable();
    return reject_multicast("cuMulticastBindMem_v2");
}

CUresult CUDAAPI cuMulticastBindAddr(
    CUmemGenericAllocationHandle multicast,
    size_t multicast_offset, CUdeviceptr ptr,
    size_t size, unsigned long long flags)
{
    typedef CUresult(CUDAAPI * function_type)(
        CUmemGenericAllocationHandle, size_t, CUdeviceptr,
        size_t, unsigned long long);
    function_type function =
        (function_type)resolve_next("cuMulticastBindAddr");

    if (!enabled)
        return function != NULL
                   ? function(
                         multicast, multicast_offset, ptr,
                         size, flags)
                   : unavailable();
    return reject_multicast("cuMulticastBindAddr");
}

CUresult CUDAAPI cuMulticastBindAddr_v2(
    CUmemGenericAllocationHandle multicast, CUdevice device,
    size_t multicast_offset, CUdeviceptr ptr,
    size_t size, unsigned long long flags)
{
    typedef CUresult(CUDAAPI * function_type)(
        CUmemGenericAllocationHandle, CUdevice, size_t,
        CUdeviceptr, size_t, unsigned long long);
    function_type function =
        (function_type)resolve_next("cuMulticastBindAddr_v2");

    if (!enabled)
        return function != NULL
                   ? function(
                         multicast, device, multicast_offset,
                         ptr, size, flags)
                   : unavailable();
    return reject_multicast("cuMulticastBindAddr_v2");
}

CUresult CUDAAPI cuMulticastUnbind(
    CUmemGenericAllocationHandle multicast,
    CUdevice device, size_t multicast_offset,
    size_t size)
{
    typedef CUresult(CUDAAPI * function_type)(
        CUmemGenericAllocationHandle, CUdevice, size_t, size_t);
    function_type function =
        (function_type)resolve_next("cuMulticastUnbind");

    if (!enabled)
        return function != NULL
                   ? function(
                         multicast, device, multicast_offset, size)
                   : unavailable();
    return reject_multicast("cuMulticastUnbind");
}

CUresult CUDAAPI cuMulticastGetGranularity(
    size_t *granularity,
    const CUmulticastObjectProp *properties,
    CUmulticastGranularity_flags option)
{
    typedef CUresult(CUDAAPI * function_type)(
        size_t *, const CUmulticastObjectProp *,
        CUmulticastGranularity_flags);
    function_type function =
        (function_type)resolve_next("cuMulticastGetGranularity");

    if (!enabled)
        return function != NULL
                   ? function(granularity, properties, option)
                   : unavailable();
    return reject_multicast("cuMulticastGetGranularity");
}

static void mark_resolver_abi_unsupported(void)
{
    if (!enabled)
        return;
    pthread_mutex_lock(&state_lock);
    mark_unsupported(
        DYN_VMM_UNSUPPORTED_RESOLVER_ABI,
        "managed symbol requested with unsupported CUDA ABI version");
    pthread_mutex_unlock(&state_lock);
}

static void *replacement_for_symbol(
    const char *symbol, int cuda_version)
{
    void *replacement = NULL;
    int minimum_version = 0;

    if (symbol == NULL)
        return NULL;
#define REPLACE(name, introduced)                                              \
    if (strcmp(symbol, #name) == 0) {                                          \
        replacement = (void *)&name;                                           \
        minimum_version = introduced;                                          \
    }
    REPLACE(cuDeviceGetAttribute, 2000);
    REPLACE(cuMemAddressReserve, 10020);
    REPLACE(cuMemAddressFree, 10020);
    REPLACE(cuMemCreate, 10020);
    REPLACE(cuMemRelease, 10020);
    REPLACE(cuMemMap, 10020);
    REPLACE(cuMemUnmap, 10020);
    REPLACE(cuMemSetAccess, 10020);
    REPLACE(cuMemGetAccess, 10020);
    REPLACE(cuMemExportToShareableHandle, 10020);
    REPLACE(cuMemImportFromShareableHandle, 10020);
    REPLACE(cuMemGetAllocationGranularity, 10020);
    REPLACE(cuMemGetAllocationPropertiesFromHandle, 10020);
    REPLACE(cuMemRetainAllocationHandle, 11000);
    REPLACE(cuMemMapArrayAsync, 11010);
    REPLACE(cuIpcGetMemHandle, 4010);
    REPLACE(cuIpcCloseMemHandle, 4010);
    REPLACE(cuMulticastCreate, 12010);
    REPLACE(cuMulticastAddDevice, 12010);
    REPLACE(cuMulticastBindMem, 12010);
    REPLACE(cuMulticastBindMem_v2, 13010);
    REPLACE(cuMulticastBindAddr, 12010);
    REPLACE(cuMulticastBindAddr_v2, 13010);
    REPLACE(cuMulticastUnbind, 12010);
    REPLACE(cuMulticastGetGranularity, 12010);
    REPLACE(cuGetProcAddress, 11030);
    REPLACE(cuGetProcAddress_v2, 12000);
    REPLACE(cuGetProcAddress_v2_ptsz, 12000);
    REPLACE(cudaGetDriverEntryPoint, 11030);
    REPLACE(cudaGetDriverEntryPointByVersion, 12050);
    REPLACE(cudaGetDriverEntryPoint_ptsz, 11030);
    REPLACE(cudaGetDriverEntryPointByVersion_ptsz, 12050);
#undef REPLACE
    if (strcmp(symbol, "cuIpcOpenMemHandle") == 0) {
        replacement = (void *)&cuIpcOpenMemHandle_v2;
        minimum_version = 4010;
    } else if (strcmp(symbol, "cuIpcOpenMemHandle_v2") == 0) {
        replacement = (void *)&cuIpcOpenMemHandle_v2;
        minimum_version = 11000;
    }
    if (replacement != NULL && cuda_version < minimum_version) {
        mark_resolver_abi_unsupported();
        return NULL;
    }
    return replacement;
}

CUresult CUDAAPI cuGetProcAddress(
    const char *symbol, void **function_pointer,
    int cuda_version, cuuint64_t flags)
{
    typedef CUresult(CUDAAPI * function_type)(
        const char *, void **, int, cuuint64_t);
    function_type function =
        (function_type)resolve_next("cuGetProcAddress");
    CUresult result = function != NULL
                          ? function(
                                symbol, function_pointer,
                                cuda_version, flags)
                          : unavailable();
    void *replacement;

    if (enabled && result == CUDA_SUCCESS &&
        function_pointer != NULL &&
        *function_pointer != NULL &&
        (replacement = replacement_for_symbol(
             symbol, cuda_version)) != NULL)
        *function_pointer = replacement;
    return result;
}

CUresult CUDAAPI cuGetProcAddress_v2(
    const char *symbol, void **function_pointer,
    int cuda_version, cuuint64_t flags,
    CUdriverProcAddressQueryResult *status)
{
    typedef CUresult(CUDAAPI * function_type)(
        const char *, void **, int, cuuint64_t,
        CUdriverProcAddressQueryResult *);
    function_type function =
        (function_type)resolve_next("cuGetProcAddress_v2");
    CUresult result = function != NULL
                          ? function(
                                symbol, function_pointer,
                                cuda_version, flags, status)
                          : unavailable();
    void *replacement;

    if (enabled && result == CUDA_SUCCESS &&
        function_pointer != NULL &&
        *function_pointer != NULL &&
        (replacement = replacement_for_symbol(
             symbol, cuda_version)) != NULL)
        *function_pointer = replacement;
    return result;
}

CUresult CUDAAPI cuGetProcAddress_v2_ptsz(
    const char *symbol, void **function_pointer,
    int cuda_version, cuuint64_t flags,
    CUdriverProcAddressQueryResult *status)
{
    return cuGetProcAddress_v2(
        symbol, function_pointer, cuda_version,
        flags | CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM,
        status);
}

static cudaError_t runtime_entry_point(
    const char *real_symbol, const char *requested,
    void **function_pointer, unsigned int cuda_version,
    unsigned long long flags,
    enum cudaDriverEntryPointQueryResult *status,
    bool has_version)
{
    typedef cudaError_t(CUDARTAPI * old_type)(
        const char *, void **, unsigned long long,
        enum cudaDriverEntryPointQueryResult *);
    typedef cudaError_t(CUDARTAPI * version_type)(
        const char *, void **, unsigned int,
        unsigned long long,
        enum cudaDriverEntryPointQueryResult *);
    void *raw = resolve_next(real_symbol);
    cudaError_t result;
    void *replacement;

    if (raw == NULL)
        return cudaErrorNotSupported;
    result = has_version
                 ? ((version_type)raw)(
                       requested, function_pointer,
                       cuda_version, flags, status)
                 : ((old_type)raw)(
                       requested, function_pointer,
                       flags, status);
    if (enabled && result == cudaSuccess &&
        function_pointer != NULL &&
        *function_pointer != NULL &&
        (replacement = replacement_for_symbol(
             requested,
             has_version ? (int)cuda_version
                         : CUDA_VERSION)) != NULL)
        *function_pointer = replacement;
    return result;
}

cudaError_t CUDARTAPI cudaGetDriverEntryPoint(
    const char *symbol, void **function_pointer,
    unsigned long long flags,
    enum cudaDriverEntryPointQueryResult *status)
{
    return runtime_entry_point(
        "cudaGetDriverEntryPoint", symbol, function_pointer,
        0, flags, status, false);
}

cudaError_t CUDARTAPI cudaGetDriverEntryPointByVersion(
    const char *symbol, void **function_pointer,
    unsigned int cuda_version, unsigned long long flags,
    enum cudaDriverEntryPointQueryResult *status)
{
    return runtime_entry_point(
        "cudaGetDriverEntryPointByVersion", symbol,
        function_pointer, cuda_version, flags, status, true);
}

cudaError_t CUDARTAPI cudaGetDriverEntryPoint_ptsz(
    const char *symbol, void **function_pointer,
    unsigned long long flags,
    enum cudaDriverEntryPointQueryResult *status)
{
    return runtime_entry_point(
        "cudaGetDriverEntryPoint_ptsz", symbol,
        function_pointer, 0, flags, status, false);
}

cudaError_t CUDARTAPI cudaGetDriverEntryPointByVersion_ptsz(
    const char *symbol, void **function_pointer,
    unsigned int cuda_version, unsigned long long flags,
    enum cudaDriverEntryPointQueryResult *status)
{
    return runtime_entry_point(
        "cudaGetDriverEntryPointByVersion_ptsz", symbol,
        function_pointer, cuda_version, flags, status, true);
}
