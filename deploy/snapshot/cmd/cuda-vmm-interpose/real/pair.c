/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define _GNU_SOURCE

#include <cuda.h>

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include "../protocol.h"

#define DATA_SIZE (2U * 1024U * 1024U)

struct app_message {
    uint32_t command;
    int32_t status;
    uint64_t address;
    uint64_t logical;
    uint64_t size;
    uint64_t offset;
    char detail[32];
};

enum app_command {
    APP_INITIAL_FD = 1,
    APP_READY = 2,
    APP_PEER_WRITE = 3,
    APP_VERIFY = 4,
    APP_VERIFY_OWNER = 5,
    APP_CLEANUP = 6,
};

struct cuda_api {
    CUresult(CUDAAPI * init)(unsigned int);
    CUresult(CUDAAPI * device_get)(CUdevice *, int);
    CUresult(CUDAAPI * context_create)(CUcontext *, unsigned int, CUdevice);
    CUresult(CUDAAPI * context_destroy)(CUcontext);
    CUresult(CUDAAPI * granularity)(
        size_t *, const CUmemAllocationProp *,
        CUmemAllocationGranularity_flags);
    CUresult(CUDAAPI * reserve)(
        CUdeviceptr *, size_t, size_t, CUdeviceptr, unsigned long long);
    CUresult(CUDAAPI * address_free)(CUdeviceptr, size_t);
    CUresult(CUDAAPI * create)(
        CUmemGenericAllocationHandle *, size_t,
        const CUmemAllocationProp *, unsigned long long);
    CUresult(CUDAAPI * import_handle)(
        CUmemGenericAllocationHandle *, void *, CUmemAllocationHandleType);
    CUresult(CUDAAPI * export_handle)(
        void *, CUmemGenericAllocationHandle,
        CUmemAllocationHandleType, unsigned long long);
    CUresult(CUDAAPI * release)(CUmemGenericAllocationHandle);
    CUresult(CUDAAPI * map)(
        CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle,
        unsigned long long);
    CUresult(CUDAAPI * unmap)(CUdeviceptr, size_t);
    CUresult(CUDAAPI * set_access)(
        CUdeviceptr, size_t, const CUmemAccessDesc *, size_t);
    CUresult(CUDAAPI * copy_htod)(CUdeviceptr, const void *, size_t);
    CUresult(CUDAAPI * copy_dtoh)(void *, CUdeviceptr, size_t);
    CUresult(CUDAAPI * copy_dtod)(CUdeviceptr, CUdeviceptr, size_t);
};

static void require(int condition, const char *message)
{
    if (!condition) {
        fprintf(stderr,
                "{\"event\":\"pair_error\",\"pid\":%d,"
                "\"detail\":\"%s\"}\n",
                getpid(), message);
        exit(1);
    }
}

static void check_cuda(CUresult result, const char *operation)
{
    if (result != CUDA_SUCCESS) {
        fprintf(stderr,
                "{\"event\":\"pair_cuda_error\",\"pid\":%d,"
                "\"operation\":\"%s\",\"result\":%d}\n",
                getpid(), operation, result);
        exit(1);
    }
}

static void *required_symbol(void *library, const char *name)
{
    void *result = dlsym(library, name);
    require(result != NULL, name);
    return result;
}

static struct cuda_api load_cuda(void)
{
    void *library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    struct cuda_api api;

    require(library != NULL, "dlopen(libcuda.so.1)");
#define LOAD(field, name) api.field = required_symbol(library, name)
    LOAD(init, "cuInit");
    LOAD(device_get, "cuDeviceGet");
    LOAD(context_create, "cuCtxCreate_v2");
    LOAD(context_destroy, "cuCtxDestroy_v2");
    LOAD(granularity, "cuMemGetAllocationGranularity");
    LOAD(reserve, "cuMemAddressReserve");
    LOAD(address_free, "cuMemAddressFree");
    LOAD(create, "cuMemCreate");
    LOAD(import_handle, "cuMemImportFromShareableHandle");
    LOAD(export_handle, "cuMemExportToShareableHandle");
    LOAD(release, "cuMemRelease");
    LOAD(map, "cuMemMap");
    LOAD(unmap, "cuMemUnmap");
    LOAD(set_access, "cuMemSetAccess");
    LOAD(copy_htod, "cuMemcpyHtoD_v2");
    LOAD(copy_dtoh, "cuMemcpyDtoH_v2");
    LOAD(copy_dtod, "cuMemcpyDtoD_v2");
#undef LOAD
    return api;
}

static void fill_pattern(unsigned char *buffer, size_t size,
                         unsigned int seed)
{
    size_t index;
    for (index = 0; index < size; index++)
        buffer[index] =
            (unsigned char)(index * 131U + seed);
}

static void send_app(int fd, const struct app_message *message,
                     int passed_fd)
{
    struct iovec iov = {
        .iov_base = (void *)message,
        .iov_len = sizeof(*message),
    };
    char control[CMSG_SPACE(sizeof(int))] = {0};
    struct msghdr envelope = {
        .msg_iov = &iov,
        .msg_iovlen = 1,
    };

    if (passed_fd >= 0) {
        struct cmsghdr *header;
        envelope.msg_control = control;
        envelope.msg_controllen = sizeof(control);
        header = CMSG_FIRSTHDR(&envelope);
        header->cmsg_level = SOL_SOCKET;
        header->cmsg_type = SCM_RIGHTS;
        header->cmsg_len = CMSG_LEN(sizeof(int));
        memcpy(CMSG_DATA(header), &passed_fd, sizeof(passed_fd));
    }
    require(sendmsg(fd, &envelope, MSG_NOSIGNAL) ==
                sizeof(*message),
            "send app message");
}

static struct app_message receive_app(int fd, int *received_fd)
{
    struct app_message message;
    struct iovec iov = {
        .iov_base = &message,
        .iov_len = sizeof(message),
    };
    char control[CMSG_SPACE(sizeof(int))] = {0};
    struct msghdr envelope = {
        .msg_iov = &iov,
        .msg_iovlen = 1,
        .msg_control = control,
        .msg_controllen = sizeof(control),
    };
    struct cmsghdr *header;

    *received_fd = -1;
    require(recvmsg(fd, &envelope, MSG_CMSG_CLOEXEC) ==
                sizeof(message) &&
                (envelope.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) == 0,
            "receive app message");
    header = CMSG_FIRSTHDR(&envelope);
    if (header != NULL) {
        require(header->cmsg_level == SOL_SOCKET &&
                    header->cmsg_type == SCM_RIGHTS &&
                    header->cmsg_len == CMSG_LEN(sizeof(int)),
                "invalid app ancillary data");
        memcpy(received_fd, CMSG_DATA(header), sizeof(*received_fd));
    }
    return message;
}

static size_t allocation_size(struct cuda_api *cuda,
                              CUmemAllocationProp *properties)
{
    size_t granularity;
    check_cuda(cuda->granularity(
                   &granularity, properties,
                   CU_MEM_ALLOC_GRANULARITY_RECOMMENDED),
               "granularity");
    return (DATA_SIZE + granularity - 1) / granularity * granularity;
}

static void owner_worker(int control_fd, int peer_fd)
{
    struct cuda_api cuda = load_cuda();
    CUmemAllocationProp properties = {0};
    CUmemAccessDesc access = {
        .location = {
            .type = CU_MEM_LOCATION_TYPE_DEVICE,
            .id = 0,
        },
        .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
    };
    CUmemGenericAllocationHandle logical;
    CUdeviceptr address;
    CUdevice device;
    CUcontext context;
    size_t peer_size;
    size_t owner_size;
    unsigned char *actual;
    unsigned char *expected;
    int export_fd;
    struct app_message message;
    int received_fd;

    check_cuda(cuda.init(0), "init");
    check_cuda(cuda.device_get(&device, 0), "device 0");
    check_cuda(cuda.context_create(&context, 0, device), "owner context");
    properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    properties.location.id = device;
    properties.requestedHandleTypes =
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    peer_size = allocation_size(&cuda, &properties);
    owner_size = peer_size * 2;
    actual = malloc(peer_size);
    expected = malloc(peer_size);
    require(actual != NULL && expected != NULL, "owner host buffers");
    fill_pattern(expected, peer_size, 29);
    check_cuda(cuda.reserve(
                   &address, owner_size, 0, 0, 0),
               "owner reserve");
    check_cuda(cuda.create(
                   &logical, owner_size, &properties, 0),
               "owner create");
    check_cuda(cuda.map(
                   address, owner_size, 0, logical, 0),
               "owner map");
    check_cuda(cuda.set_access(address, owner_size, &access, 1),
               "owner access");
    check_cuda(cuda.copy_htod(
                   address, expected, peer_size),
               "owner first-half initial copy");
    check_cuda(cuda.copy_htod(
                   address + peer_size, expected, peer_size),
               "owner peer-half initial copy");
    check_cuda(cuda.export_handle(
                   &export_fd, logical,
                   CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0),
               "owner export");
    check_cuda(cuda.release(logical), "owner application handle release");
    message = (struct app_message){
        .command = APP_INITIAL_FD,
        .address = address,
        .logical = logical,
        .size = peer_size,
        .offset = peer_size,
    };
    send_app(peer_fd, &message, export_fd);
    close(export_fd);
    message.command = APP_READY;
    send_app(control_fd, &message, -1);

    for (;;) {
        message = receive_app(control_fd, &received_fd);
        require(received_fd < 0, "owner command carried FD");
        if (message.command == APP_VERIFY_OWNER) {
            fill_pattern(expected, peer_size, 29);
            check_cuda(cuda.copy_dtoh(
                           actual, address, peer_size),
                       "owner unchanged copy");
            message.status =
                memcmp(actual, expected, peer_size) == 0 ? 0 : -1;
        } else if (message.command == APP_VERIFY) {
            fill_pattern(expected, peer_size, 83);
            check_cuda(cuda.copy_dtoh(
                           actual, address + peer_size,
                           peer_size),
                       "owner peer verify copy");
            message.status =
                memcmp(actual, expected, peer_size) == 0 ? 0 : -1;
        } else if (message.command == APP_CLEANUP) {
            check_cuda(cuda.unmap(address, owner_size), "owner unmap");
            check_cuda(cuda.address_free(address, owner_size),
                       "owner address free");
            check_cuda(cuda.context_destroy(context),
                       "owner context destroy");
            message.status = 0;
            send_app(control_fd, &message, -1);
            break;
        } else {
            message.status = -1;
        }
        send_app(control_fd, &message, -1);
    }
    free(expected);
    free(actual);
}

static void importer_worker(int control_fd, int peer_fd)
{
    struct cuda_api cuda = load_cuda();
    CUmemAllocationProp properties = {0};
    CUmemAccessDesc access = {
        .location = {
            .type = CU_MEM_LOCATION_TYPE_DEVICE,
            .id = 1,
        },
        .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
    };
    CUmemGenericAllocationHandle imported;
    CUmemGenericAllocationHandle local;
    CUdeviceptr imported_address;
    CUdeviceptr local_address;
    CUdevice device;
    CUcontext context;
    size_t size;
    unsigned char *pattern;
    struct app_message message;
    int import_fd;
    int received_fd;

    message = receive_app(peer_fd, &import_fd);
    require(message.command == APP_INITIAL_FD && import_fd >= 0,
            "importer initial FD");
    size = message.size;
    check_cuda(cuda.init(0), "init");
    check_cuda(cuda.device_get(&device, 1), "device 1");
    check_cuda(cuda.context_create(&context, 0, device),
               "importer context");
    properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    properties.location.id = device;
    properties.requestedHandleTypes =
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    check_cuda(cuda.import_handle(
                   &imported, (void *)(uintptr_t)import_fd,
                   CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
               "import owner");
    close(import_fd);
    check_cuda(cuda.reserve(
                   &imported_address, size, 0, 0, 0),
               "import reserve");
    check_cuda(cuda.map(
                   imported_address, size, message.offset,
                   imported, 0),
               "import map");
    check_cuda(cuda.set_access(
                   imported_address, size, &access, 1),
               "import access");
    check_cuda(cuda.create(&local, size, &properties, 0),
               "local create");
    check_cuda(cuda.reserve(&local_address, size, 0, 0, 0),
               "local reserve");
    check_cuda(cuda.map(local_address, size, 0, local, 0),
               "local map");
    check_cuda(cuda.set_access(local_address, size, &access, 1),
               "local access");
    pattern = malloc(size);
    require(pattern != NULL, "importer pattern");
    fill_pattern(pattern, size, 83);
    check_cuda(cuda.copy_htod(local_address, pattern, size),
               "local pattern copy");
    message = (struct app_message){
        .command = APP_READY,
        .address = imported_address,
        .logical = imported,
        .size = size,
        .offset = message.offset,
    };
    send_app(control_fd, &message, -1);

    for (;;) {
        message = receive_app(control_fd, &received_fd);
        require(received_fd < 0, "importer command carried FD");
        if (message.command == APP_PEER_WRITE) {
            check_cuda(cuda.copy_dtod(
                           imported_address, local_address, size),
                       "peer DtoD write");
            message.status = 0;
        } else if (message.command == APP_CLEANUP) {
            check_cuda(cuda.unmap(imported_address, size),
                       "imported unmap");
            check_cuda(cuda.release(imported), "imported release");
            check_cuda(cuda.address_free(imported_address, size),
                       "imported address free");
            check_cuda(cuda.unmap(local_address, size), "local unmap");
            check_cuda(cuda.release(local), "local release");
            check_cuda(cuda.address_free(local_address, size),
                       "local address free");
            check_cuda(cuda.context_destroy(context),
                       "importer context destroy");
            message.status = 0;
            send_app(control_fd, &message, -1);
            break;
        } else {
            message.status = -1;
        }
        send_app(control_fd, &message, -1);
    }
    free(pattern);
}

static int connect_agent(pid_t pid)
{
    const char *directory = getenv("DYN_SNAPSHOT_CONTROL_DIR");
    struct sockaddr_un address = {
        .sun_family = AF_UNIX,
    };
    unsigned int attempt;

    require(directory != NULL, "missing control directory");
    require(snprintf(address.sun_path, sizeof(address.sun_path),
                     "%s/%s%d.sock", directory,
                     DYN_VMM_SOCKET_PREFIX, pid) <
                (int)sizeof(address.sun_path),
            "agent path");
    for (attempt = 0; attempt < 500; attempt++) {
        int fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
        require(fd >= 0, "agent socket");
        if (connect(fd, (struct sockaddr *)&address,
                    sizeof(address)) == 0)
            return fd;
        close(fd);
        usleep(10000);
    }
    require(0, "agent unavailable");
    return -1;
}

static struct dyn_vmm_packet agent_command(
    pid_t pid, uint16_t operation, uint64_t generation,
    const struct dyn_vmm_packet *fields, int passed_fd,
    int *received_fd, bool expect_success)
{
    struct dyn_vmm_packet packet = {
        .magic = DYN_VMM_MAGIC,
        .version = DYN_VMM_VERSION,
        .operation = operation,
        .generation = generation,
        .pid = (uint32_t)pid,
    };
    struct iovec iov = {
        .iov_base = &packet,
        .iov_len = sizeof(packet),
    };
    char send_control[CMSG_SPACE(sizeof(int))] = {0};
    char receive_control[CMSG_SPACE(sizeof(int))] = {0};
    struct msghdr send_message = {
        .msg_iov = &iov,
        .msg_iovlen = 1,
    };
    struct msghdr receive_message = {
        .msg_iov = &iov,
        .msg_iovlen = 1,
        .msg_control = receive_control,
        .msg_controllen = sizeof(receive_control),
    };
    int fd = connect_agent(pid);
    int result_fd = -1;

    if (fields != NULL) {
        packet.object_dev = fields->object_dev;
        packet.object_ino = fields->object_ino;
        packet.fresh_dev = fields->fresh_dev;
        packet.fresh_ino = fields->fresh_ino;
    }
    if (passed_fd >= 0) {
        struct cmsghdr *header;
        send_message.msg_control = send_control;
        send_message.msg_controllen = sizeof(send_control);
        header = CMSG_FIRSTHDR(&send_message);
        header->cmsg_level = SOL_SOCKET;
        header->cmsg_type = SCM_RIGHTS;
        header->cmsg_len = CMSG_LEN(sizeof(int));
        memcpy(CMSG_DATA(header), &passed_fd, sizeof(passed_fd));
    }
    require(sendmsg(fd, &send_message, MSG_NOSIGNAL) ==
                sizeof(packet) &&
                recvmsg(fd, &receive_message, MSG_CMSG_CLOEXEC) ==
                    sizeof(packet),
            "agent exchange");
    if (CMSG_FIRSTHDR(&receive_message) != NULL)
        memcpy(&result_fd,
               CMSG_DATA(CMSG_FIRSTHDR(&receive_message)),
               sizeof(result_fd));
    close(fd);
    require((packet.status == 0) == expect_success,
            expect_success ? packet.message
                           : "agent unexpectedly succeeded");
    if (received_fd != NULL)
        *received_fd = result_fd;
    else if (result_fd >= 0)
        close(result_fd);
    fprintf(stdout,
            "{\"event\":\"pair_phase\",\"pid\":%d,"
            "\"operation\":%u,\"phase\":%u,"
            "\"status\":%d}\n",
            pid, operation, packet.phase, packet.status);
    fflush(stdout);
    return packet;
}

static void audit_owner(pid_t pid, uint64_t generation,
                        uint64_t *dev, uint64_t *ino)
{
    struct dyn_vmm_packet packet = {
        .magic = DYN_VMM_MAGIC,
        .version = DYN_VMM_VERSION,
        .operation = DYN_VMM_OP_AUDIT,
        .generation = generation,
        .pid = (uint32_t)pid,
    };
    int fd = connect_agent(pid);
    uint32_t index;

    require(send(fd, &packet, sizeof(packet), MSG_NOSIGNAL) ==
                sizeof(packet) &&
                recv(fd, &packet, sizeof(packet), MSG_WAITALL) ==
                    sizeof(packet) &&
                packet.status == 0,
            "pair audit");
    *dev = 0;
    *ino = 0;
    for (index = 0; index < packet.count; index++) {
        struct dyn_vmm_packet record;
        require(recv(fd, &record, sizeof(record), MSG_WAITALL) ==
                    sizeof(record),
                "pair audit record");
        if (record.role == DYN_VMM_ROLE_OWNER &&
            record.object_dev != 0) {
            *dev = record.object_dev;
            *ino = record.object_ino;
        }
    }
    close(fd);
}

static pid_t spawn_worker(const char *self, const char *role,
                          int control_fd, int peer_fd,
                          const char *interposer)
{
    pid_t pid = fork();

    require(pid >= 0, "fork launcher");
    if (pid == 0) {
        char control[16];
        char peer[16];
        snprintf(control, sizeof(control), "%d", control_fd);
        snprintf(peer, sizeof(peer), "%d", peer_fd);
        require(fcntl(control_fd, F_SETFD, 0) == 0 &&
                    fcntl(peer_fd, F_SETFD, 0) == 0,
                "clear worker descriptor CLOEXEC");
        setenv("LD_PRELOAD", interposer, 1);
        execl(self, self, role, control, peer, NULL);
        _exit(127);
    }
    return pid;
}

static void app_roundtrip(int fd, uint32_t command,
                          const char *detail)
{
    struct app_message message = {
        .command = command,
    };
    int received_fd;

    if (detail != NULL)
        strncpy(message.detail, detail,
                sizeof(message.detail) - 1);
    send_app(fd, &message, -1);
    message = receive_app(fd, &received_fd);
    require(received_fd < 0 && message.status == 0,
            "worker command failed");
}

static int controller(const char *self, const char *interposer)
{
    int owner_control[2];
    int importer_control[2];
    int peer[2];
    pid_t owner_pid;
    pid_t importer_pid;
    struct app_message owner_ready;
    struct app_message importer_ready;
    struct dyn_vmm_packet object;
    struct dyn_vmm_packet exported;
    int received_fd;
    int fresh_fd;
    uint64_t generation = 1;
    int status;

    require(socketpair(AF_UNIX, SOCK_SEQPACKET, 0, owner_control) == 0 &&
                socketpair(AF_UNIX, SOCK_SEQPACKET, 0,
                           importer_control) == 0 &&
                socketpair(AF_UNIX, SOCK_SEQPACKET, 0, peer) == 0,
            "worker socketpairs");
    {
        int all_fds[] = {
            owner_control[0], owner_control[1],
            importer_control[0], importer_control[1],
            peer[0], peer[1],
        };
        size_t index;
        for (index = 0;
             index < sizeof(all_fds) / sizeof(all_fds[0]); index++)
            require(fcntl(all_fds[index], F_SETFD, FD_CLOEXEC) == 0,
                    "set launcher descriptor CLOEXEC");
    }
    owner_pid = spawn_worker(
        self, "owner", owner_control[1], peer[0], interposer);
    importer_pid = spawn_worker(
        self, "importer", importer_control[1], peer[1], interposer);
    close(owner_control[1]);
    close(importer_control[1]);
    close(peer[0]);
    close(peer[1]);
    owner_ready = receive_app(owner_control[0], &received_fd);
    require(owner_ready.command == APP_READY && received_fd < 0,
            "owner not ready");
    importer_ready =
        receive_app(importer_control[0], &received_fd);
    require(importer_ready.command == APP_READY && received_fd < 0,
            "importer not ready");
    audit_owner(
        owner_pid, generation, &object.object_dev,
        &object.object_ino);
    {
        uint64_t ignored_dev;
        uint64_t ignored_ino;
        audit_owner(
            importer_pid, generation, &ignored_dev, &ignored_ino);
    }
    (void)agent_command(
        owner_pid, DYN_VMM_OP_SYNC_ALL_CONTEXTS, generation,
        NULL, -1, NULL, true);
    (void)agent_command(
        importer_pid, DYN_VMM_OP_SYNC_ALL_CONTEXTS, generation,
        NULL, -1, NULL, true);
    (void)agent_command(
        owner_pid, DYN_VMM_OP_DETACH_IMPORTS, generation,
        NULL, -1, NULL, true);
    (void)agent_command(
        importer_pid, DYN_VMM_OP_DETACH_IMPORTS, generation,
        NULL, -1, NULL, true);
    (void)agent_command(
        owner_pid, DYN_VMM_OP_VERIFY_LOCK_READY, generation,
        NULL, -1, NULL, true);
    (void)agent_command(
        importer_pid, DYN_VMM_OP_VERIFY_LOCK_READY,
        generation, NULL, -1, NULL, true);
    exported = agent_command(
        owner_pid, DYN_VMM_OP_EXPORT, generation, &object,
        -1, &fresh_fd, true);
    require(fresh_fd >= 0, "pair fresh FD absent");
    (void)agent_command(
        importer_pid, DYN_VMM_OP_IMPORT, generation, &exported,
        fresh_fd, NULL, true);
    close(fresh_fd);
    (void)agent_command(
        owner_pid, DYN_VMM_OP_VERIFY_ACTIVE, generation,
        NULL, -1, NULL, true);
    (void)agent_command(
        importer_pid, DYN_VMM_OP_VERIFY_ACTIVE, generation,
        NULL, -1, NULL, true);
    (void)agent_command(
        owner_pid, DYN_VMM_OP_COMMIT, generation,
        NULL, -1, NULL, true);
    (void)agent_command(
        importer_pid, DYN_VMM_OP_COMMIT, generation,
        NULL, -1, NULL, true);

    app_roundtrip(owner_control[0], APP_VERIFY_OWNER, NULL);
    app_roundtrip(importer_control[0], APP_PEER_WRITE, NULL);
    app_roundtrip(owner_control[0], APP_VERIFY, NULL);
    app_roundtrip(importer_control[0], APP_CLEANUP, NULL);
    app_roundtrip(owner_control[0], APP_CLEANUP, NULL);
    close(importer_control[0]);
    close(owner_control[0]);
    require(waitpid(owner_pid, &status, 0) == owner_pid &&
                WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "owner exit");
    require(waitpid(importer_pid, &status, 0) == importer_pid &&
                WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "importer exit");
    fprintf(stdout,
            "{\"event\":\"result\",\"level\":2,"
            "\"status\":\"pass\",\"generation\":%llu,"
            "\"owner_pid\":%d,\"importer_pid\":%d,"
            "\"owner_va\":%llu,\"importer_va\":%llu,"
            "\"owner_logical\":%llu,\"importer_logical\":%llu,"
            "\"peer_offset\":%llu,"
            "\"object\":\"%llu:%llu\","
            "\"peer_copy\":\"cuMemcpyDtoD_v2\"}\n",
            (unsigned long long)generation, owner_pid, importer_pid,
            (unsigned long long)owner_ready.address,
            (unsigned long long)importer_ready.address,
            (unsigned long long)owner_ready.logical,
            (unsigned long long)importer_ready.logical,
            (unsigned long long)importer_ready.offset,
            (unsigned long long)object.object_dev,
            (unsigned long long)object.object_ino);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc == 4 && strcmp(argv[1], "owner") == 0) {
        owner_worker(atoi(argv[2]), atoi(argv[3]));
        return 0;
    }
    if (argc == 4 && strcmp(argv[1], "importer") == 0) {
        importer_worker(atoi(argv[2]), atoi(argv[3]));
        return 0;
    }
    require(argc == 2, "usage: pair INTERPOSER");
    return controller(argv[0], argv[1]);
}
