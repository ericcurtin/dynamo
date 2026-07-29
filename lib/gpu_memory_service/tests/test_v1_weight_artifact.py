# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path

import msgspec
import pytest
from _fake_vmm import FakeVMM
from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
from gpu_memory_service.core.client.session import _GMSClientSession
from gpu_memory_service.core.protocol import AllocationRecord
from gpu_memory_service.core.server.gms import GMSServerMemoryManager
from gpu_memory_service.core.server.rpc import GMSRPCServer
from gpu_memory_service.v1 import weight_artifact

pytestmark = [pytest.mark.pre_merge, pytest.mark.integration, pytest.mark.gpu_0]


@contextmanager
def _server(path: str, vmm: FakeVMM, manager=None):
    manager = manager or GMSServerMemoryManager("GPU-0", vmm, 0)
    server = GMSRPCServer(path, manager)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
        assert not thread.is_alive()


class _ByteVMM(FakeVMM):
    def __init__(self):
        super().__init__(granularity=64)
        self._physical: dict[int, bytearray] = {}
        self._exported: dict[int, int] = {}
        self._imported: dict[int, int] = {}

    def create_tolerate_oom(self, size, device):
        allocated, handle = super().create_tolerate_oom(size, device)
        self._physical[handle] = bytearray(size)
        return allocated, handle

    def release(self, handle):
        if handle in self.server_handles:
            self._physical.pop(handle)
        self._imported.pop(handle, None)
        super().release(handle)

    def export_to_shareable_handle(self, handle):
        read_fd = super().export_to_shareable_handle(handle)
        self._exported[os.fstat(read_fd).st_ino] = handle
        return read_fd

    def import_shareable_handle_close_fd(self, fd):
        physical_handle = self._exported[os.fstat(fd).st_ino]
        imported_handle = super().import_shareable_handle_close_fd(fd)
        self._imported[imported_handle] = physical_handle
        return imported_handle

    def _allocation(self, va, size):
        mapped_size, imported_handle = self.mapped[va]
        assert size <= mapped_size
        return self._physical[self._imported[imported_handle]]

    def read(self, va, size):
        return bytes(self._allocation(va, size)[:size])

    def write(self, va, data):
        self._allocation(va, len(data))[: len(data)] = data


class _Writer:
    vmm: _ByteVMM

    def __init__(self, path, *, device):
        assert device == 0
        self._path = Path(path)
        self._data = bytearray()

    def __enter__(self):
        return self

    def write_device(self, src_ptr, byte_count):
        self._data.extend(self.vmm.read(src_ptr, byte_count))

    def __exit__(self, *_args):
        self._path.write_bytes(self._data)


class _Backend:
    def __init__(self, vmm, events=None, failure=False):
        self._vmm = vmm
        self._events = events if events is not None else []
        self._failure = failure
        self._sources = ()
        self.closed = False

    def start_restore(self, sources):
        self._sources = sources
        return self

    def restore(self, targets):
        if self._failure:
            raise RuntimeError("restore failed")
        for source in self._sources:
            target = targets[source.allocation_id]
            shard = Path(source.file_path).read_bytes()
            data = shard[source.file_offset : source.file_offset + source.byte_count]
            assert len(data) == target.byte_count
            self._vmm.write(target.va, data)

    def close(self):
        self.closed = True
        self._events.append("close")
        if self._failure:
            raise RuntimeError("transfer cleanup failed")


def _write_manifest(path, allocations, *, version=1, shard_bytes=bytes(128)):
    path.mkdir()
    manifest = weight_artifact.WeightArtifactManifest(version, tuple(allocations))
    (path / "manifest.json").write_bytes(msgspec.json.encode(manifest))
    (path / "shard.bin").write_bytes(shard_bytes)


Allocation = weight_artifact.WeightArtifactAllocation


@pytest.mark.timeout(10)
def test_byte_roundtrip_preserves_exact_allocations_on_fresh_server(
    tmp_path, monkeypatch
):
    socket_path = str(tmp_path / "weights.sock")
    artifact = tmp_path / "artifact"
    source_vmm = _ByteVMM()
    records = (
        AllocationRecord("weight-0", 64),
        AllocationRecord("weight-1", 128),
    )
    expected = {
        "weight-0": bytes(range(64)),
        "weight-1": bytes((255 - index) % 256 for index in range(128)),
    }
    _Writer.vmm = source_vmm
    monkeypatch.setattr(weight_artifact, "DeviceToFileWriter", _Writer)
    monkeypatch.setattr(weight_artifact, "get_vmm", lambda: source_vmm)

    with _server(socket_path, source_vmm):
        writer = _GMSClientSession(socket_path, RequestedLockType.RW)
        mappings = []
        for record in records:
            writer.allocate(record.allocation_id, record.aligned_size)
            mapping = weight_artifact._map_export(
                writer, record, source_vmm, 0, 64, GrantedLockType.RW
            )
            source_vmm.write(mapping[0].base, expected[record.allocation_id])
            mappings.append(mapping)
        writer.commit()
        manifest = weight_artifact.save_weights(
            str(artifact), socket_path, 0, shard_size_bytes=64
        )
        with pytest.raises(FileExistsError):
            weight_artifact.save_weights(str(artifact), socket_path, 0)
        for mapping in reversed(mappings):
            weight_artifact._release_mapping(source_vmm, mapping)
        writer.close()

    assert [
        (allocation.allocation_id, allocation.aligned_size)
        for allocation in manifest.allocations
    ] == [(record.allocation_id, record.aligned_size) for record in records]
    assert (artifact / "shards/shard_0000.bin").read_bytes() == expected["weight-0"]
    assert (artifact / "shards/shard_0001.bin").read_bytes() == expected["weight-1"]

    target_vmm = _ByteVMM()
    backend = _Backend(target_vmm)
    monkeypatch.setattr(weight_artifact, "get_vmm", lambda: target_vmm)
    monkeypatch.setattr(
        weight_artifact, "create_transfer_backend", lambda *_args, **_kwargs: backend
    )
    with _server(socket_path, target_vmm):
        weight_artifact.hydrate_weights(str(artifact), socket_path, 0)
        reader = _GMSClientSession(socket_path, RequestedLockType.RO)
        assert weight_artifact._list_allocations(reader) == records
        restored = [
            weight_artifact._map_export(
                reader, record, target_vmm, 0, 64, GrantedLockType.RO
            )
            for record in records
        ]
        assert {
            mapping.allocation_id: target_vmm.read(mapping.base, mapping.aligned_size)
            for mapping, _handle in restored
        } == expected
        for mapping in reversed(restored):
            weight_artifact._release_mapping(target_vmm, mapping)
        reader.close()
    assert backend.closed

    valid = Allocation("a", 64, "shard.bin", 0)
    for version, allocations, message in [
        (2, (valid,), "version"),
        (1, (), "no allocations"),
    ]:
        malformed = tmp_path / f"malformed-{version}-{len(allocations)}"
        _write_manifest(malformed, allocations, version=version)
        with pytest.raises(RuntimeError, match=message):
            weight_artifact.hydrate_weights(str(malformed), socket_path, 0)


class _FailingCleanupVMM(FakeVMM):
    def __init__(self, events):
        super().__init__(granularity=64)
        self._events = events

    def unmap(self, va, size):
        self._events.append("unmap")
        super().unmap(va, size)
        raise RuntimeError("unmap cleanup failed")

    def release(self, handle):
        imported = handle in self.imports
        super().release(handle)
        if imported:
            raise RuntimeError("handle cleanup failed")

    def address_free(self, va, size):
        super().address_free(va, size)
        raise RuntimeError("VA cleanup failed")


@pytest.mark.timeout(10)
def test_hydrate_preserves_transfer_error_and_attempts_all_cleanup(
    tmp_path, monkeypatch, caplog
):
    socket_path = str(tmp_path / "weights.sock")
    artifact = tmp_path / "artifact"
    allocations = (
        Allocation("weight-0", 64, "shard.bin", 0),
        Allocation("weight-1", 64, "shard.bin", 64),
    )
    _write_manifest(artifact, allocations)
    events = []
    vmm = _FailingCleanupVMM(events)
    caplog.set_level("ERROR")
    monkeypatch.setattr(weight_artifact, "get_vmm", lambda: vmm)
    monkeypatch.setattr(
        weight_artifact,
        "create_transfer_backend",
        lambda *_args, **_kwargs: _Backend(vmm, events, failure=True),
    )

    with _server(socket_path, vmm):
        with pytest.raises(RuntimeError, match="restore failed"):
            weight_artifact.hydrate_weights(str(artifact), socket_path, 0)
        assert not (vmm.imports or vmm.mapped or vmm.reservations or vmm.server_handles)
        _GMSClientSession(socket_path, RequestedLockType.RW).close()

    assert events.index("close") < events.index("unmap")
    assert events.count("unmap") == 2
    assert "resource cleanup failed" in caplog.text
