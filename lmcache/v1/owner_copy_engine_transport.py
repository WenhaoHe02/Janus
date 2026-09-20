# SPDX-License-Identifier: Apache-2.0
"""Opt-in byte-exact owner packets over CUDA IPC copy engines.

This transport intentionally uses a bounded CPU control barrier, not GPU memory
wait-value commands. Collective entry must remain in identical layer order on
all ranks. It is an experimental transport, not an automatically selected mode.
"""

from __future__ import annotations

# Standard
from dataclasses import dataclass
from datetime import timedelta
import math
from threading import RLock
from typing import Any

# Third Party
import torch
import torch.distributed as dist

# First Party
from lmcache.v1.owner_kv_packet import (
    owner_packet_layout,
    pack_owner_rows,
    scatter_owner_packets,
)


@dataclass
class _PeerSlot:
    send: torch.Tensor
    receive: torch.Tensor
    peer_receive: tuple[int, ...]
    peer_devices: tuple[int, ...]
    claims: torch.Tensor | None = None


class OwnerCopyEngineTransport:
    """Copy full aligned packets, then confirm completion on a Gloo group.

    Args:
        native_ops: Native CUDA IPC allocator/handle/peer-copy bindings.
        control_group: Dedicated Gloo process group; never the model NCCL group.
        rank: Rank within this group.
        world_size: Number of peers.
        slots: At least two receive slots, preventing overwrite during routing.
        copy_streams: Number of copy streams per GPU.
        timeout_seconds: Deadline for the CPU collective completion wait.
    Raises:
        ValueError: Group identity or buffering parameters are invalid.
        RuntimeError: Required native bindings are absent.
    Notes:
        Each call waits for the preceding local scatter before submitting new
        copies. The previous call's all-peer copy barrier proves every peer had
        completed the scatter two calls ago. With >=2 slots, the receive slot
        about to be overwritten is therefore no longer being read on any peer.
        No host/pointer flags are polled by GPU kernels or memory wait commands.
        A failed control operation fails closed; it is not reported as a valid
        partial gather. The caller must abort the request on that exception.
    """

    def __init__(
        self,
        native_ops: Any,
        control_group: Any,
        *,
        rank: int,
        world_size: int,
        slots: int = 2,
        copy_streams: int = 4,
        timeout_seconds: float = 30.0,
    ) -> None:
        if (
            slots < 2
            or not 0 <= rank < world_size
            or copy_streams < 1
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("Invalid owner CE transport parameters")
        if (
            dist.get_rank(control_group) != rank
            or dist.get_world_size(control_group) != world_size
        ):
            raise ValueError("Control group identity differs from the owner group")
        if dist.get_backend(control_group) != "gloo":
            raise ValueError("Owner CE control group must use Gloo")
        for name in (
            "allocate_cuda_ipc_buffer",
            "cuda_ipc_get_mem_handle",
            "cuda_ipc_open_mem_handle",
            "cuda_ipc_close_mem_handle",
            "cuda_memcpy_peer_async",
        ):
            if not callable(getattr(native_ops, name, None)):
                raise RuntimeError(f"Missing native owner CE binding: {name}")
        self.native_ops = native_ops
        self.control_group = control_group
        self.rank = rank
        self.world_size = world_size
        self.slot_count = slots
        self.copy_stream_count = min(copy_streams, world_size)
        self.timeout = timedelta(seconds=timeout_seconds)
        self._lock = RLock()
        self._slots: list[_PeerSlot] = []
        self._exports: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._imports: list[int] = []
        self._copies: list[torch.cuda.Stream] = []
        self._route: torch.cuda.Stream | None = None
        self._previous_done: torch.cuda.Event | None = None
        self._device: torch.device | None = None
        self._capacity = 0
        self._row_bytes = 0
        self._cursor = 0
        self._healthy = True

    def warm(self, *, capacity: int, row_bytes: int, device: torch.device) -> None:
        """Allocate dedicated cudaMalloc IPC buffers and exchange handles once.

        Args:
            capacity: Maximum contributed rows per rank in this transport.
            row_bytes: KV bytes in each row; must be divisible by eight.
            device: Explicit CUDA device for this worker.
        Raises:
            ValueError: Geometry is invalid or differs from an existing warmup.
            RuntimeError: Allocation, IPC mapping, or the control group fails.
        """
        if device.type != "cuda" or device.index is None or row_bytes % 8:
            raise ValueError(
                "Owner CE requires an explicit CUDA device and aligned rows"
            )
        _, packet_bytes = owner_packet_layout(capacity, row_bytes)
        with self._lock, torch.cuda.device(device), torch.inference_mode():
            if not self._healthy:
                raise RuntimeError("Closed or failed owner CE transport cannot warm")
            if self._slots:
                if (capacity, row_bytes, device) != (
                    self._capacity,
                    self._row_bytes,
                    self._device,
                ):
                    raise ValueError("Warm geometry cannot change in place")
                return
            # Keep partially exported allocations alive if a peer's setup
            # fails. Only a fully successful warmup enables orderly close.
            self._healthy = False
            self._capacity, self._row_bytes, self._device = capacity, row_bytes, device
            self._route = torch.cuda.Stream(device=device)
            self._copies = [
                torch.cuda.Stream(device=device) for _ in range(self.copy_stream_count)
            ]
            local = self._exports
            descriptors = []
            for _ in range(self.slot_count):
                send = self.native_ops.allocate_cuda_ipc_buffer(
                    packet_bytes, device.index
                )
                receive = self.native_ops.allocate_cuda_ipc_buffer(
                    packet_bytes * self.world_size, device.index
                )
                local.append((send, receive))
                descriptors.append(
                    (self.native_ops.cuda_ipc_get_mem_handle(receive), device.index)
                )
            peers: list[Any] = [None] * self.world_size
            # Group creation must also use a finite timeout. This setup is not
            # in the request path and carries handles, never prediction results.
            dist.all_gather_object(peers, descriptors, group=self.control_group)
            for index, (send, receive) in enumerate(local):
                pointers, devices = [], []
                for peer, descriptor in enumerate(peers):
                    handle, peer_device = descriptor[index]
                    devices.append(int(peer_device))
                    pointer = (
                        receive.data_ptr()
                        if peer == self.rank
                        else self.native_ops.cuda_ipc_open_mem_handle(
                            handle, device.index
                        )
                    )
                    pointers.append(pointer)
                    if peer != self.rank:
                        self._imports.append(pointer)
                self._slots.append(
                    _PeerSlot(send, receive, tuple(pointers), tuple(devices))
                )
            # Force lazy peer-copy setup during warmup, not the first user hit.
            with torch.cuda.stream(self._route):
                self._slots[0].send[:1].zero_()
                ready = torch.cuda.Event()
                ready.record(self._route)
            for peer in range(self.world_size):
                stream = self._copies[peer % len(self._copies)]
                stream.wait_event(ready)
                self.native_ops.cuda_memcpy_peer_async(
                    self._slots[0].peer_receive[peer] + self.rank * packet_bytes,
                    self._slots[0].peer_devices[peer],
                    self._slots[0].send.data_ptr(),
                    device.index,
                    1,
                    stream.cuda_stream,
                )
            for stream in self._copies:
                stream.synchronize()
            self._barrier()
            self._healthy = True

    def gather(
        self,
        *,
        local_ids: torch.Tensor,
        local_ready: bool,
        padded_blocks: int,
        source: torch.Tensor,
        logical_rows: torch.Tensor,
        destination: torch.Tensor,
        resident: torch.Tensor,
        input_ready: torch.cuda.Event | None = None,
    ) -> tuple[torch.cuda.Event, torch.Tensor, torch.Tensor]:
        """Copy and route one rank-aligned generation of predicted KV.

        Args:
            local_ids: CPU or CUDA int64 rank-local logical IDs.
            local_ready: False publishes a failed rank, without any valid KV.
            padded_blocks: Identical per-call capacity on every rank, <= warm capacity.
            source: uint8 physical KV rows belonging to this worker.
            logical_rows: CUDA int64 logical-to-physical mapping.
            destination: uint8 physical KV destination, possibly identical to source.
            resident: CUDA bool logical residency bitmap, updated by routing.
            input_ready: Source-ready event; captured on the calling stream if omitted.
        Returns:
            Scatter completion event, selected logical IDs and GPU all-ready flag.
            The caller must await the returned event before consuming destination.
        Raises:
            ValueError: Shape/device contract is violated.
            RuntimeError: Transport is unavailable or copy/control completion fails.
        Notes:
            Calls must have the same order and packet geometry on every rank.
            This performs no global top-K merge. The CPU barrier acknowledges
            completion of KV copies, not agreement on the selected candidates.
        """
        with self._lock, torch.inference_mode():
            if (
                not self._healthy
                or not self._slots
                or self._route is None
                or self._device is None
            ):
                raise RuntimeError("Owner CE transport is not available")
            if (
                not 0 < padded_blocks <= self._capacity
                or source.dtype != torch.uint8
                or destination.dtype != torch.uint8
                or source.shape[1] != self._row_bytes
                or destination.shape[1] != self._row_bytes
                or source.stride(1) != 1
                or destination.stride(1) != 1
                or any(
                    t.device != self._device
                    for t in (source, logical_rows, destination, resident)
                )
            ):
                self._healthy = False
                raise ValueError("Owner CE packet geometry/device mismatch")
            try:
                with torch.cuda.device(self._device):
                    if self._previous_done is not None:
                        self._previous_done.synchronize()
                    slot = self._slots[self._cursor]
                    self._cursor = (self._cursor + 1) % len(self._slots)
                    _, packet_bytes = owner_packet_layout(
                        padded_blocks, self._row_bytes
                    )
                    if input_ready is None:
                        input_ready = torch.cuda.Event()
                        input_ready.record(torch.cuda.current_stream(self._device))
                    local = local_ids
                    if local.device.type == "cpu" and not local.is_pinned():
                        local = local.pin_memory()
                    with torch.cuda.stream(self._route):
                        self._route.wait_event(input_ready)
                        local = local.to(
                            self._device, dtype=torch.int64, non_blocking=True
                        )
                        send = slot.send[:packet_bytes]
                        pack_owner_rows(
                            source,
                            logical_rows,
                            local,
                            send,
                            local_ready=bool(
                                local_ready and local.numel() <= padded_blocks
                            ),
                            padded_blocks=padded_blocks,
                        )
                        packed = torch.cuda.Event()
                        packed.record(self._route)
                    for peer in range(self.world_size):
                        stream = self._copies[peer % len(self._copies)]
                        stream.wait_event(packed)
                        self.native_ops.cuda_memcpy_peer_async(
                            slot.peer_receive[peer] + self.rank * packet_bytes,
                            slot.peer_devices[peer],
                            send.data_ptr(),
                            self._device.index,
                            packet_bytes,
                            stream.cuda_stream,
                        )
                    for stream in self._copies:
                        stream.synchronize()
                    self._barrier()
                    with torch.cuda.stream(self._route):
                        limit = min(logical_rows.numel(), resident.numel())
                        if slot.claims is None or slot.claims.numel() != limit:
                            slot.claims = torch.empty(
                                limit, dtype=torch.int32, device=self._device
                            )
                        received = slot.receive[: packet_bytes * self.world_size]
                        selected = scatter_owner_packets(
                            received,
                            slot.claims,
                            logical_rows,
                            destination,
                            resident,
                            padded_blocks=padded_blocks,
                            world_size=self.world_size,
                        )
                        all_ready = (
                            received.view(torch.int64)[:: packet_bytes // 8] != -2
                        ).all()
                        done = torch.cuda.Event()
                        done.record(self._route)
                        self._previous_done = done
                    return done, selected, all_ready
            except Exception:
                self._healthy = False
                raise

    def close(self) -> None:
        """Collectively close IPC mappings before releasing exported storage.

        All ranks must call this after their final successful gather. After a
        transport failure, raise without freeing buffers that peers might still
        access; the failed worker group must be terminated together by its owner.

        Raises:
            RuntimeError: A prior failure prevents orderly collective teardown.
        """
        with self._lock:
            if not self._slots and not self._exports:
                return
            if not self._healthy or self._device is None:
                raise RuntimeError(
                    "Failed owner CE group requires coordinated worker termination"
                )
            self._healthy = False
            with torch.cuda.device(self._device):
                if self._previous_done is not None:
                    self._previous_done.synchronize()
                for stream in self._copies:
                    stream.synchronize()
                self._barrier()
                for pointer in self._imports:
                    self.native_ops.cuda_ipc_close_mem_handle(
                        pointer, self._device.index
                    )
                self._barrier()
                self._slots.clear()
                self._exports.clear()
                self._imports.clear()
                self._copies.clear()
                self._route = None
                self._healthy = False

    def _barrier(self) -> None:
        work = dist.barrier(group=self.control_group, async_op=True)
        if not work.wait(timeout=self.timeout):
            raise RuntimeError("Owner CE control barrier did not complete")
