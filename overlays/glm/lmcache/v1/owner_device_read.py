# SPDX-License-Identifier: Apache-2.0
"""Submit already selected owner KV rows through the counted Tutti reader."""

# Standard
from typing import Protocol

# Third Party
import torch


class OwnerReadLoader(Protocol):
    """Public subset of the counted reader used by owner prefetch."""

    def device_read_queue_depth(self) -> int:
        """Return the configured queue depth without waiting on the GPU."""
        ...

    def supports_device_counted_read(self, capacity: int, io_nbytes: int) -> bool:
        """Return whether one batch fits the current queue and staging buffer."""
        ...

    def enqueue_device_counted_read(
        self,
        selected_ids: torch.Tensor,
        count: torch.Tensor,
        slba_table: torch.Tensor,
        destination_map: torch.Tensor,
        destination: torch.Tensor,
        resident_bitmap: torch.Tensor,
        input_ready_event: torch.cuda.Event,
    ) -> torch.cuda.Event:
        """Enqueue checked reads/scatter and return their completion event."""
        ...


def enqueue_owner_ids(
    loader: OwnerReadLoader,
    owner_ids: torch.Tensor,
    slba_table: torch.Tensor,
    destination_map: torch.Tensor,
    destination: torch.Tensor,
    resident_bitmap: torch.Tensor,
    prepare_stream: torch.cuda.Stream,
) -> torch.cuda.Event | None:
    """Enqueue an exact, normalized CPU owner subset without a host GPU wait.

    Args:
        loader: Counted reader retaining inputs and reservations until its drain.
        owner_ids: Sorted unique CPU int64 vector, already filtered by the caller's
            resident view. IDs are not reselected or exchanged between ranks.
        slba_table: Immutable CUDA int64 logical-block LBA table.
        destination_map: Immutable CUDA int64 logical-to-physical row table.
        destination: CUDA uint8 matrix; each KV row is contiguous.
        resident_bitmap: CUDA bool vector published by checked scatter.
        prepare_stream: Dedicated owner stream consuming joined CPU predictions;
            it must not wait for future model/indexer work.

    Returns:
        Event after checked read/scatter, or None for an empty subset or
        unsupported queue geometry. The caller keeps its existing no-read
        resident/gather ordering when no new device work is returned.

    Raises:
        ValueError: Invalid CPU IDs, tensor dtype, shape, device or row geometry.
        RuntimeError: A counted reader launch fails. Its normal drain/close must
            still run; errors are not converted into successful partial reads.

    Notes:
        No full-history bitmap is scanned. A single pinned H2D payload supplies
        IDs and count for each queue-bounded batch. Capacities use power-of-two
        buckets up to512, limiting candidate-count specialization. The caller
        must order downstream gather on the returned event and invoke the
        loader's normal forward/request drain before releasing request state.
    """
    if owner_ids.device.type != "cpu" or owner_ids.dtype != torch.int64:
        raise ValueError("owner IDs must be a CPU int64 vector")
    if owner_ids.ndim != 1:
        raise ValueError("owner IDs must be one-dimensional")
    if destination.ndim != 2 or destination.dtype != torch.uint8:
        raise ValueError("destination must be a CUDA byte matrix")
    tensors = (slba_table, destination_map, destination, resident_bitmap)
    if any(
        not tensor.is_cuda or tensor.device != destination.device for tensor in tensors
    ):
        raise ValueError("all device tensors must be on the destination GPU")
    if (
        slba_table.ndim != 1
        or destination_map.ndim != 1
        or resident_bitmap.ndim != 1
        or slba_table.dtype != torch.int64
        or destination_map.dtype != torch.int64
        or resident_bitmap.dtype != torch.bool
        or destination.stride(1) != 1
    ):
        raise ValueError("invalid logical tables or contiguous KV-row geometry")
    limit = min(slba_table.numel(), destination_map.numel(), resident_bitmap.numel())
    count = owner_ids.numel()
    if count == 0:
        return None
    # Host-only checks; the existing owner policy provides sorted unique IDs.
    if count and (int(owner_ids[0]) < 0 or int(owner_ids[-1]) >= limit):
        raise ValueError("owner ID outside the logical history table")
    queue_limit = min(512, loader.device_read_queue_depth() - 1)
    if queue_limit <= 0:
        return None
    max_capacity = 1 << (queue_limit.bit_length() - 1)
    capacity = min(max_capacity, 1 << (max(1, count) - 1).bit_length())
    if not loader.supports_device_counted_read(capacity, destination.shape[1]):
        return None
    with (
        torch.inference_mode(),
        torch.cuda.device(destination.device),
        torch.cuda.stream(prepare_stream),
    ):
        done = torch.cuda.Event()
        done.record(prepare_stream)
        for start in range(0, count, capacity):
            batch_count = min(capacity, count - start)
            # The last int64 stores a positive count whose low int32 is consumed
            # by the native API. CUDA and supported Linux hosts are little-endian.
            payload_host = torch.empty(capacity + 1, dtype=torch.int64, pin_memory=True)
            payload_host[:capacity].fill_(-1)
            payload_host[:batch_count].copy_(owner_ids[start : start + batch_count])
            payload_host[-1] = batch_count
            payload = payload_host.to(destination.device, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(prepare_stream)
            done = loader.enqueue_device_counted_read(
                payload[:capacity],
                payload[-1:].view(torch.int32)[:1],
                slba_table,
                destination_map,
                destination,
                resident_bitmap,
                ready,
            )
    return done
