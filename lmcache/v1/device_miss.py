# SPDX-License-Identifier: Apache-2.0
"""GPU-resident authoritative miss selection for counted Tutti reads."""

# Standard
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass

# Third Party
import torch
import triton
import triton.language as tl


def resolve_block_segments(
    extents: Sequence[tuple[int, int, int]],
    *,
    first_block: int,
    n_blocks: int,
    file_offset: int,
    block_bytes: int,
    physical_rows: Sequence[int],
) -> list[tuple[int, int, int, int, int]]:
    """Resolve blocks to exact (logical, SLBA, bytes, physical, offset) pieces.

    Args:
        extents: File byte offset, starting LBA, and length in512-byte sectors.
        first_block: First logical block ID.
        n_blocks: Number of consecutive blocks to resolve.
        file_offset: File offset of the first block, including layer/payload offset.
        block_bytes: Sector-aligned bytes per logical block.
        physical_rows: Destination row per block, or empty for identity mapping.

    Returns:
        Ordered command rows, with every byte of every block covered once.

    Raises:
        ValueError: Unaligned, overlapping, missing, or invalid extent geometry.
    """
    if (
        first_block < 0
        or n_blocks <= 0
        or file_offset < 0
        or file_offset % 512
        or block_bytes <= 0
        or block_bytes > 65536
        or block_bytes % 512
        or (physical_rows and len(physical_rows) != n_blocks)
    ):
        raise ValueError("invalid block segment geometry")
    ordered = sorted(extents)
    ends = []
    for offset, slba, sectors in ordered:
        if offset < 0 or offset % 512 or slba < 0 or sectors <= 0:
            raise ValueError("invalid raw extent")
        if ends and offset < ends[-1]:
            raise ValueError("overlapping raw extents")
        ends.append(offset + sectors * 512)
    rows = []
    for index in range(n_blocks):
        logical = first_block + index
        physical = int(physical_rows[index]) if physical_rows else logical
        if physical < 0:
            raise ValueError("invalid physical destination")
        start = file_offset + index * block_bytes
        position = start
        end = start + block_bytes
        extent = bisect_right(ends, position)
        while position < end:
            if extent >= len(ordered) or position < ordered[extent][0]:
                raise ValueError("raw extent gap inside a KV block")
            offset, slba, _ = ordered[extent]
            size = min(end, ends[extent]) - position
            rows.append(
                (
                    logical,
                    slba + (position - offset) // 512,
                    size,
                    physical,
                    position - start,
                )
            )
            position += size
            extent += 1
    return rows


@dataclass(frozen=True)
class DeviceSegmentPlan:
    """Immutable per-command CUDA tables; each piece maps back to one KV block."""

    logical_ids: torch.Tensor
    slbas: torch.Tensor
    byte_lengths: torch.Tensor
    destination_rows: torch.Tensor
    byte_offsets: torch.Tensor

    @classmethod
    def from_rows(
        cls, rows: Sequence[tuple[int, int, int, int, int]], device: torch.device
    ) -> "DeviceSegmentPlan":
        """Upload validated resolve_block_segments rows on the current stream.

        Args:
            rows: Nonempty resolved command rows.
            device: Destination CUDA device.
        Returns:
            Contiguous CUDA vectors owning the immutable command plan.
        Raises:
            ValueError: If rows are empty or the destination is not CUDA.
        """
        if not rows or device.type != "cuda":
            raise ValueError("segment plan requires rows and a CUDA device")
        table = torch.tensor(rows, dtype=torch.int64, device=device)
        return cls(
            table[:, 0].contiguous(),
            table[:, 1].contiguous(),
            table[:, 2].to(torch.int32),
            table[:, 3].contiguous(),
            table[:, 4].to(torch.int32),
        )


@triton.jit
def _pack_segments(
    seen,
    resident,
    logical_ids,
    ids,
    count,
    n: tl.constexpr,
    bitmap_size: tl.constexpr,
    tile: tl.constexpr,
):
    i = tl.arange(0, tile)
    logical = tl.load(logical_ids + i, i < n, other=0)
    tl.device_assert(
        (i >= n) | ((logical >= 0) & (logical < bitmap_size)),
        "invalid segment logical ID",
    )
    missing = (
        (i < n)
        & (tl.load(seen + logical, i < n, other=0) != 0)
        & ~tl.load(resident + logical, i < n, other=True)
    )
    position = tl.cumsum(missing.to(tl.int32)) - 1
    tl.store(ids + position, i.to(tl.int64), missing)
    tl.store(count, tl.sum(missing.to(tl.int32)))


def pack_device_segments(
    selected_bitmap: torch.Tensor,
    resident_bitmap: torch.Tensor,
    logical_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select all physical pieces of each selected, nonresident logical block.

    Args:
        selected_bitmap: CUDA bool/int32 authoritative union.
        resident_bitmap: CUDA bool logical residency bitmap.
        logical_ids: CUDA int64 command-to-logical-block map, at most4096 entries.
    Returns:
        Fixed-capacity command IDs and GPU int32 count. Tail IDs are undefined.
    Raises:
        ValueError: Invalid vector geometry/dtype/device; bad IDs assert on GPU.
    """
    if (
        not logical_ids.is_cuda
        or logical_ids.dtype != torch.int64
        or not 0 < logical_ids.numel() <= 4096
        or selected_bitmap.dtype not in (torch.bool, torch.int32)
        or resident_bitmap.dtype != torch.bool
        or any(
            t.ndim != 1 or not t.is_contiguous() or t.device != logical_ids.device
            for t in (logical_ids, selected_bitmap, resident_bitmap)
        )
    ):
        raise ValueError("invalid device segment selection")
    ids = torch.empty_like(logical_ids)
    count = torch.empty(1, dtype=torch.int32, device=ids.device)
    _pack_segments[(1,)](
        selected_bitmap,
        resident_bitmap,
        logical_ids,
        ids,
        count,
        n=ids.numel(),
        bitmap_size=min(selected_bitmap.numel(), resident_bitmap.numel()),
        tile=triton.next_power_of_2(ids.numel()),
        debug=True,
    )
    return ids, count


@triton.jit
def _pack_misses(seen, resident, ids, count, n: tl.constexpr, tile: tl.constexpr):
    i = tl.arange(0, tile)
    missing = (tl.load(seen + i, i < n, other=0) != 0) & ~tl.load(
        resident + i, i < n, other=True
    )
    position = tl.cumsum(missing.to(tl.int32)) - 1
    tl.store(ids + position, i.to(tl.int64), missing)
    tl.store(count, tl.sum(missing.to(tl.int32)))


def pack_device_misses(
    selected_bitmap: torch.Tensor,
    resident_bitmap: torch.Tensor,
    history_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fixed-capacity sorted missing IDs and their GPU int32 count.

    Args:
        selected_bitmap: One-dimensional CUDA bool/int32 authoritative union.
        resident_bitmap: CUDA bool bitmap on the same device.
        history_blocks: Registered history prefix to inspect, in [1, 4096].

    Returns:
        int64 IDs[history_blocks] and int32 count[1]. Only IDs[:count] are
        initialized/valid; never inspect the tail. No host scalar read occurs.

    Raises:
        ValueError: If geometry, dtype or devices violate the contract.

    Notes:
        Caller orders its stream after true indexer and resident updates.
        Duplicate indexer hits are already folded in selected_bitmap.
    """
    if (
        not 0 < history_blocks <= 4096
        or selected_bitmap.ndim != 1
        or resident_bitmap.ndim != 1
        or not selected_bitmap.is_cuda
        or selected_bitmap.device != resident_bitmap.device
        or selected_bitmap.dtype not in (torch.bool, torch.int32)
        or resident_bitmap.dtype != torch.bool
        or not selected_bitmap.is_contiguous()
        or not resident_bitmap.is_contiguous()
        or min(selected_bitmap.numel(), resident_bitmap.numel()) < history_blocks
    ):
        raise ValueError("invalid device miss bitmap geometry")
    ids = torch.empty(history_blocks, dtype=torch.int64, device=selected_bitmap.device)
    count = torch.empty(1, dtype=torch.int32, device=selected_bitmap.device)
    _pack_misses[(1,)](
        selected_bitmap,
        resident_bitmap,
        ids,
        count,
        n=history_blocks,
        tile=triton.next_power_of_2(history_blocks),
    )
    return ids, count


@triton.jit
def _scatter_counted(
    staging,
    ids,
    count,
    destination_map,
    output,
    resident,
    segment_logical,
    segment_offsets,
    segment_lengths,
    output_rows: tl.constexpr,
    map_rows: tl.constexpr,
    width: tl.constexpr,
    output_stride: tl.constexpr,
    staging_stride: tl.constexpr,
    tile: tl.constexpr,
    segmented: tl.constexpr,
    resident_size: tl.constexpr,
):
    tiles: tl.constexpr = (width + tile - 1) // tile
    for work in range(tl.program_id(0), tl.load(count) * tiles, tl.num_programs(0)):
        slot = work // tiles
        logical = tl.load(ids + slot)
        tl.device_assert((logical >= 0) & (logical < map_rows), "invalid logical KV ID")
        physical = tl.load(destination_map + logical)
        tl.device_assert(
            (physical >= 0) & (physical < output_rows), "invalid physical KV row"
        )
        piece_offset = 0
        piece_length = width
        resident_id = logical
        if segmented:
            piece_offset = tl.load(segment_offsets + logical)
            piece_length = tl.load(segment_lengths + logical)
            resident_id = tl.load(segment_logical + logical)
            tl.device_assert(
                (piece_offset >= 0)
                & (piece_length > 0)
                & (piece_offset + piece_length <= width),
                "invalid KV piece bounds",
            )
            tl.device_assert(
                (resident_id >= 0) & (resident_id < resident_size),
                "invalid KV piece resident ID",
            )
        offset = (work % tiles) * tile + tl.arange(0, tile)
        data = tl.load(
            staging + slot * staging_stride + offset, offset < piece_length, other=0
        )
        tl.store(
            output + physical * output_stride + piece_offset + offset,
            data,
            offset < piece_length,
        )
        if work % tiles == 0:
            tl.store(resident + resident_id, True)


def scatter_device_misses(
    staging: torch.Tensor,
    ids: torch.Tensor,
    count: torch.Tensor,
    destination_map: torch.Tensor,
    destination: torch.Tensor,
    resident_bitmap: torch.Tensor,
    staging_stride: int,
    *,
    segment_plan: DeviceSegmentPlan | None = None,
) -> None:
    """Scatter counted, successfully read rows and publish their resident bits.

    Args:
        staging: Flat contiguous CUDA uint8 staging allocation.
        ids: Fixed-capacity CUDA int64 logical IDs from pack_device_misses.
        count: CUDA int32[1] initialized count in [0, ids.numel()].
        destination_map: CUDA int64 logical-to-physical row map.
        destination: CUDA uint8[physical_rows, bytes_per_block], contiguous
            within each row; physical rows may have HMA padding between them.
        resident_bitmap: CUDA bool logical resident bitmap.
        staging_stride: Bytes reserved for each staging row.
        segment_plan: Optional per-command logical IDs, byte offsets and sizes.
            When present, IDs index commands, not logical blocks.

    Returns:
        None. Completion of the current stream publishes both KV and bits.

    Raises:
        ValueError: If metadata geometry is invalid. Bad device IDs assert.

    Notes:
        Must follow counted NVMe status validation on the SAME I/O stream.
        Consumers must wait for an event after this entire kernel, never
        independently poll resident bits while its row tiles are still running.
    """
    vectors = (staging, ids, count, destination_map, resident_bitmap)
    if (
        destination.ndim != 2
        or destination.dtype != torch.uint8
        or not destination.is_cuda
        or destination.stride(1) != 1
        or destination.stride(0) < destination.shape[1]
        or any(
            t.ndim != 1 or not t.is_contiguous() or t.device != destination.device
            for t in vectors
        )
        or staging.dtype != torch.uint8
        or ids.dtype != torch.int64
        or count.dtype != torch.int32
        or count.numel() != 1
        or destination_map.dtype != torch.int64
        or resident_bitmap.dtype != torch.bool
        or (segment_plan is None and destination_map.numel() > resident_bitmap.numel())
        or destination.shape[1] <= 0
        or staging_stride < destination.shape[1]
        or ids.numel() * staging_stride > staging.numel()
    ):
        raise ValueError("invalid counted scatter geometry")
    if segment_plan is not None:
        pieces = (
            segment_plan.logical_ids,
            segment_plan.byte_offsets,
            segment_plan.byte_lengths,
        )
        if any(
            t.ndim != 1
            or not t.is_contiguous()
            or t.device != destination.device
            or t.numel() != destination_map.numel()
            for t in pieces
        ):
            raise ValueError("invalid scatter segment vectors")
    _scatter_counted[(min(128, ids.numel()),)](
        staging,
        ids,
        count,
        destination_map,
        destination,
        resident_bitmap,
        segment_plan.logical_ids if segment_plan is not None else ids,
        segment_plan.byte_offsets if segment_plan is not None else ids,
        segment_plan.byte_lengths if segment_plan is not None else ids,
        output_rows=destination.shape[0],
        map_rows=destination_map.numel(),
        width=destination.shape[1],
        output_stride=destination.stride(0),
        staging_stride=staging_stride,
        tile=4096,
        segmented=segment_plan is not None,
        resident_size=resident_bitmap.numel(),
        debug=True,
    )


@triton.jit
def _scatter_counted_entries(
    staging,
    ids,
    count,
    entry_map,
    output,
    resident,
    logical_groups,
    piece_offsets,
    piece_lengths,
    capacity: tl.constexpr,
    commands: tl.constexpr,
    map_entries: tl.constexpr,
    resident_entries: tl.constexpr,
    physical_blocks: tl.constexpr,
    slots: tl.constexpr,
    entry_bytes: tl.constexpr,
    entries_per_group: tl.constexpr,
    block_stride: tl.constexpr,
    slot_stride: tl.constexpr,
    staging_stride: tl.constexpr,
    tile: tl.constexpr,
):
    width: tl.constexpr = entry_bytes * entries_per_group
    tiles: tl.constexpr = (width + tile - 1) // tile
    n = tl.load(count)
    tl.device_assert((n >= 0) & (n <= capacity), "invalid HCA command count")
    for work in range(
        tl.program_id(0), tl.minimum(n, capacity) * tiles, tl.num_programs(0)
    ):
        slot = work // tiles
        command = tl.load(ids + slot)
        ok = (command >= 0) & (command < commands)
        tl.device_assert(ok, "invalid HCA command ID")
        group = tl.load(logical_groups + command, ok, other=-1)
        begin = tl.load(piece_offsets + command, ok, other=0)
        length = tl.load(piece_lengths + command, ok, other=0)
        tl.device_assert(
            (group >= 0) & (begin >= 0) & (length > 0) & (begin + length <= width),
            "invalid HCA piece",
        )
        offset = (work % tiles) * tile + tl.arange(0, tile)
        logical_byte = begin + offset
        entry = group * entries_per_group + logical_byte // entry_bytes
        active = offset < length
        valid_entry = (entry >= 0) & (entry < map_entries) & (entry < resident_entries)
        tl.device_assert(~active | valid_entry, "invalid HCA logical entry")
        physical = tl.load(entry_map + entry, active & valid_entry, other=0)
        valid_physical = (physical >= 0) & (physical < physical_blocks * slots)
        tl.device_assert(~active | valid_physical, "invalid HCA destination")
        mask = active & valid_entry & valid_physical
        data = tl.load(staging + slot * staging_stride + offset, mask, other=0)
        target = (
            (physical // slots).to(tl.int64) * block_stride
            + (physical % slots) * slot_stride
            + logical_byte % entry_bytes
        )
        tl.store(output + target, data, mask)
        # Exact nonoverlapping pieces contain each entry's first byte once.
        # Consumers wait for the WHOLE scatter event, never for this bit alone.
        tl.store(resident + entry, True, mask & (logical_byte % entry_bytes == 0))


def scatter_device_entries(
    staging: torch.Tensor,
    ids: torch.Tensor,
    count: torch.Tensor,
    destination_map: torch.Tensor,
    destination: torch.Tensor,
    resident_bitmap: torch.Tensor,
    staging_stride: int,
    *,
    segment_plan: DeviceSegmentPlan,
    entries_per_group: int,
) -> None:
    """Scatter checked sector-aligned groups into strided HCA entry slots.

    Args:
        staging: Contiguous CUDA uint8 NVMe staging bytes.
        ids: CUDA int64 command IDs; only the GPU count prefix is valid.
        count: CUDA int32[1] command count, bounded by ids capacity.
        destination_map: CUDA int64 logical-entry to physical-entry mapping.
        destination: CUDA uint8[physical blocks, slots, entry bytes] cache.
        resident_bitmap: CUDA bool logical-entry residency.
        staging_stride: Bytes reserved per command in staging.
        segment_plan: Complete nonoverlapping pieces of each logical group.
        entries_per_group: Fixed entry count in one sector-aligned group.

    Returns:
        None. The caller records a completion event after this operation.

    Raises:
        ValueError: Unsupported shape, dtype, device or stride.
        RuntimeError: CUDA launch failure; invalid GPU indices assert.

    Notes:
        Run after NVMe status validation on the same I/O stream. An entry may
        span physical extents; byte offsets reassemble it exactly. Never
        reshape the strided cache into a copied flat tensor.
    """
    vectors = (
        staging,
        ids,
        count,
        destination_map,
        resident_bitmap,
        segment_plan.logical_ids,
        segment_plan.byte_offsets,
        segment_plan.byte_lengths,
    )
    if (
        destination.ndim != 3
        or not destination.is_cuda
        or destination.dtype != torch.uint8
        or destination.stride(2) != 1
        or entries_per_group <= 0
        or min(destination.shape) <= 0
        or destination.stride(1) < destination.shape[2]
        or destination.stride(0) < destination.shape[1] * destination.stride(1)
        or any(
            t.ndim != 1 or not t.is_contiguous() or t.device != destination.device
            for t in vectors
        )
        or staging.dtype != torch.uint8
        or ids.dtype != torch.int64
        or count.dtype != torch.int32
        or count.numel() != 1
        or destination_map.dtype != torch.int64
        or resident_bitmap.dtype != torch.bool
        or segment_plan.logical_ids.dtype != torch.int64
        or segment_plan.byte_offsets.dtype != torch.int32
        or segment_plan.byte_lengths.dtype != torch.int32
        or not (0 < ids.numel() <= segment_plan.logical_ids.numel())
        or segment_plan.logical_ids.numel() != segment_plan.byte_offsets.numel()
        or segment_plan.logical_ids.numel() != segment_plan.byte_lengths.numel()
        or ids.numel() * staging_stride > staging.numel()
        or staging_stride < entries_per_group * destination.shape[2]
        or min(destination_map.numel(), resident_bitmap.numel()) <= 0
    ):
        raise ValueError("invalid counted entry scatter geometry")
    _scatter_counted_entries[(min(128, ids.numel()),)](
        staging,
        ids,
        count,
        destination_map,
        destination,
        resident_bitmap,
        segment_plan.logical_ids,
        segment_plan.byte_offsets,
        segment_plan.byte_lengths,
        capacity=ids.numel(),
        commands=segment_plan.logical_ids.numel(),
        map_entries=destination_map.numel(),
        resident_entries=resident_bitmap.numel(),
        physical_blocks=destination.shape[0],
        slots=destination.shape[1],
        entry_bytes=destination.shape[2],
        entries_per_group=entries_per_group,
        block_stride=destination.stride(0),
        slot_stride=destination.stride(1),
        staging_stride=staging_stride,
        tile=4096,
        debug=True,
    )
