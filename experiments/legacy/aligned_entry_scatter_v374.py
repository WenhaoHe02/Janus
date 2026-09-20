# SPDX-License-Identifier: Apache-2.0
"""Scatter exact logical bytes from padded NVMe DMA without touching append KV."""

# Third Party
import torch
import triton
import triton.language as tl

# First Party
from lmcache.v1.device_miss import DeviceSegmentPlan


@triton.jit
def _scatter_entries(
    staging,
    ids,
    count,
    entry_map,
    output,
    resident,
    groups,
    offsets,
    read_lengths,
    copy_lengths,
    source_offsets,
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
    tl.device_assert((n >= 0) & (n <= capacity), "invalid aligned command count")
    for work in range(
        tl.program_id(0), tl.minimum(n, capacity) * tiles, tl.num_programs(0)
    ):
        slot = work // tiles
        command = tl.load(ids + slot)
        ok = (command >= 0) & (command < commands)
        tl.device_assert(ok, "invalid aligned command ID")
        group = tl.load(groups + command, ok, other=-1)
        begin = tl.load(offsets + command, ok, other=0)
        read_len = tl.load(read_lengths + command, ok, other=0)
        length = tl.load(copy_lengths + command, ok, other=0)
        source = tl.load(source_offsets + command, ok, other=0)
        tl.device_assert(
            (group >= 0)
            & (begin >= 0)
            & (length > 0)
            & (begin + length <= width)
            & (source >= 0)
            & (source + length <= read_len)
            & (read_len <= staging_stride),
            "invalid aligned piece",
        )
        offset = (work % tiles) * tile + tl.arange(0, tile)
        logical_byte = begin + offset
        entry = group * entries_per_group + logical_byte // entry_bytes
        active = offset < length
        valid_entry = (entry >= 0) & (entry < map_entries) & (entry < resident_entries)
        tl.device_assert(~active | valid_entry, "invalid aligned logical entry")
        physical = tl.load(entry_map + entry, active & valid_entry, other=0)
        valid_physical = (physical >= 0) & (physical < physical_blocks * slots)
        tl.device_assert(~active | valid_physical, "invalid aligned destination")
        mask = active & valid_entry & valid_physical
        data = tl.load(staging + slot * staging_stride + source + offset, mask, other=0)
        target = (
            (physical // slots).to(tl.int64) * block_stride
            + (physical % slots) * slot_stride
            + logical_byte % entry_bytes
        )
        tl.store(output + target, data, mask)
        tl.store(resident + entry, True, mask & (logical_byte % entry_bytes == 0))


def scatter_aligned_device_entries(
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
    entry_copy_lengths: torch.Tensor,
    entry_source_offsets: torch.Tensor,
) -> None:
    """Queue a byte-exact scatter with the original stream/drain contract.

    Args:
        staging: GPU uint8 DMA staging storage, one stride per selected command.
        ids: Contiguous CUDA int64 command IDs.
        count: CUDA int32 initialized prefix length in ids.
        destination_map: CUDA int64 physical slot IDs for exactly the valid entries.
        destination: CUDA uint8 [blocks, slots, entry_bytes], HMA strides preserved.
        resident_bitmap: CUDA bool residency view bounded to valid history entries.
        staging_stride: Byte stride between independent DMA command buffers.
        segment_plan: Existing DMA group, byte offset and transfer-length tables.
        entries_per_group: Logical entries in each full group, including a short tail.
        entry_copy_lengths: CUDA int32 exact payload bytes per command.
        entry_source_offsets: CUDA int32 leading DMA padding bytes to skip.

    Raises:
        ValueError: Invalid shapes/dtypes/devices. Device-invalid values assert.

    Notes:
        Runs on the caller's current stream. The normal loader completion event
        must cover the entire scatter before any consumer uses resident entries.
    """
    commands = segment_plan.logical_ids.numel()
    tensors = (
        staging,
        ids,
        count,
        destination_map,
        destination,
        resident_bitmap,
        segment_plan.logical_ids,
        segment_plan.byte_offsets,
        segment_plan.byte_lengths,
        entry_copy_lengths,
        entry_source_offsets,
    )
    if (
        any(not t.is_cuda or t.device != destination.device for t in tensors)
        or destination.ndim != 3
        or destination.dtype != torch.uint8
        or staging.dtype != torch.uint8
        or staging_stride <= 0
        or destination.stride(2) != 1
        or entries_per_group <= 0
        or destination.shape[2] * entries_per_group > 65536
        or ids.dtype != torch.int64
        or destination_map.dtype != torch.int64
        or resident_bitmap.dtype != torch.bool
        or count.dtype != torch.int32
        or count.numel() != 1
        or not 0 < ids.numel() <= commands
        or ids.numel() * staging_stride > staging.numel()
        or min(destination_map.numel(), resident_bitmap.numel()) <= 0
        or any(
            t.dtype != torch.int32 or t.numel() != commands or not t.is_contiguous()
            for t in (
                entry_copy_lengths,
                entry_source_offsets,
                segment_plan.byte_offsets,
                segment_plan.byte_lengths,
            )
        )
        or any(
            not t.is_contiguous()
            for t in (ids, count, destination_map, resident_bitmap)
        )
    ):
        raise ValueError("invalid aligned entry scatter geometry")
    _scatter_entries[(min(128, ids.numel()),)](
        staging,
        ids,
        count,
        destination_map,
        destination,
        resident_bitmap,
        segment_plan.logical_ids,
        segment_plan.byte_offsets,
        segment_plan.byte_lengths,
        entry_copy_lengths,
        entry_source_offsets,
        capacity=ids.numel(),
        commands=commands,
        map_entries=destination_map.numel(),
        resident_entries=resident_bitmap.numel(),
        physical_blocks=destination.shape[0],
        slots=destination.shape[1],
        entry_bytes=destination.shape[2],
        entries_per_group=entries_per_group,
        block_stride=destination.stride(0),
        slot_stride=destination.stride(1),
        staging_stride=staging_stride,
        tile=256,
    )
