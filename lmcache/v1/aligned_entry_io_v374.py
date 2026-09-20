# SPDX-License-Identifier: Apache-2.0
"""Plan sector-aligned DMA while copying only the exact logical entry bytes."""

# Standard
from bisect import bisect_right
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class EntryIOPlan:
    """DMA rows plus exact per-command source offsets and logical copy lengths."""

    rows: tuple[tuple[int, int, int, int, int], ...]
    source_offsets: tuple[int, ...]
    copy_lengths: tuple[int, ...]


def resolve_entry_io(
    extents: Sequence[tuple[int, int, int]],
    *,
    file_offset: int,
    entries: int,
    entry_bytes: int,
    entries_per_group: int,
) -> EntryIOPlan:
    """Resolve complete logical entries, including unaligned starts and tails.

    Args:
        extents: File byte offset, starting LBA and length in 512-byte sectors.
        file_offset: Byte offset of this layer, not necessarily sector-aligned.
        entries: Exact number of logical history entries; no padded entry is added.
        entry_bytes: Bytes in one logical entry.
        entries_per_group: Logical grouping used by the destination scatter.

    Returns:
        NVMe command rows in DeviceSegmentPlan order, source offsets into each
        DMA result, and exact byte counts to copy. DMA padding is never published.

    Raises:
        ValueError: Invalid geometry, overlapping extents or missing coverage.

    Notes:
        Each DMA command stays inside one registered extent and is at most
        64 KiB. Adjacent commands may overlap on-disk padding but their logical
        destination intervals never overlap. No byte is read from an unmapped gap.
    """
    width = entry_bytes * entries_per_group
    if file_offset < 0 or entries <= 0 or entry_bytes <= 0 or not 0 < width <= 65536:
        raise ValueError("invalid entry geometry")
    ordered = sorted(extents)
    ends = []
    for offset, slba, sectors in ordered:
        if offset < 0 or offset % 512 or slba < 0 or sectors <= 0:
            raise ValueError("invalid raw extent")
        if ends and offset < ends[-1]:
            raise ValueError("overlapping raw extents")
        ends.append(offset + sectors * 512)
    rows = []
    source_offsets = []
    copy_lengths = []
    total_bytes = entries * entry_bytes
    for group in range((entries + entries_per_group - 1) // entries_per_group):
        begin = file_offset + group * width
        end = file_offset + min((group + 1) * width, total_bytes)
        position = begin
        extent_index = bisect_right(ends, position)
        while position < end:
            if extent_index >= len(ordered) or position < ordered[extent_index][0]:
                raise ValueError("raw extent gap inside logical entries")
            offset, slba, _ = ordered[extent_index]
            dma_start = position // 512 * 512
            copy_end = min(end, ends[extent_index], dma_start + 65536)
            dma_end = (copy_end + 511) // 512 * 512
            if (
                dma_start < offset
                or dma_end > ends[extent_index]
                or copy_end <= position
            ):
                raise ValueError("aligned DMA escapes its registered extent")
            rows.append(
                (
                    group,
                    slba + (dma_start - offset) // 512,
                    dma_end - dma_start,
                    group,
                    position - begin,
                )
            )
            source_offsets.append(position - dma_start)
            copy_lengths.append(copy_end - position)
            position = copy_end
            if position == ends[extent_index]:
                extent_index += 1
    return EntryIOPlan(tuple(rows), tuple(source_offsets), tuple(copy_lengths))
