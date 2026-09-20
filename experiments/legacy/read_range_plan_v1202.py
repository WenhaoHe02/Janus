# SPDX-License-Identifier: Apache-2.0
"""Reference contract for adjacent SSD reads with unchanged SGL GPU destinations."""

# Standard
from dataclasses import dataclass


@dataclass(frozen=True)
class ReadSpan:
    """One original READ: namespace/version, sector address, bytes and GPU IOVA."""

    namespace: int
    generation: str
    slba: int
    byte_length: int
    gpu_iova: int


@dataclass(frozen=True)
class ReadCommand:
    """One contiguous SSD READ with original per-block SGL destinations preserved."""

    namespace: int
    generation: str
    slba: int
    byte_length: int
    member_indices: tuple[int, ...]
    destinations: tuple[tuple[int, int], ...]


def coalesce_reads(spans: list[ReadSpan], *, sector_bytes: int,
                   max_command_bytes: int, max_sgl_entries: int,
                   enabled: bool = True) -> list[ReadCommand]:
    """Group adjacent selected reads, without sorting, over-reading, or deduping.

    Args:
        spans: Original submission order; each IOVA range is already registered.
        sector_bytes: Actual namespace logical sector size.
        max_command_bytes: Device-verified transfer limit, not an assumed MDTS.
        max_sgl_entries: Verified descriptor-list capacity per command.
        enabled: False emits one command per original span using the same format.
    Returns:
        Commands whose SGL entries preserve original GPU addresses and lengths.
    Raises:
        ValueError: Invalid limits or original spans that violate those limits.
    """
    if (sector_bytes <= 0 or sector_bytes & (sector_bytes - 1)
            or max_command_bytes <= 0 or max_command_bytes % sector_bytes
            or max_sgl_entries <= 0):
        raise ValueError('Invalid device-verified command/SGL limits')
    groups: list[ReadCommand] = []
    for index, span in enumerate(spans):
        if (not 0 < span.namespace <= 0xFFFFFFFF or span.slba < 0
                or span.byte_length <= 0 or span.byte_length % sector_bytes
                or span.byte_length > max_command_bytes or span.gpu_iova <= 0
                or span.slba + span.byte_length // sector_bytes > 1 << 64):
            raise ValueError('Original read is not a valid bounded namespace range')
        last = groups[-1] if groups else None
        merge = (enabled and last is not None
                 and last.namespace == span.namespace and last.generation == span.generation
                 and last.slba + last.byte_length // sector_bytes == span.slba
                 and last.byte_length + span.byte_length <= max_command_bytes
                 and len(last.member_indices) < max_sgl_entries)
        if merge:
            groups[-1] = ReadCommand(last.namespace, last.generation, last.slba,
                                     last.byte_length + span.byte_length,
                                     last.member_indices + (index,),
                                     last.destinations + ((span.gpu_iova, span.byte_length),))
        else:
            groups.append(ReadCommand(span.namespace, span.generation, span.slba,
                                      span.byte_length, (index,), ((span.gpu_iova, span.byte_length),)))
    return groups
