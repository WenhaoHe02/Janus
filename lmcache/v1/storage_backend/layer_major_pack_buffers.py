# SPDX-License-Identifier: Apache-2.0
"""Allocate only the buffers consumed by synchronous layer-major packing."""

# Standard
from collections.abc import Sequence


def allocate_pack_buffers(
    wave_nbytes: Sequence[int], *, asynchronous: bool, gpu_pack: bool
) -> list[bytearray | None]:
    """Return two alternating CPU pack buffers only for the synchronous path.

    Async staging owns a separate payload for each queued write. GPU packing
    owns pinned tensors. Neither path consumes these CPU bounce buffers.

    Args:
        wave_nbytes: Validated physical byte counts in submission order.
        asynchronous: Whether a CPU staging queue owns the write payloads.
        gpu_pack: Whether packing uses its separate pinned GPU-pack buffers.
    Returns:
        Two ``None`` entries for async/GPU paths; otherwise two zero-initialized
        buffers sized to the largest even/odd wave, preserving empty-wave behavior.
    Raises:
        ValueError: A synchronous buffer size is negative.
        MemoryError: A required synchronous buffer cannot be allocated.
    """
    if asynchronous or gpu_pack:
        return [None, None]
    return [bytearray(max(wave_nbytes[index::2], default=0)) for index in range(2)]
