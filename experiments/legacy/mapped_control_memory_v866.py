# SPDX-License-Identifier: Apache-2.0
"""Non-migrating, CPU/GPU-visible storage for small NVMe control words."""

# Standard
import ctypes
from typing import Any


def allocate_mapped_control(size: int, runtime: Any) -> int:
    """Allocate mapped pinned bytes with an identical host/device UVA address.

    Args:
        size: Positive allocation size in bytes.
        runtime: CUDA runtime library exposing host allocation and mapping APIs.

    Returns:
        A pointer usable by both CPU ctypes and existing GPU kernels.

    Raises:
        ValueError: If size is not positive.
        RuntimeError: On allocation/mapping failure or non-identical UVA aliases.

    The caller must quiesce GPU users before calling free_mapped_control.
    No managed-memory migration is involved in CPU reads of these bytes.
    """
    if size <= 0:
        raise ValueError("Control allocation size must be positive")
    host = ctypes.c_void_p()
    # cudaHostAllocPortable | cudaHostAllocMapped; deliberately not write-combined.
    status = runtime.cudaHostAlloc(
        ctypes.byref(host), ctypes.c_size_t(size), ctypes.c_uint(3)
    )
    if status != 0 or host.value is None:
        raise RuntimeError(f"cudaHostAlloc control allocation failed: {status}")
    try:
        device = ctypes.c_void_p()
        status = runtime.cudaHostGetDevicePointer(
            ctypes.byref(device), host, ctypes.c_uint(0)
        )
        if status != 0 or device.value != host.value:
            raise RuntimeError(
                "Control allocation requires identical CPU/GPU UVA aliases "
                f"(mapping status {status})"
            )
    except BaseException:
        runtime.cudaFreeHost(host)
        raise
    return host.value


def free_mapped_control(pointer: int, runtime: Any) -> None:
    """Free a quiesced mapped control allocation; zero is a no-op.

    Args:
        pointer: Host pointer returned by allocate_mapped_control, or zero.
        runtime: CUDA runtime library that allocated the pointer.

    Raises:
        RuntimeError: If CUDA cannot free the allocation.
    """
    if pointer:
        status = runtime.cudaFreeHost(ctypes.c_void_p(pointer))
        if status != 0:
            raise RuntimeError(f"cudaFreeHost control release failed: {status}")
