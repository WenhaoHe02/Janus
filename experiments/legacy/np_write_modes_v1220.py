# SPDX-License-Identifier: Apache-2.0
"""Real NP write-stage replacements; never add dummy GPU work for an ablation."""

# Standard
import ctypes
from collections.abc import Iterator
from contextlib import contextmanager
import logging
import os
import time
from typing import Any

LOG = logging.getLogger(__name__)


@contextmanager
def mapped_source(torch: Any, source: memoryview, offset: int,
                  count: int, stream: Any) -> Iterator[int]:
    """Temporarily map pageable payload pages, retaining already pinned ownership.

    No payload copy is made. The caller must retain the source until this scope
    exits. Synchronize this copy stream before unregistering pages owned here.
    CUDA registration failures raise instead of falling back to CPU packing.
    """
    address = ctypes.addressof(ctypes.c_char.from_buffer(source)) + offset
    tensor = torch.frombuffer(source, dtype=torch.uint8, count=count, offset=offset)
    owned = not tensor.is_pinned()
    base = address & ~4095
    size = ((address + count + 4095) & ~4095) - base
    runtime = torch.cuda.cudart()
    if owned:
        status = runtime.cudaHostRegister(base, size, 2)
        if int(status) != 0:
            raise RuntimeError(f'Cannot map original NP payload: CUDA {status}')
    try:
        yield address
    finally:
        if owned:
            stream.synchronize()
            status = runtime.cudaHostUnregister(base)
            if int(status) != 0:
                raise RuntimeError(f'Cannot release NP payload mapping: CUDA {status}')


def validate_block(source: memoryview, offset: int, count: int, total: int,
                   disk_offset: int, capacity: int) -> None:
    """Reject invalid byte/sector geometry before any GPU action or disk write."""
    if (offset < 0 or count < 0 or total <= 0 or count > total or total > capacity
            or offset + count > len(source) or total % 512 or disk_offset < 0
            or disk_offset % 512 or not source.contiguous):
        raise ValueError('Invalid original NP write-block geometry')


class MappedGpuBlockCopy:
    """Replace existing CPU copy with GPU copy into a reusable pinned write buffer.

    This performs alignment staging, not a new layer-major layout conversion.
    CPU pwrite remains the actual SSD writer. The original admission scope,
    block size, file offset and payload lifetime are retained by the caller.
    """

    def __init__(self, capacity: int, cuda_device: int) -> None:
        """Allocate one persistent pinned buffer on the specified rank's GPU context."""
        import torch
        from lmcache import c_ops

        self.torch = torch
        self.ops = c_ops
        self.capacity = capacity
        self.device = cuda_device
        self.verified = False
        self.calls = 0
        self.bytes = 0
        with torch.cuda.device(cuda_device):
            self.storage = torch.empty(capacity + 4095, dtype=torch.uint8, device='cpu', pin_memory=True)
            offset = (-self.storage.data_ptr()) % 4096
            self.tensor = self.storage[offset:offset + capacity]
            self.stream = torch.cuda.Stream(device=cuda_device)
        self.view = memoryview(self.tensor.numpy()).cast('B')
        if self.tensor.data_ptr() % 4096:
            raise RuntimeError('Pinned CPU write buffer is not page aligned')

    def execute(self, fd: int, original_buffer: Any, source: memoryview,
                source_offset: int, source_nbytes: int, total_nbytes: int,
                disk_offset: int) -> int:
        """GPU-copy the original bytes, CPU-zero padding, then issue the original write.

        Raises before pwrite on unmapped sources, invalid geometry or mismatched
        first-block bytes. No CPU-copy fallback is permitted. Returns actual
        pwrite bytes, which the parent writer checks for a short write.
        """
        validate_block(source, source_offset, source_nbytes, total_nbytes,
                       disk_offset, self.capacity)
        if source.readonly:
            raise ValueError('NP payload must retain writable pinned snapshot storage')
        torch = self.torch
        started = time.time_ns()
        destination = self.tensor[:total_nbytes]
        # This is the same required alignment padding as the original CPU copy.
        if source_nbytes < total_nbytes:
            destination[source_nbytes:].zero_()
        if source_nbytes:
            offsets = list(range(0, source_nbytes, 65536))
            with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
                with mapped_source(torch, source, source_offset, source_nbytes, self.stream) as address:
                    self.ops.pack_pinned_host_segments(
                        [address], [0] * len(offsets), offsets, offsets,
                        [min(65536, source_nbytes - n) for n in offsets],
                        destination, torch.device('cuda', self.device))
            self.stream.synchronize()
        finished = time.time_ns()
        target = self.view[:total_nbytes]
        try:
            if not self.verified:
                expected = bytes(source[source_offset:source_offset + source_nbytes])
                actual = bytes(target)
                if actual != expected + bytes(total_nbytes - source_nbytes):
                    raise RuntimeError('GPU alignment-pack bytes differ from CPU contract')
                if source_nbytes:
                    self.verified = True
                    LOG.info('NP_GPU_PACK_V1220_VERIFIED rank=%d bytes=%d', self.device, total_nbytes)
            written = os.pwrite(fd, target, disk_offset)
        finally:
            target.release()
        if written == total_nbytes:
            self.calls += 1
            self.bytes += written
            LOG.info('NP_GPU_PACK_V1220_DONE rank=%d bytes=%d calls=%d total_bytes=%d '
                     'pack_start_ns=%d pack_end_ns=%d', self.device, written,
                     self.calls, self.bytes, started, finished)
        return written

    def close(self) -> None:
        """Release buffer views only after all caller-serialized writes have finished."""
        self.stream.synchronize()
        self.view.release()
        self.tensor = None
        self.storage = None


class GpuNvmeBlockWrite:
    """Write the original NP payload with Tutti's existing GPU WRITE/poll ops.

    The loader provides its registered staging, serialized I/O stream and queue
    pointers. This never binds a controller or invents destination extents.
    The caller retains the existing per-block write admission and byte limits.
    """

    def __init__(self, *, cuda_device: int, capacity: int, staging: Any,
                 stream: Any, io_lock: Any, page_iovas: list[int],
                 queue: dict[str, int]) -> None:
        """Use only the already registered queue/staging owned by this loader."""
        import torch
        from lmcache import c_ops

        if not hasattr(c_ops, 'tutti_submit_batch_sgl_write'):
            raise RuntimeError('This deployed extension lacks real NVMe WRITE')
        self.torch, self.ops = torch, c_ops
        self.device, self.capacity = cuda_device, capacity
        self.staging, self.stream, self.io_lock = staging, stream, io_lock
        self.queue = dict(queue)
        self.pages = min(64, len(page_iovas), staging.numel() // 65536, queue['depth'] - 1)
        if self.pages <= 0 or stream is None:
            raise ValueError('No usable registered write staging/stream')
        with torch.cuda.device(cuda_device):
            self.iovas = torch.tensor(page_iovas[:self.pages], dtype=torch.int64,
                                      device=f'cuda:{cuda_device}')
            self.status = torch.zeros(self.pages, dtype=torch.int32, device=f'cuda:{cuda_device}')
        self.verified = False
        self.calls = 0
        self.bytes = 0

    def submit(self, lba: int, size: int, *, write: bool) -> None:
        """Submit/poll exactly this sector-aligned slice and fail on any CQ error."""
        torch, q = self.torch, self.queue
        lengths = [min(65536, size - start) for start in range(0, size, 65536)]
        count = len(lengths)
        if count <= 0 or count > self.pages or size % 512 or lba < 0:
            raise ValueError('Invalid GPU NVMe batch')
        slbas = torch.tensor([lba + i * 128 for i in range(count)], dtype=torch.int64,
                             device=f'cuda:{self.device}')
        lens = torch.tensor(lengths, dtype=torch.int32, device=f'cuda:{self.device}')
        common = (q['sq'], q['cq'], q['sq_db'], q['cq_db'])
        operation = (self.ops.tutti_submit_batch_sgl_write if write
                     else self.ops.tutti_submit_batch_sgl_read)
        operation(*common, q['tail'], q['depth'], q['qid'], q['nsid'],
                  self.iovas[:count], slbas, lens, self.stream.cuda_stream)
        self.ops.tutti_poll_batch(*common, q['head'], q['phase'], q['depth'], count,
                                 self.status[:count], q['timeout'], 100_000_000,
                                 self.stream.cuda_stream)
        self.stream.synchronize()
        if (ctypes.c_int32.from_address(q['timeout']).value
                or any((int(value) >> 1) & 0x7FFF for value in self.status[:count].cpu().tolist())):
            raise RuntimeError('NVMe GPU WRITE/readback timed out or CQ failed')

    def execute(self, fd: int, original_buffer: Any, source: memoryview,
                source_offset: int, source_nbytes: int, total_nbytes: int,
                disk_offset: int) -> int:
        """Stage actual NP bytes in GPU memory and persist only their given LBAs.

        The first real batch is read back during excluded warmup. No CPU SSD
        write fallback is allowed. The original CPU snapshot is needed because
        the model's source HBM pages may already have been reused after offload.
        """
        validate_block(source, source_offset, source_nbytes, total_nbytes,
                       disk_offset, self.capacity)
        torch = self.torch
        started = time.time_ns()
        with self.io_lock, torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            self.stream.synchronize()
            for cursor in range(0, total_nbytes, self.pages * 65536):
                size = min(self.pages * 65536, total_nbytes - cursor)
                data_bytes = min(size, max(0, source_nbytes - cursor))
                target = self.staging[:size]
                if data_bytes:
                    src = torch.frombuffer(source, dtype=torch.uint8, count=data_bytes,
                                           offset=source_offset + cursor)
                    target[:data_bytes].copy_(src, non_blocking=True)
                if data_bytes < size:
                    target[data_bytes:].zero_()
                lba = (disk_offset + cursor) // 512
                self.submit(lba, size, write=True)
                if not self.verified:
                    expected = bytes(source[source_offset + cursor:source_offset + cursor + data_bytes])
                    expected += bytes(size - data_bytes)
                    target.fill_(expected[0] ^ 255)
                    self.submit(lba, size, write=False)
                    if target.cpu().numpy().tobytes() != expected:
                        raise RuntimeError('Actual NVMe GPU-written bytes failed readback')
                    self.verified = True
                    LOG.info('NP_GPU_WRITE_V1220_VERIFIED rank=%d bytes=%d lba=%d',
                             self.device, size, lba)
        self.calls += 1
        self.bytes += total_nbytes
        LOG.info('NP_GPU_WRITE_V1220_DONE rank=%d bytes=%d calls=%d total_bytes=%d '
                 'write_start_ns=%d write_end_ns=%d', self.device, total_nbytes,
                 self.calls, self.bytes, started, time.time_ns())
        return total_nbytes

    def close(self) -> None:
        """Drain this helper; the loader, not this helper, owns the NVMe session."""
        self.stream.synchronize()
