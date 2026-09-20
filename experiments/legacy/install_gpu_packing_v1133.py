# SPDX-License-Identifier: Apache-2.0
"""Enable strict GPU packing within the unchanged bounded CPU write queue."""
# Standard
import argparse
import hashlib
import json
from pathlib import Path


def transform(source: str) -> str:
    """Replace only the async-stage packing operation, keeping CPU default intact.

    GPU packing operates on mapped pinned host buffers using the existing CUDA
    packer. This is not a claim of pre-DMA HBM relayout. Missing pinned sources
    or packer support raises instead of silently falling back to CPU packing.
    """
    before = '''                            staged_payload = bytearray(wave_nbytes)
                            writer_record, payload, pack_cpu_time_s = (
                                _pack_layer_major_wave(wave, staged_payload)
                            )'''
    after = '''                            if _env_flag("LMCACHE_PACKING_ABLATION_GPU_V1133"):
                                packer = self.kv_object_gpu_layer_major_packer
                                if packer is None:
                                    raise RuntimeError("GPU packing ablation has no CUDA packer")
                                packing_buffer = torch.empty(
                                    wave_nbytes, dtype=torch.uint8, device="cpu", pin_memory=True
                                )
                                writer_record, payload, pack_cpu_time_s = (
                                    _pack_layer_major_wave_on_gpu(wave, packing_buffer, packer)
                                )
                                # A separate buffer view owns the pinned tensor until
                                # the existing background queue has persisted it.
                                staged_payload = memoryview(packing_buffer.numpy())
                                logger.info("PACKING_ABLATION_GPU_V1133 bytes=%d", wave_nbytes)
                            else:
                                staged_payload = bytearray(wave_nbytes)
                                writer_record, payload, pack_cpu_time_s = (
                                    _pack_layer_major_wave(wave, staged_payload)
                                )'''
    if source.count(before) != 1 or 'PACKING_ABLATION_GPU_V1133' in source:
        raise ValueError('Backend stage changed or patch already present')
    result = source.replace(before, after)
    compile(result, 'gpu_packing_backend', 'exec')
    return result


def main() -> None:
    """Install before workers import the backend, preserving the exact original."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lmcache-root', type=Path, required=True)
    args = parser.parse_args()
    path = args.lmcache_root / 'v1/storage_backend/local_disk_backend.py'
    before = path.read_text()
    after = transform(before)
    backup = path.with_suffix('.py.before_gpu_packing_v1133')
    if backup.exists():
        raise FileExistsError(backup)
    backup.write_bytes(path.read_bytes())
    path.write_text(after)
    print(json.dumps(dict(patch='GPU_PACKING_V1133',
                         before_sha256=hashlib.sha256(before.encode()).hexdigest(),
                         after_sha256=hashlib.sha256(after.encode()).hexdigest(),
                         algorithm='mapped-pinned-memory GPU packing; same CPU raw writer/queue')))


if __name__ == '__main__':
    main()
