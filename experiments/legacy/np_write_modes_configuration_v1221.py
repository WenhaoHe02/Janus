# SPDX-License-Identifier: Apache-2.0
"""Two actual NP write-stage replacements, preserving native read batching."""

# Standard
import os

# Local
from np_gpu_write_pack_v1218 import environment as original_environment


def environment(arm: str) -> dict[str, str]:
    """Select real alignment-copy GPU or real NVMe GPU writer, never the unused hook."""
    mode = os.environ.get('LMCACHE_NP_WRITE_MODE_V1220')
    if mode not in ('gpu_pack_cpu_write', 'gpu_nvme_write'):
        raise ValueError('An explicit real NP write mode is required')
    flags = original_environment(arm)
    flags.update(LMCACHE_PACKING_ABLATION_GPU_V1133='0',
                 LMCACHE_NP_WRITE_MODE_V1220=mode,
                 LMCACHE_TUTTI_PAUSE_WRITES_DURING_DECODE='0',
                 LMCACHE_TUTTI_UNLIMITED_WRITES_DURING_DECODE='1',
                 LMCACHE_TUTTI_DECODE_WRITE_MIBPS='0')
    return flags
