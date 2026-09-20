# SPDX-License-Identifier: Apache-2.0
"""No-prefetch native read batching with GPU packing of mapped CPU write buffers."""

# Local
from serving_ds_configuration_v890 import environment as serving_environment


def environment(arm: str) -> dict[str, str]:
    """Return NP flags, preserving native read batching and decode write pause."""
    if arm not in ('baseline', 'no_prefetch'):
        raise ValueError('Only the NP GPU-write-packing arm is supported')
    flags = serving_environment('no_prefetch')
    flags.update(
        LMCACHE_CORRECTIVE_ONLY_V1144='0',
        LMCACHE_TUTTI_UNPACKED_SUBMIT_V1139='0',
        LMCACHE_PACKING_ABLATION_GPU_V1133='1',
        LMCACHE_WRITE_CAP_EXPERIMENT_V1143='0',
        LMCACHE_READ_RANGE_MODE_V1204='off',
        LMCACHE_READ_RANGE_VERIFY_V1205='0',
    )
    return flags
