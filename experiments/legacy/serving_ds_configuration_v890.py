# SPDX-License-Identifier: Apache-2.0
"""DS serving controls, preserving the measured interleave-16 K path."""

# Local
from ds_ttft_configuration_v830 import environment as performance_environment


def environment(arm: str) -> dict[str, str]:
    """Return explicit serving controls for K, NP, disk or DRAM.

    Args:
        arm: k_local, no_prefetch, ssd_only or dram.
    Returns:
        Worker flags; no query sharding, no timing/accuracy instrumentation.
    Raises:
        ValueError: Unsupported arm.
    """
    flags = performance_environment('ssd_only' if arm == 'dram' else arm)
    flags.update(LMCACHE_SERVING_EXPERIMENT='1',
        LMCACHE_SERVING_ARM=arm,
        LMCACHE_CSA_PREFETCH_CP_INTERLEAVE='16',
        LMCACHE_LOOKUP_SERVER_WORKER_IDS='0,1,2,3,4,5,6,7',
        LMCACHE_GLM_IO_VOLUME_PROFILE='0', LMCACHE_GLM_VOLUME_V750='0',
        LMCACHE_TUTTI_MAPPED_CONTROL='1',
        LMCACHE_DSV4_DEVICE_OWNER_READ='1',
        LMCACHE_CSA_OWNER_KV_PACKET='1',
        LMCACHE_INDEXER_ENABLE_REUSE_PREFETCH='1' if arm == 'k_local' else '0',
        LMCACHE_INDEXER_ENABLE_PREFETCH='1' if arm == 'k_local' else '0')
    return flags
