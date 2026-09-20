# SPDX-License-Identifier: Apache-2.0
"""Final GLM serving flags applied after legacy startup installers."""


def environment(arm: str, *, stage_timing_diagnostic: bool = False) -> dict[str, str]:
    """Return matched serving controls; only the K arm enables prediction.

    Args:
        arm: k_local, no_prefetch, conventional ssd_only, or dram.
        stage_timing_diagnostic: Explicitly enable host stage timestamps for an
            isolated diagnostic, never for a formal serving measurement.
    Returns:
        Flags for normal-startup installation, no query sharding or profiling.
    Raises:
        ValueError: Unsupported comparison arm.
    """
    if arm not in ('k_local', 'no_prefetch', 'ssd_only', 'dram'):
        raise ValueError('unsupported serving comparison arm')
    enabled = '1' if arm == 'k_local' else '0'
    flags = dict(LMCACHE_SERVING_EXPERIMENT='1', 
        LMCACHE_SERVING_ARM=arm, LMCACHE_GLM_DSA_LAYER_MAJOR=enabled,
        LMCACHE_GLM_DSA_PREDICTIVE_PREFETCH=enabled, LMCACHE_GLM_DSA_OWNER_PARTITION=enabled,
        LMCACHE_PROXY_INDEXER_CP='0', LMCACHE_PROXY_INDEXER_K_CP=enabled,
        LMCACHE_PROXY_INDEXER_CP_SIZE='8', LMCACHE_PROXY_INDEXER_K_CP_SIZE='8',
        LMCACHE_PROXY_INDEXER_CP_EXCHANGE='0', LMCACHE_TRUE_INDEXER_CP='0',
        LMCACHE_GLM_DSA_ASYNC_PREDICTION='0', LMCACHE_GLM_DSA_FULL_LAYER_LOOKAHEAD='1',
        LMCACHE_CSA_OWNER_BLOCKS_PER_RANK='512', LMCACHE_CSA_OWNER_APPEND_RESERVE_BLOCKS='32',
        LMCACHE_GLM_DEVICE_SELECTION_MISS_READ=enabled, LMCACHE_CSA_OWNER_GPU_METADATA='1',
        LMCACHE_CSA_OWNER_PINNED_METADATA='0', LMCACHE_DSV4_OWNER_NONBLOCKING_METADATA='1',
        LMCACHE_CSA_OWNER_GPU_COMPACT_WIDTH='1', LMCACHE_GLM_DSA_SHARED_CORRECTION_AT_CONSUMER='1',
        LMCACHE_GLM_DSA_PIPELINE_SHARED_GATHERS='1', LMCACHE_CSA_PROXY_QUERY_TILE_ROWS='8192',
        LMCACHE_CSA_PROXY_KEY_BUCKET='1',
        LMCACHE_DSV4_CPU_RAW_WRITE='1', LMCACHE_TUTTI_CPU_STAGE_ENABLE='1',
        LMCACHE_TUTTI_CPU_STAGE_GIB='2',
        LMCACHE_TUTTI_PAUSE_WRITES_DURING_DECODE='1',
        LMCACHE_TUTTI_UNLIMITED_WRITES_DURING_DECODE='0',
        LMCACHE_TUTTI_DECODE_WRITE_MIBPS='0')
    if arm in ('ssd_only', 'dram'):
        # The SSD baseline uses the existing filesystem/CPU retrieval path,
        # not Tutti's sparse on-demand loader or its raw write-stage queue.
        flags['LMCACHE_GLM_DSA_LAYER_MAJOR'] = '0'
        flags['LMCACHE_GLM_DEVICE_SELECTION_MISS_READ'] = '0'
    for key in ('LMCACHE_GLM_VOLUME_V750', 'LMCACHE_GLM_IO_VOLUME_PROFILE',
        'LMCACHE_TUTTI_COUNTED_VOLUME_PROFILE', 'LMCACHE_TUTTI_DEVICE_TIMESTAMPS',
        'LMCACHE_DS_COMPLETE_IO_V510', 'LMCACHE_INDEXER_PROFILE_ACCURACY',
        'LMCACHE_INDEXER_TIMING', 'LMCACHE_CSA_ATTENTION_KV_TIMING',
        'LMCACHE_CSA_PIPELINE_NVTX', 'LMCACHE_TTFT_STAGE_PROFILE', 'LMCACHE_D2H_TIMING',
        'LMCACHE_NSYS_CAPTURE', 'LMCACHE_TUTTI_PROFILE', 'LMCACHE_GLM_NATIVE_AUDIT',
        'LMCACHE_CSA_DECODE_SSD_PROFILE', 'LMCACHE_GLM_DSA_DECODE_COVERAGE_PROFILE',
        'LMCACHE_GLM_LAYER_NVTX', 'LMCACHE_EXPERIMENT_GLM_NVTX_V387'):
        flags[key] = '0'
    if stage_timing_diagnostic:
        flags['LMCACHE_TTFT_STAGE_PROFILE'] = '1'
    return flags
