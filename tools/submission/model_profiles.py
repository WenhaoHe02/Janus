# SPDX-License-Identifier: Apache-2.0
"""Versionless serving controls derived from the selected measured deployments."""

# Standard
import json
from pathlib import Path


def environment(profile_file: Path, model: str, arm: str) -> dict[str, str]:
    """Select one explicit model/backend profile with normal decode demand reads.

    Args:
        profile_file: Generated source configuration JSON, not experiment results.
        model: flash, pro or glm.
        arm: k_local, no_prefetch, ssd_only or dram.
    Returns:
        New environment dictionary. Raises ValueError for unknown selections or
        query-parallel/unsafe profile content. No modules from old runners load.
    """
    if model not in {"flash", "pro", "glm"} or arm not in {
        "k_local",
        "no_prefetch",
        "ssd_only",
        "dram",
    }:
        raise ValueError("Unknown model or serving path")
    profiles = json.loads(profile_file.read_text())
    family = "glm" if model == "glm" else "dsv4"
    flags = dict(profiles[family][arm])
    if any(
        flags.get(key) != "0"
        for key in ("LMCACHE_PROXY_INDEXER_CP", "LMCACHE_TRUE_INDEXER_CP")
    ):
        raise ValueError("Query/true-indexer sharding is not part of this profile")
    if any("ASSUMED_HBM" in key for key in flags):
        raise ValueError("Decode-read bypass is forbidden")
    if any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in flags.items()
    ):
        raise ValueError("Worker environment must be string-valued")
    tutti = arm in {"k_local", "no_prefetch"}
    enabled = "1" if arm == "k_local" else "0"
    flags.update(
        LMCACHE_SERVING_FIX_BACKEND_ALIAS="1",
        LMCACHE_ENGINE_LOGICAL_BLOCK_SIZE="64" if model == "glm" else "256",
        LMCACHE_DSV4_OPTIMIZED_TAIL_TOKENS="256",
        LMCACHE_KV_OBJECT_STORE_ENABLE="1" if tutti else "0",
        LMCACHE_KV_OBJECT_STORE_TUTTI_RAW_ENABLE="1" if tutti else "0",
        LMCACHE_DSV4_OPTIMIZED_KV="1" if tutti else "0",
        LMCACHE_DSV4_CSA_ATTENTION_KV_FILTER=enabled if model != "glm" else "0",
        LMCACHE_INDEXER_ENABLE_PREFETCH=enabled,
        LMCACHE_INDEXER_ENABLE_REUSE_PREFETCH=enabled,
        LMCACHE_INDEXER_FULL_OVERLAP=enabled,
        LMCACHE_INDEXER_ENABLE_RESIDUAL_PROXY=enabled,
        LMCACHE_INDEXER_TUTTI_BACKEND=enabled,
        LMCACHE_INDEXER_ENABLE_DECODE_PREFETCH="0",
        LMCACHE_INDEXER_ENABLE_PREFILL_RESIDUAL_PROXY="0",
        LMCACHE_INDEXER_ENABLE_PREFILL_EVICTION="0",
        LMCACHE_INDEXER_ENABLE_POOL_SCORING="0",
        LMCACHE_REUSE_PREFETCH_ASYNC="0",
        LMCACHE_DSV4_HCA_WALKER="1" if tutti and model != "glm" else "0",
        LMCACHE_HCA_ENABLE_PREFETCH="0",
        LMCACHE_HCA_ENABLE_DECODE_HOOK="0",
        LMCACHE_TUTTI_COUNTED_INPUT_KEEPALIVE="1",
        LMCACHE_TUTTI_COUNTED_HOLD_GIL="0",
        LMCACHE_INDEXER_IO_WORKERS="8",
        LMCACHE_NATIVE_INDEXER_STAGE0_LAYERS="1",
        LMCACHE_NATIVE_INDEXER_WINDOW_LAYERS="1",
        LMCACHE_SSD_TP_LOCAL_DIRECT_COVERAGE_RATIO="0",
        LMCACHE_CSA_ATTENTION_KV_PROXY_MICROBATCH_ROWS="64",
    )
    if model != "glm":
        flags.update(
            LMCACHE_DS_STATIC_COUNTED_RUNTIME="1" if tutti else "0",
            LMCACHE_DSV4_SHAPE_TEMPLATE_CACHE="1",
            LMCACHE_DSV4_OWNER_SCORER_FUSED="1",
            LMCACHE_DSV4_OWNER_EXACT_PADDING="1",
            LMCACHE_CSA_PIPELINE_SINGLE_LAYER_GATHERS="1",
            LMCACHE_DSV4_OWNER_FUSED_ROUTE="0",
            LMCACHE_DSV4_OWNER_PROXY_GRAPH="0",
            LMCACHE_DSV4_DEVICE_MISS_READ="1",
            LMCACHE_DSV4_DEVICE_MISS_SEGMENTS="1",
            LMCACHE_DSV4_STATIC_INDEXED_PLAN="1",
            LMCACHE_DSV4_DEVICE_HCA_READ="1",
            LMCACHE_DSV4_DEVICE_NATIVE_INDEXER_READ="1",
            LMCACHE_DSV4_DEVICE_DETERMINISTIC_SUBMIT="1",
            LMCACHE_DSV4_ALIGNED_ENTRY_IO="1",
            LMCACHE_DSV4_GPU_OWNER_SUBMIT="1",
            LMCACHE_DSV4_PROXY_FIXED_METADATA="1",
            LMCACHE_DSV4_FUSED_PROXY_BLOCK_UNION="1",
            LMCACHE_CSA_OWNER_GPU_METADATA="1",
            LMCACHE_CSA_OWNER_PINNED_METADATA="0",
        )
    else:
        flags.update(
            LMCACHE_GLM_DSA_DEVICE_OWNER_READ=enabled,
            LMCACHE_GLM_DSA_OWNER_READ_WORKERS="1",
            LMCACHE_CSA_OWNER_KV_PACKET="0",
            LMCACHE_GLM_DSA_FIXED_BOUNDARY_GATHER="0",
        )
    return flags
