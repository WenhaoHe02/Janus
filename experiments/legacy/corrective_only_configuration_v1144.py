# SPDX-License-Identifier: Apache-2.0
"""Explicit three-model corrective-only serving controls, never full KV restore."""
# Standard
from collections.abc import Mapping


def corrective_only_flags(model: str, baseline: Mapping[str, str]) -> dict[str, str]:
    """Return NP controls with only real-indexer sparse correction enabled.

    Args:
        model: glm, flash or pro; uses that model's existing NP configuration.
        baseline: Flags returned by the existing no_prefetch environment helper.
    Returns:
        Independent flags. GLM enables its layer-major consumer pipeline with
        prediction off. DS retains its existing authoritative sparse demand path.
    Raises:
        ValueError: Unsupported model or a non-NP baseline.
    """
    if model not in ('glm','flash','pro'):
        raise ValueError('Unsupported model')
    if baseline.get('LMCACHE_SERVING_ARM') != 'no_prefetch':
        raise ValueError('Corrective-only requires the existing NP arm')
    flags=dict(baseline)
    if model=='glm':
        flags['LMCACHE_GLM_DSA_LAYER_MAJOR']='1'
    flags.update(LMCACHE_GLM_DSA_PREDICTIVE_PREFETCH='0',
        LMCACHE_PROXY_INDEXER_CP='0',LMCACHE_PROXY_INDEXER_K_CP='0',
        LMCACHE_TRUE_INDEXER_CP='0',LMCACHE_INDEXER_ENABLE_PREFETCH='0',
        LMCACHE_INDEXER_ENABLE_REUSE_PREFETCH='0',
        LMCACHE_TUTTI_UNPACKED_SUBMIT_V1139='0',
        LMCACHE_PACKING_ABLATION_GPU_V1133='0',
        LMCACHE_WRITE_CAP_EXPERIMENT_V1143='0',
        LMCACHE_CORRECTIVE_ONLY_V1144='1',LMCACHE_CORRECTIVE_MODEL_V1144=model)
    return flags
