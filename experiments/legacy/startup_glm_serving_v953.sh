#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Stable GLM serving entrypoint. The outer launcher selects the arm by setting
# LMCACHE_GLM_DSA_PREDICTIVE_PREFETCH or LMCACHE_STORAGE_MODE; this wrapper
# only fixes the paper serving measurement contract.
export LMCACHE_SERVING_EXPERIMENT=1
# Keep every import spelling on the same backend so counted-read extensions
# registered during startup remain visible to the actual Tutti loader.
export LMCACHE_SERVING_FIX_BACKEND_ALIAS=1
# The legacy Docker launcher forces a 30-second worker handoff timeout even
# though synchronous host release can take longer. This only affects startup;
# the physical-readiness gate still rejects failed loaders before measurement.
export LMCACHE_TUTTI_HANDOFF_TIMEOUT_SEC=${LMCACHE_SERVING_HANDOFF_TIMEOUT_SEC:-180}
export LMCACHE_GLM_IO_VOLUME_PROFILE=0
export LMCACHE_GLM_VOLUME_V750=0
export LMCACHE_TUTTI_COUNTED_VOLUME_PROFILE=0
export LMCACHE_TUTTI_DEVICE_TIMESTAMPS=0
export LMCACHE_INDEXER_PROFILE_ACCURACY=0
export LMCACHE_INDEXER_TIMING=0
export LMCACHE_CSA_ATTENTION_KV_TIMING=0
export LMCACHE_TTFT_STAGE_PROFILE=0
export LMCACHE_EXPERIMENT_NVTX=0
unset LMCACHE_EXEC_PREFIX NVTX_INJECTION64_PATH CUDA_INJECTION64_PATH

case "${LMCACHE_STORAGE_MODE:-tutti}" in
  cpu)
    export LMCACHE_SERVING_ARM=dram
    # This is the DRAM cache tier, not SSD scratch or the write-staging window.
    export LMCACHE_CPU_PER_RANK_GB=${LMCACHE_CPU_PER_RANK_GB:-128}
    export LMCACHE_CPU_FIRST_RANK_GB=${LMCACHE_CPU_FIRST_RANK_GB:-$LMCACHE_CPU_PER_RANK_GB}
    ;;
  ssd) export LMCACHE_SERVING_ARM=ssd_only ;;
  tutti)
    if [ "${LMCACHE_GLM_DSA_PREDICTIVE_PREFETCH:-0}" = 1 ]; then
      export LMCACHE_SERVING_ARM=k_local
    else
      export LMCACHE_SERVING_ARM=no_prefetch
    fi
    ;;
  *) echo 'Unknown GLM storage mode' >&2; exit 2 ;;
esac
export LMCACHE_EXEC_PREFIX="python3 /tmp/exec_glm_lookup_v894.py"
exec bash /tmp/startup_glm_paper_v385.sh
