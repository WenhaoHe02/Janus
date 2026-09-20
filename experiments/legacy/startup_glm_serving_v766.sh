#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
export LMCACHE_SERVING_EXPERIMENT=1
# The launcher supplies prediction flags; derive an explicit arm inside the
# container even when the legacy Docker wrapper does not forward new variables.
if [ "${LMCACHE_STORAGE_MODE:-tutti}" = ssd ]; then
  export LMCACHE_SERVING_ARM=ssd_only
elif [ "${LMCACHE_GLM_DSA_PREDICTIVE_PREFETCH:-0}" = 1 ]; then
  export LMCACHE_SERVING_ARM=k_local
else
  export LMCACHE_SERVING_ARM=no_prefetch
fi
export LMCACHE_GLM_VOLUME_V750=0
export LMCACHE_GLM_IO_VOLUME_PROFILE=0
export LMCACHE_TUTTI_COUNTED_VOLUME_PROFILE=0
export LMCACHE_TUTTI_DEVICE_TIMESTAMPS=0
export LMCACHE_CSA_ATTENTION_KV_TIMING=0
export LMCACHE_EXEC_PREFIX="python3 /tmp/exec_glm_serving_v766.py"
exec bash /tmp/startup_glm_paper_v385.sh
