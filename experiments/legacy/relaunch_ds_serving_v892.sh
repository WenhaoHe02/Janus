#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
model_kind=${1:?flash or pro required}
prediction_arm=${2:-k_owner}
arm=ours
if [ "$prediction_arm" = no_prefetch ]; then arm=tutti_no_prefetch; fi
if [ "$prediction_arm" = ssd_only ]; then arm=ssd_only; fi
if [ "$prediction_arm" = dram ]; then arm=dram; fi
case "$arm" in
  ours) container_case=on ;;
  tutti_no_prefetch|ssd_only|dram) container_case=off ;;
  *) exit 2 ;;
esac
case "$model_kind" in
  flash) checkpoint=/home/zbuser02/models/DeepSeek-V4-Flash-HF; gpu_util=0.55; kv_bytes=17179869184 ;;
  pro) checkpoint=/mnt/nvme0/models/DeepSeek-V4-Pro-HF64; gpu_util=0.80; kv_bytes="${LMCACHE_ABLATION_KV_CACHE_MEMORY_BYTES:-6442450944}" ;;
  *) exit 2 ;;
esac
root=/home/zbuser02/lmcache_v026_20260806_run
version="$root/versions/20260901_dsv4_proxy_bucketed_v216"
exec 9</tmp/glm_cluster_relaunch_v306.lock
flock -n 9 || { echo 'another launcher is active'; exit 1; }
archive="$root/results/ds_accuracy_launch_v369/${model_kind}_${arm}_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$archive"
for name in dsv4-csa-cp8-off dsv4-csa-cp8-on; do
  if docker inspect "$name" >/dev/null 2>&1; then
    docker inspect "$name" > "$archive/$name.previous.json"
    docker stop -t 120 "$name"
    docker rename "$name" "${name}-saved-v365-$(date -u +%Y%m%dT%H%M%SZ)"
  fi
done
bash "$root/versions/20260825_dsv4_pro_replicated_sparse_v12/restore_gpu002_nvme_mounts_recovered.sh"
test -f "$checkpoint/config.json"
export LMCACHE_MATRIX_VERSION_DIR="$version"
export LMCACHE_ABLATION_PATCH_DIR="$version/pro_cp_workspace_overlay"
export LMCACHE_ABLATION_STARTUP_SCRIPT=${LMCACHE_EXPERIMENT_STARTUP_SCRIPT:-/tmp/startup_ds_serving_v890.sh}
export LMCACHE_ABLATION_MODEL_HOST="$checkpoint"
export LMCACHE_ABLATION_MAX_MODEL_LEN="${LMCACHE_ABLATION_MAX_MODEL_LEN:-278528}"
export LMCACHE_ABLATION_MAX_BATCHED_TOKENS=8192
export LMCACHE_ABLATION_GPU_UTIL="$gpu_util"
export LMCACHE_ABLATION_KV_CACHE_MEMORY_BYTES="$kv_bytes"
export LMCACHE_ABLATION_HCA_WALKER=1
export LMCACHE_ABLATION_TUTTI_N_SLOTS=4
export LMCACHE_ABLATION_TUTTI_SLOT_MB=128
export LMCACHE_SSD_INTERNAL_API_ENABLED=true
export LMCACHE_DOCKER_STOP_TIMEOUT_SEC=120
export LMCACHE_CSA_PROXY_QUERY_TILE_ROWS=8192
export LMCACHE_CSA_PROXY_KEY_BUCKET=1
case "$prediction_arm" in
  k_owner|no_prefetch|ssd_only|dram)
    export LMCACHE_CSA_PREFETCH_CP_MODE=key_sharded_owner
    export LMCACHE_CSA_PREFETCH_CP_SIZE=8
    export LMCACHE_CSA_PREFETCH_CP_INTERLEAVE=16
    export LMCACHE_CSA_PREFETCH_CP_EXCHANGE_IDS=0
    ;;
  q_sync)
    export LMCACHE_CSA_PREFETCH_CP_MODE=query
    export LMCACHE_CSA_PREFETCH_CP_SIZE=8
    export LMCACHE_CSA_PREFETCH_CP_INTERLEAVE=1
    export LMCACHE_CSA_PREFETCH_CP_EXCHANGE_IDS=1
    ;;
  unsharded)
    export LMCACHE_CSA_PREFETCH_CP_MODE=query
    export LMCACHE_CSA_PREFETCH_CP_SIZE=1
    export LMCACHE_CSA_PREFETCH_CP_INTERLEAVE=1
    export LMCACHE_CSA_PREFETCH_CP_EXCHANGE_IDS=0
    ;;
  *) echo "unsupported prediction arm: $prediction_arm" >&2; exit 2 ;;
esac
export LMCACHE_CSA_PREFETCH_LOOKAHEAD_BY_LAYER=default:2
export LMCACHE_CSA_OWNER_BLOCKS_PER_RANK=512
export LMCACHE_PROXY_INDEXER_CP=0
export LMCACHE_PROXY_INDEXER_K_CP=0
export LMCACHE_TRUE_INDEXER_CP=0
export LMCACHE_GLM_DSA_LAYER_MAJOR=0
export LMCACHE_GLM_DSA_PREDICTIVE_PREFETCH=0
export LMCACHE_DSV4_PROXY_MHC_EXACT_ROWS=1
export LMCACHE_CSA_OWNER_P2P_PACKET=0
export LMCACHE_TUTTI_CPU_STAGE_ENABLE=0
export LMCACHE_TUTTI_PROFILE=0
export LMCACHE_INDEXER_PROFILE_ACCURACY=0
export LMCACHE_INDEXER_TIMING=0
export LMCACHE_CSA_ATTENTION_KV_TIMING=0
export LMCACHE_TTFT_STAGE_PROFILE=0
export LMCACHE_CSA_PIPELINE_NVTX=0
export LMCACHE_NSYS_CAPTURE=0
export LMCACHE_NSYS_FULL_CAPTURE=0
export CUDA_LAUNCH_BLOCKING=0
unset LMCACHE_PROXY_WARMUP_PREP_SCRIPT LMCACHE_PROXY_WARMUP_SCRIPT
# Keep the launch lock in this parent only; spawned container helpers must
# not retain descriptor9 after this launcher finishes.
bash /tmp/launch_ds_storage_v892.sh "$arm" 9>&-
if [ "$container_case" = on ]; then
  docker rename dsv4-csa-cp8-on dsv4-csa-cp8-off
  container_case=off
fi
docker inspect "dsv4-csa-cp8-${container_case}" > "$archive/current_container.json"
echo "DS_PAPER_DEPLOYED model=$model_kind arm=$arm archive=$archive"
