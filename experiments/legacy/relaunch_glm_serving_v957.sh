#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

arm=${1:?q_sync or unsharded or k_local required}
phase=${2:-ttft}
case "$phase" in
  ttft) accuracy=0; timing=0 ;;
  accuracy) accuracy=1; timing=0 ;;
  profile) accuracy=0; timing=1 ;;
  io) accuracy=0; timing=1 ;;
  *) echo 'phase must be ttft, accuracy or profile' >&2; exit 2 ;;
esac
predict=${LMCACHE_GLM_DSA_PREDICTIVE_PREFETCH:-1}
idxdiag=${LMCACHE_CSA_INDEXED_READ_DIAG:-0}
slotdiag=${LMCACHE_CSA_SLOTMAP_DIAG:-0}
ovdiag=${LMCACHE_CSA_APPEND_OVERLAP_DIAG:-0}
owner_partition=0
if [ "$arm" = k_local ]; then owner_partition=1; fi
case "$arm" in
  q_sync) query_cp=1; key_cp=0; exchange=1 ;;
  unsharded) query_cp=0; key_cp=0; exchange=0 ;;
  k_local) query_cp=0; key_cp=1; exchange=0 ;;
  no_prefetch) query_cp=0; key_cp=0; exchange=0; predict=0 ;;
  ssd_only) query_cp=0; key_cp=0; exchange=0; predict=0 ;;
  dram) query_cp=0; key_cp=0; exchange=0; predict=0 ;;
  *) exit 2 ;;
esac
exec 9</tmp/glm_cluster_relaunch_v306.lock
if ! flock -n 9; then
  echo 'A cluster-arm container launch is already in progress.'
  exit 75
fi
if [ "${LMCACHE_EXPERIMENT_FORCE_RELAUNCH:-0}" != 1 ] && python3 - "$query_cp" "$key_cp" "$exchange" "$accuracy" "$timing" "$owner_partition" "$predict" "$idxdiag" "$slotdiag" "$ovdiag" <<'PY'
import json, subprocess, sys
try:
    info = json.loads(subprocess.check_output(['docker','inspect','dsv4-csa-cp8-off'],stderr=subprocess.DEVNULL))[0]
except subprocess.CalledProcessError:
    raise SystemExit(1)
env = dict(item.split('=',1) for item in info['Config']['Env'] if '=' in item)
keys = ('LMCACHE_PROXY_INDEXER_CP','LMCACHE_PROXY_INDEXER_K_CP','LMCACHE_PROXY_INDEXER_CP_EXCHANGE','LMCACHE_INDEXER_PROFILE_ACCURACY','LMCACHE_CSA_ATTENTION_KV_TIMING','LMCACHE_GLM_DSA_OWNER_PARTITION','LMCACHE_GLM_DSA_PREDICTIVE_PREFETCH','LMCACHE_CSA_INDEXED_READ_DIAG','LMCACHE_CSA_SLOTMAP_DIAG','LMCACHE_CSA_APPEND_OVERLAP_DIAG')
matches = info['State']['Running'] and all(env.get(key)==value for key,value in zip(keys,sys.argv[1:]))
matches = matches and any(m.get('Source') == '/tmp/startup_glm_paper_v385.sh'
                          for m in info.get('Mounts', []))
if matches and sys.argv[4] == '1':
    matches = subprocess.run([
        'docker','exec','dsv4-csa-cp8-off','grep','-q',
        'GLM_DSA_EXACT_SETS_FILE_V317',
        '/usr/local/lib/python3.12/dist-packages/lmcache/v1/glm_dsa_predictive_prefetch.py',
    ]).returncode == 0
if matches:
    matches = subprocess.run([
        'docker','exec','dsv4-csa-cp8-off','grep','-q',
        'GLM_PROXY_PRIVATE_WORKSPACE_V326',
        '/usr/local/lib/python3.12/dist-packages/lmcache/integration/vllm/glm_dsa_vllm_0202.py',
    ]).returncode == 0
if matches:
    matches = subprocess.run([
        'docker','exec','dsv4-csa-cp8-off','grep','-q',
        'GLM_COMPACT_MAIN_GUARD_V363',
        '/usr/local/lib/python3.12/dist-packages/lmcache/v1/cache_engine.py',
    ]).returncode == 0
raise SystemExit(0 if matches else 1)
PY
then
  printf 'Requested arm is already running: %s\n' "$arm"
  exit 0
fi
root=/home/zbuser02/lmcache_v026_20260806_run
version=${LMCACHE_GLM_VERSION_DIR:-$root/versions/20260831_glm_fixed_route_v80}
archive=$root/results/glm_paper_relaunch_v385/${arm}_${phase}_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$archive"
if docker inspect dsv4-csa-cp8-on >/dev/null 2>&1; then
  docker inspect dsv4-csa-cp8-on > "$archive/previous_pro_container.json"
  docker stop -t 120 dsv4-csa-cp8-on
  docker rename dsv4-csa-cp8-on "pro-saved-before-glm-$(date -u +%Y%m%dT%H%M%SZ)"
fi
# PREVIOUS CONTAINER GUARD: nothing to archive on a clean host.
if docker inspect dsv4-csa-cp8-off >/dev/null 2>&1; then
  docker inspect dsv4-csa-cp8-off > "$archive/previous_container_inspect.json"
  docker cp dsv4-csa-cp8-off:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer.py "$archive/sparse_attn_indexer.before.py"
  docker stop -t 120 dsv4-csa-cp8-off
  docker rename dsv4-csa-cp8-off "glm-cluster-saved-$(date -u +%Y%m%dT%H%M%SZ)"
else
  printf "no previous container to archive
" > "$archive/no_previous_container.txt"
fi
bash "$root/versions/20260825_dsv4_pro_replicated_sparse_v12/restore_gpu002_nvme_mounts_recovered.sh"

# Save only small Pro tokenizer/config inputs during the normal mounted stage.
# Never mount an snvme-owned device just to prepare the next model's corpus.
pro_tokenizer_source=/mnt/nvme0/models/DeepSeek-V4-Pro-HF64
pro_tokenizer_copy=/home/zbuser02/datasets/agent_tokenizers_v323/pro
if [ -r "$pro_tokenizer_source/tokenizer.json" ]; then
  mkdir -p "$pro_tokenizer_copy"
  for name in tokenizer.json tokenizer_config.json config.json; do
    if [ -r "$pro_tokenizer_source/$name" ] && [ ! -e "$pro_tokenizer_copy/$name" ]; then
      cp "$pro_tokenizer_source/$name" "$pro_tokenizer_copy/$name"
    fi
  done
fi

export LMCACHE_MATRIX_VERSION_DIR="$version"
export LMCACHE_ABLATION_PATCH_DIR="$version/pro_cp_workspace_overlay"
export LMCACHE_ABLATION_STARTUP_SCRIPT="$version/startup_cp8_ab.sh"
export LMCACHE_ABLATION_MODEL_HOST=/local/GLM-5.2-FP8
export LMCACHE_ABLATION_MAX_MODEL_LEN=${LMCACHE_ABLATION_MAX_MODEL_LEN:-524288}
export LMCACHE_ABLATION_GPU_UTIL=0.82
export LMCACHE_ABLATION_MAX_BATCHED_TOKENS=16384
# 16 GiB holds only 318464 tokens and repeatedly preempts the 362K seed.
# Keep every GLM serving arm on the same capacity for the unchanged29 inputs.
export LMCACHE_ABLATION_KV_CACHE_MEMORY_BYTES=${LMCACHE_ABLATION_KV_CACHE_MEMORY_BYTES:-21474836480}
export LMCACHE_PROXY_INDEXER_CP="$query_cp"
export LMCACHE_PROXY_INDEXER_CP_SIZE=8
export LMCACHE_PROXY_INDEXER_CP_MIN_ROWS=8
export LMCACHE_PROXY_INDEXER_CP_EXCHANGE="$exchange"
export LMCACHE_PROXY_INDEXER_K_CP="$key_cp"
export LMCACHE_PROXY_INDEXER_K_CP_SIZE=8
if [ "$arm" = k_local ]; then
  export LMCACHE_CSA_PROXY_QUERY_TILE_ROWS=8192
  export LMCACHE_CSA_PROXY_KEY_BUCKET=1
else
  export LMCACHE_CSA_PROXY_QUERY_TILE_ROWS=0
  export LMCACHE_CSA_PROXY_KEY_BUCKET=0
fi
export LMCACHE_TRUE_INDEXER_CP=0
export LMCACHE_GLM_DSA_FULL_LAYER_LOOKAHEAD=${LMCACHE_GLM_DSA_FULL_LAYER_LOOKAHEAD:-1}
export LMCACHE_GLM_DSA_PREDICTIVE_PREFETCH="$predict"
export LMCACHE_GLM_DSA_NATIVE_INDEXER_STREAM=${LMCACHE_GLM_DSA_NATIVE_INDEXER_STREAM:-0}
export LMCACHE_CSA_INDEXED_READ_DIAG="$idxdiag"
export LMCACHE_CSA_SLOTMAP_DIAG="$slotdiag"
export LMCACHE_CSA_APPEND_OVERLAP_DIAG="$ovdiag"
export LMCACHE_GLM_DSA_OWNER_PARTITION="$owner_partition"
export LMCACHE_GLM_DSA_CPU_K_BOUNDS=1
export LMCACHE_CSA_OWNER_BLOCKS_PER_RANK=512
export LMCACHE_CSA_OWNER_APPEND_RESERVE_BLOCKS=32
export LMCACHE_CSA_OWNER_GPU_METADATA=1
export LMCACHE_CSA_OWNER_KV_PACKET=0
export LMCACHE_GLM_DSA_PREFETCH_SHARED_MISSES=0
export LMCACHE_GLM_DSA_OWNER_READ_WORKERS=1
export LMCACHE_GLM_DSA_PREFIRE_OWNER_GATHER=0
export LMCACHE_GLM_DSA_SHARE_TOPK_SELECTION=1
export LMCACHE_GLM_DSA_ASYNC_SHARED_CORRECTION=0
export LMCACHE_CSA_LAYER_MAJOR_INDEXED_SPARSE=1
export LMCACHE_INDEXER_PROFILE_ACCURACY="$accuracy"
export LMCACHE_INDEXER_TIMING=0
export LMCACHE_CSA_ATTENTION_KV_TIMING=0
export LMCACHE_CSA_PIPELINE_NVTX=0
export LMCACHE_TTFT_STAGE_PROFILE=0
export LMCACHE_D2H_TIMING=0
export LMCACHE_NSYS_CAPTURE=0
export LMCACHE_TUTTI_PROFILE=0
export LMCACHE_GLM_NATIVE_AUDIT=0
export LMCACHE_CSA_DECODE_SSD_PROFILE=0
export LMCACHE_DSV4_CONTINUATION_CORRECTION=0

export LMCACHE_ABLATION_STARTUP_SCRIPT=${LMCACHE_EXPERIMENT_STARTUP_SCRIPT:-/tmp/startup_glm_paper_v385.sh}
if [ "$arm" = no_prefetch ] || [ "$arm" = ssd_only ] || [ "$arm" = dram ]; then
  [ "$phase" = ttft ] || { echo 'baseline profile uses a separate capture launcher' >&2; exit 2; }
  baseline_mode=tutti_no_prefetch
  if [ "$arm" = ssd_only ]; then baseline_mode=ssd_only; fi
  if [ "$arm" = dram ]; then
    export LMCACHE_SSD_LOCAL_CPU_GB=128
    bash /tmp/launch_glm_dram_v879.sh dram 9>&-
  else
    # Parent retains the launch lock; background helpers must not inherit it.
    bash /tmp/launch_glm_baseline_v385.sh "$baseline_mode" 9>&-
  fi
elif [ "$phase" = accuracy ] || [ "$phase" = profile ] || [ "$phase" = ttft ] || [ "$phase" = io ]; then
  if [[ "${LMCACHE_EXPERIMENT_STARTUP_SCRIPT:-}" = /tmp/startup_glm_serving*.sh ]]; then
    # Serving installs the stable implementation before worker import. Its
    # frozen-corpus warmup is separate from this container-health launcher.
    unset LMCACHE_PROXY_WARMUP_PREP_SCRIPT LMCACHE_PROXY_WARMUP_SCRIPT LMCACHE_PROXY_WARMUP_RESULT
    export LMCACHE_PROXY_WARMUP_REQUIRED=0
  elif [ "$arm" = k_local ] && [ "$phase" != accuracy ]; then
    export LMCACHE_PROXY_WARMUP_PREP_SCRIPT=/tmp/prepare_glm_stable_k_v487.py
    export LMCACHE_PROXY_WARMUP_SCRIPT=/tmp/prewarm_glm_proxy_buckets_v438.py
    export LMCACHE_PROXY_WARMUP_RESULT=/tmp/glm_proxy_bucket_warmup_v438.json
    export LMCACHE_PROXY_WARMUP_REQUIRED=1
  else
    unset LMCACHE_PROXY_WARMUP_PREP_SCRIPT
    unset LMCACHE_PROXY_WARMUP_SCRIPT LMCACHE_PROXY_WARMUP_RESULT
    export LMCACHE_PROXY_WARMUP_REQUIRED=0
  fi
  export LMCACHE_RUN_CONTAINER_SCRIPT=/tmp/run_container_glm_cluster_measure_v438.sh
  if [ "$phase" = io ]; then
    export LMCACHE_CSA_ATTENTION_KV_TIMING=1
    export LMCACHE_TUTTI_PROFILE=1
    unset LMCACHE_EXEC_PREFIX
  fi
  if [ "$phase" = profile ]; then
    nsys_session=${LMCACHE_NSYS_SESSION_NAME:-glm_cluster_${arm}_v307}
    export LMCACHE_CSA_ATTENTION_KV_TIMING=1
    export LMCACHE_CSA_PIPELINE_NVTX=1
    export LMCACHE_TUTTI_PROFILE=1
    export LMCACHE_EXEC_PREFIX="/opt/nvidia/nsight-systems/2025.3.2/target-linux-x64/nsys launch --session-new ${nsys_session} --trace=cuda,nvtx,osrt --cuda-event-trace=false --cuda-flush-interval=5000"
  fi
  bash /tmp/launch_glm_cluster_measure_v438.sh 9>&-
else
  bash "$version/launch_glm52_owner_io_v60.sh"
fi
docker inspect dsv4-csa-cp8-off > "$archive/new_container_inspect.json"
printf 'ARM_CONTAINER_LAUNCHED arm=%s archive=%s\n' "$arm" "$archive"
