#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
export LMCACHE_CSA_OWNER_GPU_METADATA=1
export LMCACHE_EXPERIMENT_NVTX=0
export LMCACHE_EXPERIMENT_COUNTED_BINDING_V373=0
export LMCACHE_TUTTI_COUNTED_HOLD_GIL=0
export LMCACHE_DSV4_ALIGNED_ENTRY_IO=1
export LMCACHE_DSV4_GPU_OWNER_SUBMIT=1
export LMCACHE_DSV4_PROXY_FIXED_METADATA=1
# key_sharded_owner never exchanged Top-K IDs. This formerly dead flag also
# blocked its existing GPU submission branch; it is not disabling a collective.
export LMCACHE_CSA_PREFETCH_CP_EXCHANGE_IDS=0
exec bash /tmp/startup_ds_correctness_v365.sh
