#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
# Base on measured v376, not the mixed-result v379 lifetime experiment.
export LMCACHE_EXPERIMENT_COUNTED_LIFETIME_V379=0
export LMCACHE_DSV4_FUSED_PROXY_BLOCK_UNION=1
exec bash /tmp/startup_ds_cyclic_submit_v376.sh
