#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
export LMCACHE_EXPERIMENT_CYCLIC_BOUND_V376=1
exec bash /tmp/startup_ds_gpu_submit_v375.sh
