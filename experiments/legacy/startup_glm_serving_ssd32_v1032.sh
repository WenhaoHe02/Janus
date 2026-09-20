#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
# 386K restores ~19.4 GB/rank; the legacy 16 GiB staging pool cannot hold it.
# Keep local_cpu=false and O_DIRECT=true: this is not a CPU cache tier.
export LMCACHE_SSD_LOCAL_CPU_GB=32
exec bash /tmp/startup_glm_serving_baseline_v1031.sh
