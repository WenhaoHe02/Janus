#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Unified startup; build, warmup and readiness remain in the Python supervisor.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

if [[ $# -eq 0 ]]; then
    set -- --help
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/tools/submission/start.py" "$@"
