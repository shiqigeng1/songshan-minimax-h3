#!/usr/bin/env bash
(
set -eo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/common.sh"
h3_load_environment
python_executable="$WORK_HOME/venvs/nine/bin/python"
test -x "$python_executable" || { echo 'Run: bash scripts/prepare_env.sh nine'; exit 1; }
scratch="$(mktemp -d "${IFAB_WORKSPACE_ROOT:-/tmp}/h3-nine.XXXXXX")"
export PYTHONPATH="$script_dir:$WORK_HOME/runtime-gfx936/python${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_MANAGER=h3_triton_cache:SiblingFileCacheManager
export TRITON_CACHE_DIR="$scratch/triton-cache"
export TMPDIR="$scratch/tmp"
mkdir -p "$TMPDIR"
log="$WORK_HOME/logs/gated-residual-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
"$python_executable" -u "$H3_REPO_ROOT/h3_ops/gated_residual_nt.py" 2>&1 | tee "$log"
)
