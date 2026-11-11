#!/usr/bin/env bash
(
set -eo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/common.sh"
h3_load_environment
if [[ -z "${FA_SO_PATH:-}" ]]; then
  export FA_SO_PATH="$("$PYTHON" -c 'from importlib.util import find_spec; s=find_spec("flash_attn_2_cuda"); print(s.origin if s else "")')"
fi
mkdir -p "$WORK_HOME/logs"
log="$WORK_HOME/logs/fl2va-run-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PYTHON" -u "$script_dir/run_fl2va_vae_math.py" \
    "$WORK_HOME/runtime-gfx936/examples/minimax_h3.py" fl2va "$@" \
  2>&1 | tee "$log"
printf 'Log: %s\n' "$log"
)
