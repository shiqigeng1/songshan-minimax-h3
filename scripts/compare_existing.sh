#!/usr/bin/env bash
(
set -eo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/common.sh"
reference="${1:?Reference MP4 path required}"
candidate="${2:?Candidate MP4 path required}"
python_executable="$WORK_HOME/venvs/compare/bin/python"
test -x "$python_executable" || { echo 'Run: bash scripts/prepare_env.sh compare'; exit 1; }
entry="$H3_REPO_ROOT/upstream/InfiniLM/examples/minimax_h3_compare.py"
result="$WORK_HOME/checks/fl2va-compare-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$result"
for file in "$reference" "$candidate" "$entry"; do test -r "$file"; done
sha256sum "$reference" "$candidate" "$entry" > "$result/inputs.sha256"
"$python_executable" -u "$entry" --tasks fl2va \
  --model-path "${H3_MODEL_PATH:-/public/home/aclsylqidf/shared/MiniMax-H3/diffusers}" \
  --reference-fl2va-video "$reference" --infinilm-fl2va-video "$candidate" \
  --output-dir "$result" 2>&1 | tee "$result/compare.log"
printf 'Comparison: %s\n' "$result"
)
