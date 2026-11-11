#!/usr/bin/env bash
set -euo pipefail
: "${WORK_HOME:?Enter the Pod with WORK_HOME set to the persistent login home}"
source "$WORK_HOME/setup_env.sh"
OPS_DIR="$WORK_HOME/Projects/h3_ops"
mkdir -p "$OPS_DIR/results"
trap 'echo FAILED > "$OPS_DIR/results/silu.status"' ERR
echo RUNNING > "$OPS_DIR/results/silu.status"
python "$OPS_DIR/benchmark_silu.py" --backend native --mode validate --output "$OPS_DIR/results/silu-native-correctness.json"
bash "$WORK_HOME/run_ninetoothed.sh" "$OPS_DIR/benchmark_silu.py" --backend nine-plan --mode validate \
  --native-fixture "$OPS_DIR/results/silu-native-correctness.pt" --output "$OPS_DIR/results/silu-nine-correctness.json"
python "$OPS_DIR/benchmark_silu.py" --backend native --mode bench --output "$OPS_DIR/results/silu-native-a.json"
bash "$WORK_HOME/run_ninetoothed.sh" "$OPS_DIR/benchmark_silu.py" --backend nine --mode bench --output "$OPS_DIR/results/silu-nine-jit.json"
bash "$WORK_HOME/run_ninetoothed.sh" "$OPS_DIR/benchmark_silu.py" --backend nine-plan --mode bench --output "$OPS_DIR/results/silu-nine-plan.json"
python "$OPS_DIR/benchmark_silu.py" --backend native --mode bench --output "$OPS_DIR/results/silu-native-b.json"
echo COMPLETE > "$OPS_DIR/results/silu.status"
