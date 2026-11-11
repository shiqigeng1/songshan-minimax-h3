#!/usr/bin/env bash
# Python environment preparation: runtime, compare, or nine.
(
set -eo pipefail
umask 0002
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/common.sh"
mode="${1:-runtime}"
case "$mode" in runtime|compare|nine|flash) ;; *) echo "Usage: bash $0 [runtime|compare|nine|flash]"; exit 2;; esac
mkdir -p "$WORK_HOME"/{venvs,logs,pip-cache}
export PIP_CACHE_DIR="$WORK_HOME/pip-cache"
export DTK_ROOT="${DTK_ROOT:-/opt/dtk}"
if [[ -r "$DTK_ROOT/env.sh" ]]; then source "$DTK_ROOT/env.sh"; fi
base_python="${H3_BASE_PYTHON:-python3}"
"$base_python" -c 'import sys,torch; assert sys.version_info[:2]==(3,11); assert torch.version.hip, "The platform Hygon Torch is required"; print("Base:",sys.executable,torch.__version__,torch.version.hip)'
venv_name="$mode"
[[ "$mode" == runtime || "$mode" == flash ]] && venv_name=native
venv="$WORK_HOME/venvs/$venv_name"
if [[ ! -x "$venv/bin/python" ]]; then
  "$base_python" -m venv --without-pip --system-site-packages "$venv"
fi
python_executable="$venv/bin/python"
"$python_executable" -c 'import sys,torch; assert sys.prefix != sys.base_prefix; assert torch.version.hip'
pip_source=()
if [[ -n "${H3_WHEEL_DIR:-}" ]]; then
  test -d "$H3_WHEEL_DIR"
  pip_source=(--no-index --find-links "$H3_WHEEL_DIR")
elif [[ "${H3_OFFLINE:-0}" == 1 ]]; then
  echo 'Offline preparation requires H3_WHEEL_DIR with individual wheel files.' >&2
  exit 1
fi
log="$WORK_HOME/logs/prepare-${mode}-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
case "$mode" in
  runtime)
    # Existing platform Torch satisfies the fixed constraint; do not replace it.
    h3_pip "$python_executable" install --only-binary=:all: --no-compile \
      "${pip_source[@]}" -c "$H3_REPO_ROOT/requirements/runtime-constraints.txt" \
      -r "$H3_REPO_ROOT/requirements/runtime.txt" 2>&1 | tee "$log"
    h3_pip "$python_executable" install --only-binary=:all: --no-compile --no-deps \
      "${pip_source[@]}" 'accelerate==1.13.0' 2>&1 | tee -a "$log"
    if [[ -n "${H3_WHEEL_DIR:-}" ]]; then
      h3_pip "$python_executable" install --no-compile --no-deps "${pip_source[@]}" 'torchvision==0.24.0+cpu' 2>&1 | tee -a "$log"
    else
      h3_pip "$python_executable" install --no-compile --no-deps \
        --index-url https://download.pytorch.org/whl/cpu 'torchvision==0.24.0+cpu' 2>&1 | tee -a "$log"
    fi
    "$python_executable" "$script_dir/write_environment.py" "$WORK_HOME"
    ;;
  compare)
    h3_pip "$python_executable" install --only-binary=:all: --no-compile \
      "${pip_source[@]}" -r "$H3_REPO_ROOT/requirements/compare.txt" 2>&1 | tee "$log"
    ;;
  nine)
    if [[ -z "${H3_TRITON_WHEEL:-}" || ! -r "$H3_TRITON_WHEEL" ]]; then
      echo 'Set H3_TRITON_WHEEL to the platform-matched triton-3.5.1+das.opt1.dtk2604.torch290 wheel.' >&2
      exit 1
    fi
    h3_pip "$python_executable" install --no-compile --no-deps "$H3_TRITON_WHEEL" 2>&1 | tee "$log"
    h3_pip "$python_executable" install --only-binary=:all: --no-compile \
      "${pip_source[@]}" -r "$H3_REPO_ROOT/requirements/nine.txt" 2>&1 | tee -a "$log"
    ;;
  flash)
    name='flash_attn-2.6.1+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl'
    wheel="${H3_FLASH_WHEEL:-${H3_WHEEL_DIR:-}/$name}"
    if [[ ! -r "$wheel" ]]; then
      [[ "${H3_OFFLINE:-0}" != 1 ]] || { echo 'Offline: set H3_FLASH_WHEEL to the platform wheel.'; exit 1; }
      wheel="https://pypi.sourcefind.cn/das_release/dtk2604-rc1/+f/854/6e46bcbfcd346/$name"
    fi
    h3_pip "$python_executable" install --no-compile --no-deps "$wheel" 2>&1 | tee "$log"
    ;;
esac
printf 'Environment: %s\nLog: %s\n' "$venv" "$log"
)
