#!/usr/bin/env bash
# Compile expanded sources in local scratch; persist installations and logs.
(
set -eo pipefail
umask 0002
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/common.sh"
mode="${1:-core}"
case "$mode" in core|lm|deps) ;; *) echo "Usage: bash $0 [deps|core|lm]"; exit 2;; esac
h3_load_environment
mkdir -p "$WORK_HOME/logs"
log="$WORK_HOME/logs/build-${mode}-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
if [[ -z "${H3_BUILD_LOG:-}" ]]; then
  H3_BUILD_LOG="$log" bash "${BASH_SOURCE[0]}" "$mode" 2>&1 | tee "$log"
  exit 0
fi
exec 9>"$WORK_HOME/logs/native-build.lock"
flock -n 9 || { echo 'Another build is running.'; exit 1; }
scratch="$(mktemp -d "${H3_SCRATCH_ROOT:-${IFAB_WORKSPACE_ROOT:-/tmp}}/h3-build.XXXXXX")"
export TMPDIR="$scratch/tmp"
mkdir -p "$TMPDIR" "$scratch/tools"
cp -- "$H3_REPO_ROOT/tools/xmake" "$scratch/tools/xmake"
chmod u+x "$scratch/tools/xmake"
export PATH="$scratch/tools:$(dirname "$PYTHON"):$DTK_ROOT/llvm/bin:$PATH"
export XMAKE_ROOT=y
export XMAKE_GLOBALDIR="$WORK_HOME/xmake-global"
export HYGON_CUDA_HOME="${HYGON_CUDA_HOME:-$DTK_ROOT/cuda/cuda}"
export INFINILM_CXX11_ABI="$("$PYTHON" -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')"
export PIP_REQUIRE_VIRTUALENV=1
export PYTHONPYCACHEPREFIX="$scratch/pycache"
jobs="${BUILD_JOBS:-4}"
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid BUILD_JOBS'; exit 2; }
for tool in gcc g++ cmake make patch; do command -v "$tool"; done
test -x "$HYGON_CUDA_HOME/bin/nvcc"
xmake --version
printf 'Mode: %s\nScratch: %s\nInstall: %s\nABI: %s\n' "$mode" "$scratch" "$WORK_HOME" "$INFINILM_CXX11_ABI"

# The pinned package definitions are plain source directories in this repository.
repo_cache="$XMAKE_GLOBALDIR/.xmake/repositories/xmake-repo"
mkdir -p "$repo_cache"
cp -R "$H3_REPO_ROOT/dependencies/xmake-repo/." "$repo_cache/"
xmake g -y --policies=package.precompiled:n,build.ccache:n \
  --pkg_installdir="$WORK_HOME/xmake-packages" --pkg_cachedir="$scratch/package-cache"
if [[ "${H3_OFFLINE:-0}" == 1 ]]; then xmake g -y --network=private; fi
if [[ -n "${H3_PACKAGE_SEARCHDIR:-}" ]]; then xmake g -y --pkg_searchdirs="$H3_PACKAGE_SEARCHDIR"; fi

if [[ "$mode" == deps ]]; then
  mkdir -p "$scratch/probe"
  cat > "$scratch/probe/xmake.lua" <<'LUA'
add_requires("boost 1.89.0", {configs={stacktrace=true}})
add_requires("pybind11 v3.0.1")
target("dependencies")
  set_kind("phony")
  add_packages("boost", "pybind11")
LUA
  cd "$scratch/probe"
  xmake f -y -m release --ccache=n
  echo "Build dependencies prepared: $WORK_HOME/xmake-packages"
  exit 0
fi

project=InfiniCore
[[ "$mode" == lm ]] && project=InfiniLM
cp -R "$H3_REPO_ROOT/upstream/$project" "$scratch/$project"
cd "$scratch/$project"
if [[ "$mode" == core ]]; then
  patch --batch -p1 < "$H3_REPO_ROOT/patches/gfx936-hygon-mha-aten.patch"
fi
"$PYTHON" - "$PWD/xmake.lua" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]); text=p.read_text()
text=text.replace('add_requires("boost",', 'add_requires("boost 1.89.0",', 1)
text=text.replace('add_requires("pybind11")', 'add_requires("pybind11 v3.0.1")', 1)
p.write_text(text)
PY
export XMAKE_CONFIGDIR="$scratch/xmake-config"
mkdir -p "$XMAKE_CONFIGDIR"
if [[ "$mode" == core ]]; then
  xmake f -y -m release --ccache=n --hygon-dcu=y --hygon-arch=gfx936 \
    --cuda="$HYGON_CUDA_HOME" --ccl=y --aten=y \
    --cxxflags="-D_GLIBCXX_USE_CXX11_ABI=$INFINILM_CXX11_ABI"
  xmake build -y -j "$jobs" infinicore_cpp_api
  for target in infinirt infiniop infiniccl infinicore_cpp_api; do xmake install -y "$target"; done
  xmake build -y -j "$jobs" _infinicore
  xmake install -y _infinicore
  package=infinicore
else
  test -s "$INFINI_ROOT/lib/libinfinicore_cpp_api.so" || { echo 'Build core first.'; exit 1; }
  xmake f -y -m release --ccache=n --cxx11-abi="$INFINILM_CXX11_ABI"
  xmake build -y -j "$jobs" _infinilm
  xmake install -y _infinilm
  package=infinilm
fi
"$PYTHON" "$script_dir/install_runtime.py" "$PWD" "$WORK_HOME" "$package"
printf 'Installed %s. Source/build: %s\nLog: %s\n' "$package" "$PWD" "$H3_BUILD_LOG"
)
