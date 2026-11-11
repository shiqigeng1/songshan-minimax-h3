#!/usr/bin/env bash
# Shared path setup; source this file from the launchers.
H3_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export H3_WORK_HOME="${H3_WORK_HOME:-${IFAB_AFS_HOME:-$HOME}/songshan-h3}"
export WORK_HOME="$H3_WORK_HOME"
export SETUPTOOLS_USE_DISTUTILS=stdlib
export PYTHONDONTWRITEBYTECODE=1

h3_load_environment() {
  if [[ ! -r "$H3_WORK_HOME/env_gfx936.sh" ]]; then
    echo "Run first: bash scripts/prepare_env.sh runtime" >&2
    return 1
  fi
  source "$H3_WORK_HOME/env_gfx936.sh"
}

h3_pip() {
  local python_executable="$1"
  shift
  PIP_REQUIRE_VIRTUALENV=1 "$python_executable" -c '
import mimetypes, runpy
mimetypes.knownfiles = []
mimetypes.init(files=[])
import pip._internal.network.session as session
session.user_agent = lambda: "pip/songshan-h3"
runpy.run_module("pip", run_name="__main__")
' --disable-pip-version-check --timeout 30 --retries 2 "$@"
}
