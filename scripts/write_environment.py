"""Write a relocatable-prefix shell environment without importing InfiniCore."""
from pathlib import Path
import shlex
import sys

work = Path(sys.argv[1]).resolve()
content = """# Source in each new shell. Override H3_WORK_HOME to select another installation.
if [[ -z "${H3_WORK_HOME:-}" ]]; then
  export H3_WORK_HOME=DEFAULT_WORK
fi
export WORK_HOME="$H3_WORK_HOME"
export DTK_ROOT="${DTK_ROOT:-/opt/dtk}"
if [[ -r "$DTK_ROOT/env.sh" ]]; then source "$DTK_ROOT/env.sh"; fi
export SETUPTOOLS_USE_DISTUTILS=stdlib
export INFINI_ROOT="$WORK_HOME/.infini-gfx936"
export PYTHON="$WORK_HOME/venvs/native/bin/python"
export HYGON_ARCH=gfx936
export PATH="$WORK_HOME/venvs/native/bin:$PATH"
export PYTHONPATH="$WORK_HOME/runtime-gfx936/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
h3_torch_lib="$("$PYTHON" -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parent / "lib")')"
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:$DTK_ROOT/lib:$DTK_ROOT/cuda/cuda/lib64:$h3_torch_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -z "${FA_SO_PATH:-}" ]]; then
  h3_fa="$("$PYTHON" -c 'from importlib.util import find_spec; spec=find_spec("flash_attn_2_cuda"); print(spec.origin if spec else "")')"
  if [[ -n "$h3_fa" ]]; then export FA_SO_PATH="$h3_fa"; fi
fi
unset h3_torch_lib h3_fa
""".replace("DEFAULT_WORK", shlex.quote(str(work)))
(work / "env_gfx936.sh").write_text(content, encoding="utf-8")
print(work / "env_gfx936.sh")
