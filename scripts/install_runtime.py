"""Persist built modules using data-only copies compatible with shared storage."""
from pathlib import Path
import json
import shutil
import sys
import sysconfig

source, work = map(lambda p: Path(p).resolve(), sys.argv[1:3])
package = sys.argv[3]
module = source / "python" / package
suffix = sysconfig.get_config_var("EXT_SUFFIX")
extension = list(module.rglob(f"_{package}{suffix}"))
if not any(p.is_file() and p.stat().st_size for p in extension):
    raise SystemExit(f"Missing compiled extension in {module}")
runtime = work / "runtime-gfx936"


def copy_contents(src, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
        if entry.name == "__pycache__" or entry.suffix == ".pyc":
            continue
        target = dst / entry.name
        if entry.is_dir():
            copy_contents(entry, target)
        else:
            shutil.copyfile(entry, target)


copy_contents(module, runtime / "python" / package)
if package == "infinilm":
    copy_contents(source / "examples", runtime / "examples")
site = Path(sysconfig.get_path("purelib")).resolve()
if not site.is_relative_to(work / "venvs/native"):
    raise SystemExit(f"Unexpected install location: {site}")
(site / "songshan_h3_runtime.pth").write_text(str(runtime / "python") + "\n")
(runtime / f"{package}-build.json").write_text(json.dumps({
    "source_build_path": str(source), "python": sys.executable,
    "compiled_extension": [str(p.relative_to(module)) for p in extension],
    "device_execution_verified_by_this_script": False,
}, indent=2) + "\n")
print("Runtime installed:", runtime / "python" / package)
