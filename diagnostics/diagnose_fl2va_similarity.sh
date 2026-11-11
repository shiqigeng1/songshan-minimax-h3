#!/usr/bin/env bash
# Inspect existing outputs on CPU without altering the evaluation script/model.
(
set -eo pipefail
export SETUPTOOLS_USE_DISTUTILS=stdlib
source "${IFAB_AFS_HOME:?}/minimax-h3/env_gfx936.sh"
out="$WORK_HOME/checks/fl2va-diagnose-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$out"
{
"$PYTHON" - "$WORK_HOME" "$out" <<'PY'
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
import av
import numpy as np
from PIL import Image, ImageDraw, ImageOps

work, out = map(Path, sys.argv[1:])
videos = [
    ("PyTorch reference", "fl2va-reference-20261111T105437Z"),
    ("InfiniLM candidate", "fl2va-candidate-20261111T085622Z"),
    ("InfiniLM Nov09", "fl2va-songshan-20261109T123355Z"),
]
indexes = (0, 62, 123)
sheet = Image.new("RGB", (3 * 280, 4 * 360), "#eeeeee")
draw = ImageDraw.Draw(sheet)
input_path = work.parent / "songshan-fl2va-first-frame.png"
record = {"sample_indexes": indexes, "videos": {}, "diagnostic_only": True}
if input_path.is_file():
    with Image.open(input_path) as im:
        im = ImageOps.contain(im.convert("RGB"), (270, 325))
        sheet.paste(im, ((280 - im.width) // 2, 28))
    record["current_input_sha256"] = hashlib.sha256(input_path.read_bytes()).hexdigest()
    print("[input]", record["current_input_sha256"], flush=True)
draw.text((8, 8), "Current input PNG", fill="black")
draw.text((288, 60), "Columns: frame 0 / 62 / 123", fill="black")
draw.text((288, 82), "Preview only; evaluation videos unchanged.", fill="black")
samples = {}
for row, (label, run) in enumerate(videos, start=1):
    path = work / "outputs" / f"{run}.mp4"
    if not path.is_file():
        print("[missing]", path, flush=True)
        draw.text((8, row * 360 + 8), f"Missing: {label}", fill="red")
        continue
    selected = {}
    frame_count = 0
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            frame_count += 1
            if index in indexes:
                selected[index] = frame.to_ndarray(format="rgb24")
    record["videos"][run] = {"frames": frame_count, "samples": {}}
    samples[run] = selected
    for column, index in enumerate(indexes):
        draw.text((column * 280 + 8, row * 360 + 8), f"{label} | frame {index}", fill="black")
        if index not in selected:
            continue
        array = selected[index]
        im = ImageOps.contain(Image.fromarray(array), (270, 325))
        sheet.paste(im, (column * 280 + (280 - im.width) // 2, row * 360 + 28))
        stats = {"mean": float(array.mean()), "std": float(array.std())}
        record["videos"][run]["samples"][str(index)] = stats
        print(f"[pixels] {label} frame={index} mean={stats['mean']:.3f} std={stats['std']:.3f}", flush=True)
old = samples.get(videos[2][1], {})
current = samples.get(videos[1][1], {})
record["old_vs_candidate_sampled_pixels"] = {}
for index in indexes:
    if index not in old or index not in current or old[index].shape != current[index].shape:
        continue
    result = {"equal": bool(np.array_equal(old[index], current[index])),
              "mae": float(np.abs(old[index].astype(np.int16) - current[index].astype(np.int16)).mean())}
    record["old_vs_candidate_sampled_pixels"][str(index)] = result
    print(f"[native sampled A/B] frame={index} equal={result['equal']} MAE={result['mae']:.6f}", flush=True)
sheet.save(out / "preview.jpg", quality=92)
(out / "sampled-pixels.json").write_text(json.dumps(record, indent=2) + "\n")
print("[preview]", out / "preview.jpg", flush=True)
PY

for run in fl2va-reference-20261111T105437Z fl2va-candidate-20261111T085622Z fl2va-songshan-20261109T123355Z; do
    printf '\n[configuration] %s\n' "$run"
    log="$WORK_HOME/logs/$run.log"
    if [[ -r "$log" ]]; then
        grep -E '^\[model\]|^[[:space:]]+(text|first frame|last frame|frames|steps|backend):|Some weights|not initialized|unexpected keys|missing keys' "$log" || true
    else
        echo "Missing generation log: $log"
    fi
done
source_file="$WORK_HOME/build-state-gfx936-v8/gfx936-build-kit/InfiniLM/csrc/models/minimax_h3/minimax_h3_transformer.cpp"
if [[ -r "$source_file" ]]; then
    printf '\n[source excerpt] %s\n' "$source_file"
    sed -n '325,357p' "$source_file"
fi
} 2>&1 | tee "$out/diagnose.log"
printf '\nDiagnostic results: %s\n' "$out"
)
