#!/usr/bin/env bash
# Diagnostic only: same small input, real unchanged DiT weights, separate processes.
(
set -eo pipefail
umask 0002
ulimit -c 0
export SETUPTOOLS_USE_DISTUTILS=stdlib
export PYTHONFAULTHANDLER=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source "${IFAB_AFS_HOME:?}/minimax-h3/env_gfx936.sh"
"$PYTHON" -c 'import torch; assert torch.cuda.device_count() >= 8, "Use the allocated eight-DCU terminal"'
out="$WORK_HOME/checks/h3-dit-paired-$(date -u +%Y%m%dT%H%M%SZ)-$$"
tmp="$(mktemp -d "${IFAB_WORKSPACE_ROOT:-$PWD}/h3-dit-probe.XXXXXX")"
mkdir -p "$out" "$tmp/tmp"
export TMPDIR="$tmp/tmp"
cat > "$tmp/dit_probe.py" <<'PY'
from pathlib import Path
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import time
import torch

mode, out = sys.argv[1], Path(sys.argv[2])
work = Path(os.environ["WORK_HOME"])
model_root = Path("/public/home/aclsylqidf/shared/MiniMax-H3/diffusers")

def sync_all():
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)

def write_json(name, value):
    (out / name).write_text(json.dumps(value, indent=2, default=str) + "\n")

def summarize(tensor):
    x = tensor.float()
    return {"shape": list(x.shape), "dtype": str(tensor.dtype),
            "finite": bool(torch.isfinite(x).all()), "mean": float(x.mean()),
            "std": float(x.std()), "rms": float(x.square().mean().sqrt()),
            "first_values": x.flatten()[:8].tolist()}

def save_output(name, video, audio):
    sync_all()
    tensors = {"video": video.detach().cpu().contiguous(),
               "audio": audio.detach().cpu().contiguous()}
    torch.save(tensors, out / f"{name}.pt")
    summary = {key: summarize(value) for key, value in tensors.items()}
    write_json(f"{name}.json", summary)
    print("[dit-output]", name, json.dumps(summary), flush=True)

if mode == "prepare":
    generator = torch.Generator("cpu").manual_seed(42)
    def randn(shape, dtype=torch.float32):
        return torch.randn(shape, generator=generator).to(dtype)
    text_rows, video_rows, audio_rows = 8, 64, 32
    n = text_rows + video_rows + audio_rows
    row = torch.arange(n, dtype=torch.float64)
    values = {
        "video_hidden_states": randn((1, video_rows, 96)),
        "audio_hidden_states": randn((1, audio_rows, 32)),
        "encoder_hidden_states": randn((1, text_rows, 5120), torch.bfloat16),
        "timestep": torch.tensor([0.0, 500.0], dtype=torch.float32),
        "timestep_indices": torch.cat((torch.zeros(text_rows, dtype=torch.int64),
                                         torch.ones(video_rows + audio_rows, dtype=torch.int64))),
        "token_tags": torch.tensor([1] * text_rows + [0] * video_rows + [2] * audio_rows, dtype=torch.int64),
        "position_ids": torch.stack((row / 16, (row % 8) / 8, (row % 13) / 13), dim=1),
        "text_indices": torch.arange(text_rows, dtype=torch.int64),
        "video_indices": torch.arange(text_rows, text_rows + video_rows, dtype=torch.int64),
        "audio_indices": torch.arange(text_rows + video_rows, n, dtype=torch.int64),
    }
    torch.save(values, out / "input.pt")
    write_json("input.json", {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                              for key, value in values.items()})
    print("[prepare] Saved one shared input; diagnostic sequence length=104, no video generation.", flush=True)
    raise SystemExit(0)

if mode == "compare":
    outputs = {p.stem: torch.load(p, map_location="cpu", weights_only=True)
               for p in out.glob("*.pt") if p.name != "input.pt"}
    results = []
    pairs = [(f"native{tp}.{variant}", "reference")
             for tp in (2, 8) for variant in ("normal", "sync")]
    pairs += [(f"native{tp}.normal", f"native{tp}.repeat") for tp in (2, 8)]
    pairs += [("native2.sync", "native8.sync")]
    for left, right in pairs:
        if left not in outputs or right not in outputs:
            print(f"[dit-missing] {left} or {right}; see stage log.", flush=True)
            continue
        for stream in ("video", "audio"):
            a, b = outputs[left][stream].float(), outputs[right][stream].float()
            record = {"left": left, "right": right, "stream": stream,
                      "same_shape": a.shape == b.shape}
            if a.shape == b.shape:
                error = a - b
                rmse = error.square().mean().sqrt()
                denominator = b.square().mean().sqrt().clamp_min(1e-12)
                record.update(finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                              max_abs=float(error.abs().max()), rmse=float(rmse),
                              relative_rmse=float(rmse / denominator),
                              equal=bool(torch.equal(a, b)))
            results.append(record)
            print("[dit-compare]", json.dumps(record), flush=True)
    complete = all(name in outputs for name in ("reference", "native2.sync", "native8.sync"))
    write_json("comparison.json", {"diagnostic_only": True, "model_quality_validated": False,
                                    "required_stages_complete": complete, "comparisons": results})
    print(f"[dit-probe] required_stages_complete={complete}; full_model_quality_validated=False", flush=True)
    print("[results]", out, flush=True)
    raise SystemExit(0 if complete else 1)

if torch.cuda.device_count() < 8:
    raise SystemExit("This stage needs the eight-DCU allocation.")

# Import the existing entry, preserving its reference loader/device hooks.
entry = work / "runtime-gfx936/examples/minimax_h3.py"
spec = importlib.util.spec_from_file_location("h3_paired_entry", entry)
example = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = example
spec.loader.exec_module(example)
from infinilm.multimodal.minimax_h3 import MiniMaxH3TransformerInputs, MiniMaxH3TransformerRunner
from infinilm.lib import _infinilm
cpu_inputs = torch.load(out / "input.pt", map_location="cpu", weights_only=True)
stage = {"mode": mode, "torch": torch.__version__, "hip": torch.version.hip,
         "extension": str(_infinilm.__file__), "model_root": str(model_root),
         "extension_sha256": hashlib.sha256(Path(_infinilm.__file__).read_bytes()).hexdigest(),
         "entry_sha256": hashlib.sha256(entry.read_bytes()).hexdigest(),
         "diagnostic_only": True}
print("[stage]", mode, "loading unchanged DiT weights", flush=True)
started = time.perf_counter()

with torch.inference_mode():
    if mode == "reference":
        model = example.MiniMaxH3Transformer3DModel.from_pretrained(
            str(model_root), subfolder="transformer", dtype=torch.bfloat16,
            device_map="balanced", max_memory=example.make_max_memory([4, 5, 6, 7], 0.4),
            low_cpu_mem_usage=True, local_files_only=True)
        model.requires_grad_(False)
        model.eval()
        example.install_legacy_output_device_hooks(model)
        device = example.module_execution_device(model.context_embedder)
        stage["device_map"] = getattr(model, "hf_device_map", {})
        stage["projection_dtypes"] = {name: str(next(getattr(model, name).parameters()).dtype)
                                      for name in ("proj_in", "audio_proj_in", "proj_out", "context_embedder")}
        args = {key: value.to(device) for key, value in cpu_inputs.items()}
        args["hidden_states"] = args.pop("video_hidden_states")
        sync_all()
        print(f"[stage] reference loaded in {time.perf_counter() - started:.1f}s; one small forward", flush=True)
        from torch.nn.attention import SDPBackend, sdpa_kernel
        # Small diagnostic sequence only: MATH fits easily and avoids Flash availability differences.
        with sdpa_kernel(SDPBackend.MATH):
            video, audio = model(**args, return_dict=False)
        save_output("reference", video, audio)
    else:
        tp = {"native2": 2, "native8": 8}[mode]
        runner = MiniMaxH3TransformerRunner(model_root / "transformer", device="nvidia",
                                           tp_device_ids=list(range(tp)))
        stage["hf_config"] = runner.engine.hf_config
        keys = list(runner.engine.state_dict_keyname())
        stage["parameter_key_count"] = len(keys)
        stage["parameter_key_sample"] = keys[:20]
        values = {key: value.to("cuda:0") for key, value in cpu_inputs.items()}
        inputs = MiniMaxH3TransformerInputs(**values)
        sync_all()
        print(f"[stage] {mode} loaded in {time.perf_counter() - started:.1f}s; normal/repeat/sync forwards", flush=True)
        for variant in ("normal", "repeat"):
            prediction = runner(inputs)
            save_output(f"{mode}.{variant}", prediction.sample, prediction.audio_sample)
        original_forward = runner.engine.forward_raw
        def synchronized_forward(*args, **kwargs):
            sync_all()
            output = original_forward(*args, **kwargs)
            sync_all()
            return output
        runner.engine.forward_raw = synchronized_forward
        prediction = runner(inputs)
        save_output(f"{mode}.sync", prediction.sample, prediction.audio_sample)
        stage["mutated_input_fields"] = [key for key, value in values.items()
                                         if not torch.equal(value.cpu(), cpu_inputs[key])]
    stage["stage_seconds"] = time.perf_counter() - started
    write_json(f"{mode}.stage.json", stage)
    print("[stage-done]", mode, flush=True)
PY
cp "$tmp/dit_probe.py" "$out/probe_source.py"
"$PYTHON" -u "$tmp/dit_probe.py" prepare "$out" 2>&1 | tee "$out/prepare.log"
# Process exit releases the model before the next stage; no simultaneous model copies.
for stage in reference native2 native8; do
    printf '\n[launch] %s\n' "$stage"
    if "$PYTHON" -u "$tmp/dit_probe.py" "$stage" "$out" 2>&1 | tee "$out/$stage.log"; then
        echo "[completed] $stage"
    else
        echo "[stage-failed] $stage; retaining the log and continuing other independent stages."
    fi
done
"$PYTHON" -u "$tmp/dit_probe.py" compare "$out" 2>&1 | tee "$out/compare.log"
printf '\nDiT diagnostic directory: %s\n' "$out"
)
