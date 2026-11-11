#!/usr/bin/env bash
# Diagnostic only: paired Qwen3-VL layer-50 embeddings, no video generation.
(
set -eo pipefail
umask 0002
ulimit -c 0
export SETUPTOOLS_USE_DISTUTILS=stdlib
export PYTHONFAULTHANDLER=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source "${IFAB_AFS_HOME:?}/minimax-h3/env_gfx936.sh"
"$PYTHON" -c 'import torch; assert torch.cuda.device_count() >= 8, "Use the allocated eight-DCU terminal"'
if [[ -z "${FA_SO_PATH:-}" ]]; then
  export FA_SO_PATH="$("$PYTHON" -c 'from importlib.util import find_spec; s=find_spec("flash_attn_2_cuda"); print(s.origin if s else "")')"
fi
out="$WORK_HOME/checks/h3-condition-paired-$(date -u +%Y%m%dT%H%M%SZ)-$$"
tmp="$(mktemp -d "${IFAB_WORKSPACE_ROOT:-$PWD}/h3-condition.XXXXXX")"
mkdir -p "$out" "$tmp/tmp"
export TMPDIR="$tmp/tmp"
cat > "$tmp/condition_probe.py" <<'PY'
from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib.util
import json
import os
import sys
import time
import torch

mode, out = sys.argv[1], Path(sys.argv[2])
work = Path(os.environ["WORK_HOME"])
model_root = Path("/public/home/aclsylqidf/shared/MiniMax-H3/diffusers")


def write_json(name, data):
    (out / name).write_text(json.dumps(data, indent=2, default=str) + "\n")


def sync_all():
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)


def stats(tensor):
    value = tensor.float()
    return {"shape": list(value.shape), "dtype": str(tensor.dtype),
            "finite": bool(torch.isfinite(value).all()), "mean": float(value.mean()),
            "rms": float(value.square().mean().sqrt()),
            "first_values": value.flatten()[:8].tolist()}


if mode == "prepare":
    # Also surface the small-DiT input-mutation check already saved last round.
    previous = work / "checks/h3-dit-paired-20261111T143714Z-1275606"
    for name in ("native2", "native8"):
        path = previous / f"{name}.stage.json"
        if path.is_file():
            data = json.loads(path.read_text())
            print("[previous-dit]", name, "mutated_input_fields=",
                  data.get("mutated_input_fields", "not recorded"), flush=True)
    from PIL import Image
    from diffusers import ModularPipeline
    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3FL2VATextEncoderStep
    from diffusers.modular_pipelines.modular_pipeline import PipelineState

    image_path = Path(os.environ["IFAB_AFS_HOME"]) / "songshan-fl2va-first-frame.png"
    if not image_path.is_file():
        raise SystemExit(f"Missing official first frame: {image_path}")
    config = json.loads((model_root / "text_encoder/config.json").read_text())
    pipe = ModularPipeline.from_pretrained(str(model_root), local_files_only=True)
    pipe.load_components(names=["processor", "tokenizer"],
                         pretrained_model_name_or_path=str(model_root), local_files_only=True)
    prepared = {}

    class Captured(Exception):
        pass

    class CaptureModel:
        def __call__(self, **kwargs):
            # This intercept is diagnostic-only and stops before model computation.
            # The official block/helper creates all token IDs and vision inputs.
            prepared.clear()
            prepared.update({key: value.detach().cpu().clone() if torch.is_tensor(value) else value
                             for key, value in kwargs.items()})
            raise Captured()

    capture_encoder = SimpleNamespace(
        config=SimpleNamespace(text_config=SimpleNamespace(**config["text_config"])),
        dtype=torch.bfloat16, model=CaptureModel())
    components = SimpleNamespace(
        processor=pipe.processor, tokenizer=pipe.tokenizer, text_tag=1, video_tag=0,
        text_encoder_layer=50, text_encoder=capture_encoder,
        _execution_device=torch.device("cpu"))
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    manifest = {"diagnostic_only": True, "prompt": "A dog falls asleep",
                "image": str(image_path),
                "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(), "cases": {}}
    for name, frames in (("image", [image]), ("text", [])):
        state = PipelineState(values={"keyframes": frames, "prompt": manifest["prompt"]})
        try:
            MiniMaxH3FL2VATextEncoderStep()(components, state)
        except Captured:
            pass
        else:
            raise RuntimeError("Official conditioner input capture did not run")
        torch.save(prepared, out / f"input.{name}.pt")
        manifest["cases"][name] = {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                                   if torch.is_tensor(value) else value
                                   for key, value in prepared.items()}
        print("[condition-input]", name, json.dumps(manifest["cases"][name]), flush=True)
    write_json("input.json", manifest)
    raise SystemExit(0)


if mode == "compare":
    records = []
    complete = True
    for case in ("image", "text"):
        a_path, b_path = out / f"native8.{case}.pt", out / f"reference.{case}.pt"
        if not a_path.is_file() or not b_path.is_file():
            print("[condition-missing]", case, "; see stage logs", flush=True)
            complete = False
            continue
        actual = torch.load(a_path, map_location="cpu", weights_only=True)["50"].float()
        reference = torch.load(b_path, map_location="cpu", weights_only=True)
        for layer, tensor in reference.items():
            expected = tensor.float()
            record = {"case": case, "reference_layer": int(layer),
                      "same_shape": list(actual.shape) == list(expected.shape)}
            if record["same_shape"]:
                delta = actual - expected
                rmse = delta.square().mean().sqrt()
                cos = torch.nn.functional.cosine_similarity(actual, expected, dim=-1)
                record.update(finite=bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
                              max_abs=float(delta.abs().max()), rmse=float(rmse),
                              relative_rmse=float(rmse / expected.square().mean().sqrt().clamp_min(1e-12)),
                              mean_token_cosine=float(cos.mean()), min_token_cosine=float(cos.min()))
                if case == "image" and layer == "50":
                    ids = torch.load(out / "input.image.pt", map_location="cpu", weights_only=True)
                    mask = ids["mm_token_type_ids"][0]
                    for tag, name in ((0, "text"), (1, "image")):
                        selected = mask == tag
                        if selected.any():
                            err = delta[:, selected]
                            ref = expected[:, selected]
                            record[f"{name}_relative_rmse"] = float(
                                err.square().mean().sqrt() / ref.square().mean().sqrt().clamp_min(1e-12))
                            record[f"{name}_mean_cosine"] = float(cos[:, selected].mean())
            records.append(record)
            print("[condition-compare]", json.dumps(record), flush=True)
    write_json("comparison.json", {"diagnostic_only": True, "model_quality_validated": False,
                                    "required_stages_complete": complete, "comparisons": records})
    print(f"[condition-probe] required_stages_complete={complete}; full_model_quality_validated=False", flush=True)
    print("[results]", out, flush=True)
    raise SystemExit(0 if complete else 1)


entry = work / "runtime-gfx936/examples/minimax_h3.py"
spec = importlib.util.spec_from_file_location("h3_condition_entry", entry)
example = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = example
spec.loader.exec_module(example)
from infinilm.multimodal.minimax_h3 import MiniMaxH3ConditionEncoder
from infinilm.cache import StaticKVCacheConfig
from infinilm.lib import _infinilm

stage = {"mode": mode, "diagnostic_only": True, "torch": torch.__version__, "hip": torch.version.hip,
         "extension": str(_infinilm.__file__),
         "extension_sha256": hashlib.sha256(Path(_infinilm.__file__).read_bytes()).hexdigest(),
         "model_root": str(model_root), "target_hidden_state": 50}
print("[condition-stage]", mode, "loading unchanged Qwen3-VL weights", flush=True)
started = time.perf_counter()
with torch.inference_mode():
    if mode == "reference":
        encoder = example.Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_root), subfolder="text_encoder", dtype=torch.bfloat16,
            device_map="balanced_low_0", max_memory=example.make_max_memory([0, 1, 2, 3], 0.4),
            low_cpu_mem_usage=True, local_files_only=True)
        stage["device_map"] = getattr(encoder, "hf_device_map", {})
        device = example.module_execution_device(encoder)
    elif mode == "native8":
        encoder = MiniMaxH3ConditionEncoder(model_root / "text_encoder", dtype=torch.bfloat16,
                                           tp_device_ids=list(range(8)))
        device = encoder.torch_device
        stage["hf_config"] = encoder.engine.hf_config
    else:
        raise ValueError(mode)
    encoder.requires_grad_(False).eval()
    print(f"[condition-stage] {mode} loaded in {time.perf_counter() - started:.1f}s", flush=True)
    for case in ("image", "text"):
        cpu_inputs = torch.load(out / f"input.{case}.pt", map_location="cpu", weights_only=True)
        inputs = {key: value.to(device) if torch.is_tensor(value) else value
                  for key, value in cpu_inputs.items()}
        if mode == "native8" and case != "image":
            sync_all()
            encoder.engine.reset_cache(StaticKVCacheConfig(1, 32768))
        hook = getattr(encoder, "_hf_hook", None)
        if hook is not None and hasattr(hook, "pre_forward"):
            hook.pre_forward(encoder)
        sync_all()
        print("[condition-forward]", mode, case, "tokens=", inputs["input_ids"].shape[-1], flush=True)
        result = encoder.model(**inputs)
        sync_all()
        layers = (49, 50, 51) if mode == "reference" else (50,)
        saved = {str(layer): result.hidden_states[layer].detach().cpu().contiguous() for layer in layers}
        torch.save(saved, out / f"{mode}.{case}.pt")
        summary = {layer: stats(value) for layer, value in saved.items()}
        write_json(f"{mode}.{case}.json", summary)
        print("[condition-output]", mode, case, json.dumps(summary), flush=True)
        del result, saved, inputs
    stage["stage_seconds"] = time.perf_counter() - started
    write_json(f"{mode}.stage.json", stage)
    print("[condition-stage-done]", mode, flush=True)
PY
cp "$tmp/condition_probe.py" "$out/probe_source.py"
"$PYTHON" -u "$tmp/condition_probe.py" prepare "$out" 2>&1 | tee "$out/prepare.log"
# Separate processes release all model memory between the two implementations.
for stage in reference native8; do
  if "$PYTHON" -u "$tmp/condition_probe.py" "$stage" "$out" 2>&1 | tee "$out/$stage.log"; then
    echo "[completed] $stage"
  else
    echo "[stage-failed] $stage; log retained"
  fi
done
"$PYTHON" -u "$tmp/condition_probe.py" compare "$out" 2>&1 | tee "$out/compare.log"
printf '\nCondition diagnostic directory: %s\n' "$out"
)
