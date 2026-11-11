#!/usr/bin/env bash
# Paste into a DCU terminal. No model weights, inference generation, or source edits.
(
set -eo pipefail
ulimit -c 0
export SETUPTOOLS_USE_DISTUTILS=stdlib
export PYTHONFAULTHANDLER=1
source "${IFAB_AFS_HOME:?}/minimax-h3/env_gfx936.sh"
out="$WORK_HOME/checks/h3-core-numeric-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$out"
"$PYTHON" -u - "$out" <<'PY' 2>&1 | tee "$out/probe.log"
from pathlib import Path
import hashlib
import inspect
import json
from importlib import metadata
import os
import sys
import traceback
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

if not torch.cuda.is_available():
    raise SystemExit("This probe needs a DCU terminal; no model weights are loaded.")

import infinicore as ic
import infinicore.nn.functional as icf
from infinicore.ops.mha import mha
from infinilm.multimodal.minimax_h3 import MiniMaxH3TransformerRunner as Bridge
from infinilm.lib import _infinilm

out = Path(sys.argv[1])
device = torch.device("cuda", 0)
ic.set_device(ic.device("cuda", 0))
generator = torch.Generator("cpu").manual_seed(123)
record = {"diagnostic_only": True, "model_quality_validated": False,
          "torch": torch.__version__, "hip": torch.version.hip,
          "device": torch.cuda.get_device_name(0), "cases": []}
record["python_dependencies"] = {name: metadata.version(name) for name in ("diffusers", "transformers", "accelerate")}
print("[environment]", record, flush=True)
for name, path in (("extension", Path(_infinilm.__file__)),
                   ("bridge_source", Path(inspect.getfile(Bridge)))):
    record[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    print(f"[{name}]", record[name], flush=True)

def random(shape, dtype):
    return torch.randn(shape, generator=generator).to(device=device, dtype=dtype)

def native(tensor):
    result = Bridge._to_infinicore(tensor)
    torch.cuda.synchronize(device)
    return result

def restored(tensor):
    torch.cuda.synchronize(device)
    result = Bridge._to_torch(tensor, device)
    torch.cuda.synchronize(device)
    return result

def measure(name, actual, expected, atol, rtol):
    a, b = actual.float(), expected.float()
    finite = bool(torch.isfinite(a).all())
    same_shape = tuple(a.shape) == tuple(b.shape)
    entry = {"name": name, "shape": list(a.shape), "dtype": str(actual.dtype),
             "finite": finite, "same_shape": same_shape, "atol": atol, "rtol": rtol}
    if finite and same_shape:
        error = a - b
        entry.update(max_abs=float(error.abs().max()),
                     rmse=float(error.square().mean().sqrt()),
                     reference_rms=float(b.square().mean().sqrt()),
                     passed=bool(torch.allclose(a, b, atol=atol, rtol=rtol)))
    else:
        entry["passed"] = False
    record["cases"].append(entry)
    print("[numeric]", json.dumps(entry), flush=True)

def case(name, function):
    print("[RUN]", name, flush=True)
    try:
        function()
    except Exception as error:
        record["cases"].append({"name": name, "passed": False,
                                 "error": str(error), "exception": type(error).__name__})
        traceback.print_exc()
    (out / "probe.json").write_text(json.dumps(record, indent=2) + "\n")

def roundtrip(dtype, noncontiguous):
    x = torch.arange(1 * 16 * 96, device=device, dtype=torch.float32).reshape(1, 16, 96).to(dtype)
    if noncontiguous:
        x = x.transpose(1, 2)
    name = f"roundtrip/{dtype}/noncontiguous={noncontiguous}"
    measure(name, restored(native(x)), x, 0.0, 0.0)

def projection(dtype, rows, n_in, n_out):
    x = random((1, rows, n_in), dtype)
    weight = (random((n_out, n_in), dtype) * 0.02).to(dtype)
    bias = (random((n_out,), dtype) * 0.02).to(dtype)
    expected = F.linear(x, weight, bias)
    nx, nw, nb = native(x), native(weight), native(bias)
    actual = restored(icf.linear(nx, nw, nb))
    tolerance = 1e-4 if dtype == torch.float32 else 0.03
    measure(f"linear/{dtype}/{rows}x{n_in}->{n_out}", actual, expected, tolerance, tolerance)

def normalization(dtype, width):
    x = random((1, 32, width), dtype)
    weight = (torch.ones(width, device=device) + random((width,), torch.float32) * 0.05).to(dtype)
    expected = F.rms_norm(x, (width,), weight, eps=1e-5)
    nx, nw = native(x), native(weight)
    actual = restored(icf.rms_norm(nx, nw.shape, nw, eps=1e-5))
    tolerance = 1e-4 if dtype == torch.float32 else 0.03
    measure(f"rms_norm/{dtype}/{width}", actual, expected, tolerance, tolerance)

def packed_rows(dtype):
    base = torch.zeros((1, 13, 96), device=device, dtype=dtype)
    indexes = torch.tensor([1, 4, 8, 12], device=device, dtype=torch.int64)
    source = random((1, 4, 96), dtype)
    expected = base.index_copy(1, indexes, source)
    nb, ni, ns = native(base), native(indexes), native(source)
    actual = restored(ic.index_copy(nb, 1, ni, ns))
    measure(f"index_copy/{dtype}", actual, expected, 0.0, 0.0)
    table = random((13, 96), dtype)
    nt, ni = native(table), native(indexes)
    actual = restored(icf.embedding(ni, nt))
    measure(f"embedding/{dtype}", actual, table.index_select(0, indexes), 0.0, 0.0)

def attention(dtype, causal):
    q, k, v = (random((1, 181, 7, 128), dtype) for _ in range(3))
    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            dropout_p=0.0, is_causal=causal).transpose(1, 2).contiguous()
    nq, nk, nv = native(q), native(k), native(v)
    actual = restored(mha(nq, nk, nv, scale=128 ** -0.5, is_causal=causal))
    tolerance = 3e-4 if dtype == torch.float32 else 0.03
    measure(f"mha/{dtype}/causal={causal}", actual, expected, tolerance, tolerance)

with torch.inference_mode():
    for dtype in (torch.float32, torch.bfloat16):
        for noncontiguous in (False, True):
            case(f"roundtrip {dtype} {noncontiguous}", lambda d=dtype, n=noncontiguous: roundtrip(d, n))
        for rows, n_in, n_out in ((32, 96, 5376), (2048, 96, 5376), (32, 5376, 96)):
            case(f"linear {dtype} {rows} {n_in} {n_out}", lambda d=dtype, m=rows, k=n_in, n=n_out: projection(d, m, k, n))
        for width in (128, 5376):
            case(f"rms_norm {dtype} {width}", lambda d=dtype, w=width: normalization(d, w))
        case(f"packed_rows {dtype}", lambda d=dtype: packed_rows(d))
        for causal in (False, True):
            case(f"attention {dtype} {causal}", lambda d=dtype, c=causal: attention(d, c))

failures = sum(not entry["passed"] for entry in record["cases"])
record["failures"] = failures
(out / "probe.json").write_text(json.dumps(record, indent=2) + "\n")
print(f"[probe] DONE: failures={failures}; full_model_quality_validated=False", flush=True)
print("[results]", out, flush=True)
raise SystemExit(1 if failures else 0)
PY
)
