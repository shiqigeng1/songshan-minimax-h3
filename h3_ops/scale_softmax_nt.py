"""Experimental NineToothed scale/softmax for 2D BF16 H3 score rows."""

import argparse
from functools import lru_cache, partial
from statistics import median

import ninetoothed as nt
import ninetoothed.language as ntl
import torch


SCALE = 128**-0.5


def arrangement(scores, out, *, block):
    arranged = []
    for tensor in (scores, out):
        rows = tensor.tile((1, block)).flatten()
        rows.dtype = rows.dtype.squeeze(0)
        arranged.append(rows)
    return tuple(arranged)


def application(scores, out):
    # Preserve the separate native multiply's BF16 rounding before softmax.
    # The head dimension is 128, so the scale is 1 / sqrt(128).
    scaled = ntl.cast(ntl.cast(scores, ntl.float32) * 0.08838834764831845, ntl.bfloat16)
    values = ntl.cast(scaled, ntl.float32)
    numerator = ntl.exp(values - ntl.max(values))
    out = numerator / ntl.sum(numerator)


@lru_cache(maxsize=4)
def kernel_for_width(width, warps):
    block = 1 << (width - 1).bit_length()
    return nt.make(
        partial(arrangement, block=block),
        application,
        (
            nt.Tensor(2, dtype=nt.bfloat16, other=float("-inf")),
            nt.Tensor(2, dtype=nt.bfloat16),
        ),
        num_warps=warps,
        num_stages=1,
        max_num_configs=1,
        kernel_name=f"h3_bf16_scale_softmax_w{warps}",
    )


def average_ms(function):
    samples = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 10)
    return median(samples)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--warps", type=int, choices=(4, 8, 16), default=8)
    args = parser.parse_args()
    if not 1 <= args.rows <= 4096:
        parser.error("--rows must be between 1 and 4096")

    torch.manual_seed(42)
    for width in (257, 27570):
        scores = torch.randn(
            (args.rows, width), device="cuda", dtype=torch.bfloat16
        ).mul_(11.0)
        scaled = torch.empty_like(scores)
        expected = torch.empty_like(scores)
        out = torch.empty_like(scores)
        kernel = kernel_for_width(width, args.warps)

        def native():
            torch.mul(scores, SCALE, out=scaled)
            torch.ops.aten._softmax.out(scaled, -1, False, out=expected)

        def fused():
            kernel(scores, out)

        native()
        fused()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected, rtol=0.012, atol=1e-7)
        max_error = (out.float() - expected.float()).abs().max().item()
        print(
            f"[softmax-check] rows={args.rows} width={width} warps={args.warps} "
            f"dtype=bf16 max_abs_error={max_error:.8g}",
            flush=True,
        )
        if width == 27570:
            native_ms = average_ms(native)
            fused_ms = average_ms(fused)
            print(
                f"[softmax-bench] rows={args.rows} width={width} warps={args.warps} "
                f"native_ms={native_ms:.4f} nine_ms={fused_ms:.4f} "
                f"speedup={native_ms / fused_ms:.3f}x",
                flush=True,
            )


if __name__ == "__main__":
    main()
