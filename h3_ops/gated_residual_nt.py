"""Benchmark the H3 DiT BF16 gate * branch + residual on one DCU."""

import time
from statistics import median

import infinicore
import ninetoothed as nt
import ninetoothed.language as ntl
import torch


def arrangement(residual, gate, branch, out):
    tiles = []
    for tensor in (residual, gate, branch, out):
        rows = tensor.tile((1, 1024)).flatten()
        rows.dtype = rows.dtype.squeeze(0)
        tiles.append(rows)
    return tuple(tiles)


def application(residual, gate, branch, out):
    product = ntl.cast(
        ntl.cast(gate, ntl.float32) * ntl.cast(branch, ntl.float32),
        ntl.bfloat16,
    )
    out = ntl.cast(residual, ntl.float32) + ntl.cast(product, ntl.float32)


def median_ms(call):
    iterations = 10
    samples = []
    for _ in range(5):
        infinicore.sync_device()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            call()
        infinicore.sync_device()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000 / iterations)
    return median(samples)


@torch.inference_mode()
def main():
    if getattr(infinicore, "use_ntops", False):
        raise RuntimeError("Native comparison requires InfiniCore's original operators")
    torch.manual_seed(42)
    width = 5376
    options = {"device": "cuda", "dtype": torch.bfloat16}
    infinicore.set_device(infinicore.device("cuda", torch.cuda.current_device()))
    fused_kernel = nt.make(
        arrangement, application, tuple(nt.Tensor(2, dtype=nt.bfloat16) for _ in range(4)),
        num_warps=4, num_stages=1, max_num_configs=1,
        kernel_name="h3_gated_residual_bf16",
    )

    for rows in (512, 27570):
        residual = torch.randn((rows, width), **options)
        branch = torch.randn_like(residual)
        packed = torch.empty((rows, 6 * width), **options)
        gate = packed[:, 2 * width : 3 * width].normal_()
        product = torch.empty_like(residual)
        expected = torch.empty_like(residual)
        actual = torch.empty_like(residual)

        # from_torch() wraps a contiguous shape in this InfiniCore revision.
        span = (rows - 1) * gate.stride(0) + width
        storage = gate.as_strided((span,), (1,), storage_offset=gate.storage_offset())
        native_gate = infinicore.from_torch(storage).as_strided(
            list(gate.shape), list(gate.stride())
        )
        native_residual, native_branch, native_product, native_expected = map(
            infinicore.from_torch, (residual, branch, product, expected)
        )

        def native():
            infinicore.mul(native_gate, native_branch, out=native_product)
            infinicore.add(native_residual, native_product, out=native_expected)

        def fused():
            fused_kernel(residual, gate, branch, actual)

        torch.cuda.synchronize()
        native()
        infinicore.sync_device()
        fused()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        print(f"[gate-check] rows={rows} width={width} stride={gate.stride()} exact", flush=True)
        native_ms = median_ms(native)
        fused_ms = median_ms(fused)
        print(
            f"[gate-bench] rows={rows} native_ms={native_ms:.4f} "
            f"nine_ms={fused_ms:.4f} speedup={native_ms / fused_ms:.3f}x",
            flush=True,
        )


if __name__ == "__main__":
    main()
