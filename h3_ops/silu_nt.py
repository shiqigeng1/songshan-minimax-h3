"""NineToothed FP32 SiLU used by MiniMaxH3TimeEmbedder.

Scope: contiguous 2D float32 tensors on the supplied gfx906 Pod. This preserves
the H3 module's actual FP32 dtype. Low-precision experimental kernels are not
called by this implementation.
"""
import functools
import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed.language import libdevice
import torch


def _arrangement(x, out):
    arranged = []
    for t in (x,out):
        t = t.tile((1,1024)).flatten()
        t.dtype = t.dtype.squeeze(0)
        arranged.append(t)
    return tuple(arranged)


def _application(x,out):
    value = ntl.cast(x,ntl.float32)
    out = value * (1.0 / (1.0 + libdevice.exp(-value)))


@functools.lru_cache(maxsize=1)
def _kernel():
    return nt.make(_arrangement,_application,
                   (nt.Tensor(2,dtype=nt.float32),nt.Tensor(2,dtype=nt.float32)),
                   num_warps=4,num_stages=1,max_num_configs=1,
                   kernel_name="h3_time_embed_silu_fp32")


def _validate(x,out):
    if x.ndim != 2 or x.shape != out.shape:
        raise ValueError("input/output must share a two-dimensional shape")
    if x.dtype != torch.float32 or out.dtype != torch.float32:
        raise TypeError("this H3 TimeEmbedder specialization requires float32")
    if x.device.type != "cuda" or x.device != out.device:
        raise ValueError("input/output must share the HIP device")
    if not x.is_contiguous() or not out.is_contiguous():
        raise ValueError("this specialization requires contiguous input/output")


def silu(x,*,out=None):
    if out is None:
        out=torch.empty_like(x)
    _validate(x,out)
    if x.numel(): _kernel()(x,out)
    return out


def prepare(x,*,out):
    """Compile with Nine, then bind stable tensors/shape/stream to its launcher.

    Every invocation recomputes SiLU. Updating input contents in-place is valid;
    storage/shape/dtype/device/stream changes require a new plan. Preparation
    must be single-threaded. It uses fixed-version compiler host interfaces.
    """
    _validate(x,out)
    if not x.numel(): return lambda: None
    handle=_kernel()
    jit=handle._kernel
    original=jit.run
    captured=[]

    def capture(*args,**kwargs):
        compiled=original(*args,**kwargs)
        captured.append((compiled,args))
        return compiled

    jit.run=capture
    try:
        handle(x,out)
    finally:
        jit.run=original
    if len(captured)!=1: raise RuntimeError("Unexpected Nine launch sequence")
    compiled,args=captured[0]
    grid=(x.shape[0]*((x.shape[1]+1023)//1024),1,1)
    stream=torch.cuda.current_stream(x.device)
    from triton import knobs
    enter=knobs.runtime.launch_enter_hook
    exit_hook=knobs.runtime.launch_exit_hook
    host=compiled.run
    if (enter is None and exit_hook is None
            and compiled.metadata.target.backend=="hip"
            and host.global_scratch_size==0 and host.profile_scratch_size==0
            and not host.launch_cooperative_grid):
        # The normal Nine call above has validated pointers on this device.
        # Pin storage lifetimes and bind its compiler-provided C launch API.
        # Integer pointers are an explicitly supported HIP launcher argument
        # form; this avoids repeated data_ptr()/hipPointerGetAttribute calls.
        # Reprepare if storage, stream, or instrumentation hooks change.
        packed_args=tuple(a.data_ptr() if isinstance(a,torch.Tensor) else a for a in args)
        call=functools.partial(host.launch,host.launch_cooperative_grid,
                               grid[0],grid[1],grid[2],stream.cuda_stream,compiled.function,
                               None,None,compiled.packed_metadata,None,None,None,
                               host.warp_size,host.arg_annotations,host.kernel_signature,packed_args)
    else:
        call=functools.partial(compiled[grid],*args,stream=stream.cuda_stream)
    call.input=x
    call.output=out
    call.stream=stream
    call.compiled=compiled
    call.ninetoothed_source=handle._source
    return call
