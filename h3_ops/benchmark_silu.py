"""Native vs Nine SiLU, all FP32 as in H3's actual time embedding module."""
import argparse
import functools
import json
from pathlib import Path
import statistics
import time
import torch

# Respect this Pod's CPU quota and keep input-generation thread pools from
# introducing CPU-throttling bursts into the subsequent GPU launch benchmark.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def prepared(backend,x,out):
    if backend=="native":
        import infinicore
        from infinicore.nn.functional import silu
        if getattr(infinicore,"use_ntops",False):
            raise RuntimeError("Native baseline must not activate ntops")
        return functools.partial(silu,infinicore.from_torch(x),out=infinicore.from_torch(out))
    from silu_nt import _kernel,prepare
    if backend=="nine-plan": return prepare(x,out=out)
    return functools.partial(_kernel(),x,out)


def fixture(n,d,scale):
    gen=torch.Generator().manual_seed(20261027+n+d)
    x=(torch.randn((n,d),generator=gen)*scale).to("cuda")
    edges=torch.tensor([-80.,-20.,-1.,0.,1.,20.,80.],device="cuda")
    x[0,:min(d,7)]=edges[:min(d,7)]
    return x


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--backend",choices=("native","nine","nine-plan"),required=True)
    p.add_argument("--mode",choices=("validate","bench"),required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--native-fixture")
    p.add_argument("--repeats",type=int,default=11)
    p.add_argument("--iterations",type=int,default=300)
    args=p.parse_args()
    dest=Path(args.output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    native=torch.load(args.native_fixture,weights_only=True) if args.native_fixture else {}
    if args.mode=="validate":
        cases=[(n,d,s) for n in (1,4,17) for d in (257,5376) for s in (1.,10.)]
    else: cases=[(n,5376,1.) for n in (1,4,17,128,1024)]
    records,tensors=[],{}
    for n,d,scale in cases:
        key=f"n{n}_d{d}_scale{scale:g}"
        x=fixture(n,d,scale)
        out=torch.empty_like(x)
        torch.cuda.synchronize()
        start=time.perf_counter()
        call=prepared(args.backend,x,out)
        call()
        torch.cuda.synchronize()
        preparation=time.perf_counter()-start
        expected=torch.nn.functional.silu(x)
        actual=out.cpu()
        torch.testing.assert_close(actual,expected.cpu(),atol=1e-5,rtol=1e-5)
        record={"key":key,"rows":n,"channels":d,"dtype":"float32","scale":scale,
                "preparation_seconds":preparation,
                "max_abs_vs_torch":(actual-expected.cpu()).abs().max().item()}
        if native:
            torch.testing.assert_close(actual,native[key],atol=1e-5,rtol=1e-5)
            record["max_abs_vs_native"]=(actual-native[key]).abs().max().item()
        if args.mode=="validate":
            if args.backend!="native":
                from silu_nt import silu
                public=silu(x)
                torch.cuda.synchronize()
                torch.testing.assert_close(public.cpu(),actual,atol=0,rtol=0)
            x.neg_()
            call()
            torch.cuda.synchronize()
            torch.testing.assert_close(out.cpu(),torch.nn.functional.silu(x).cpu(),atol=1e-5,rtol=1e-5)
            record["changed_input_pass"]=True
            tensors[key]=actual
        else:
            for _ in range(40): call()
            torch.cuda.synchronize()
            samples=[]
            for _ in range(args.repeats):
                torch.cuda.synchronize()
                start=time.perf_counter_ns()
                for _ in range(args.iterations): call()
                torch.cuda.synchronize()
                samples.append((time.perf_counter_ns()-start)/args.iterations/1000)
            torch.testing.assert_close(out.cpu(),expected.cpu(),atol=1e-5,rtol=1e-5)
            record.update(samples_us=samples,median_us=statistics.median(samples),
                          min_us=min(samples),max_us=max(samples))
        records.append(record)
        print(json.dumps(record),flush=True)
    if tensors: torch.save(tensors,dest.with_suffix(".pt"))
    dest.write_text(json.dumps({"status":"PASS","backend":args.backend,"mode":args.mode,
       "device":torch.cuda.get_device_name(0),"arch":torch.cuda.get_device_properties(0).gcnArchName,
       "torch":torch.__version__,"block":1024,"warps":4,"dtype":"float32",
       "iterations":args.iterations,"repeats":args.repeats,"torch_cpu_threads":1,
       "timestamp":time.strftime("%Y-%m-%dT%H:%M:%S%z"),
       "timing":"synchronized batched wall time; includes host dispatch; excludes allocation/preparation/JIT",
       "cases":records},indent=2)+"\n")
    print(f"{args.backend.upper()}_{args.mode.upper()}_PASS {len(records)} cases",flush=True)


if __name__=="__main__": main()
