"""Opt-in compiler configuration: use the platform's elementary math bitcode.

This changes compiler builtin linkage, not tensor kernels. Numerical code still
comes entirely from the NineToothed arrangement/application. No binary or source
in the installed toolchain is overwritten. Cache keys include selected paths.
"""
from pathlib import Path
import hashlib
import subprocess


def configure():
    from triton.backends.amd.compiler import HIPOptions
    if getattr(HIPOptions, "_h3_dtk_math", False):
        return
    root = Path("/opt/dtk/amdgcn/bitcode")
    inputs = tuple(root / name for name in (
        "ocml.bc", "oclc_daz_opt_off.bc", "oclc_unsafe_math_off.bc",
        "oclc_finite_only_off.bc", "oclc_correctly_rounded_sqrt_off.bc",
        "oclc_isa_version_906.bc"))
    for path in inputs:
        if not path.is_file():
            raise RuntimeError(f"DTK builtin library missing: {path}")
    digest = hashlib.sha256(b"override-v2" + b"".join(p.read_bytes() for p in inputs)).hexdigest()[:16]
    build = Path(__file__).resolve().parent / "build"
    build.mkdir(exist_ok=True)
    ocml = build / f"ocml-dtk-dazoff-{digest}.bc"
    if not ocml.is_file():
        # DTK 26.04 declares __oclc_daz_opt as i8. Supply its own definition;
        # Triton's newer stock math library uses a different builtin ABI.
        # --override preserves linkonce_odr builtin definitions before there is
        # a model kernel referencing them. Ordinary pre-linking drops them.
        subprocess.run(["/opt/dtk/llvm/bin/llvm-link", str(inputs[0]),
                        *(f"--override={p}" for p in inputs),
                        "-o", str(ocml)], check=True)
    paths = {"ocml": str(ocml)}
    original = HIPOptions.__post_init__

    def post_init(self):
        original(self)
        if self.arch != "gfx906":
            return
        libs = dict(self.extern_libs)
        libs.update(paths)
        object.__setattr__(self, "extern_libs", tuple(libs.items()))
        object.__setattr__(self, "enable_fp_fusion", False)

    HIPOptions.__post_init__ = post_init
    HIPOptions._h3_dtk_math = True

