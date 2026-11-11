"""Run the InfiniLM example with math SDPA scoped to video VAE decode."""

import sys
from functools import wraps
from importlib.util import module_from_spec, spec_from_file_location

from torch.nn.attention import SDPBackend, sdpa_kernel


def main():
    entry = sys.argv.pop(1)
    sys.argv[0] = entry
    spec = spec_from_file_location("h3_entry", entry)
    example = module_from_spec(spec)
    sys.modules[spec.name] = example
    spec.loader.exec_module(example)
    load_pipeline = example.load_pipeline

    @wraps(load_pipeline)
    def load_pipeline_with_math_decode(*args, **kwargs):
        pipe, text_encoder, transformer = load_pipeline(*args, **kwargs)
        decode = pipe.vae.decode

        @wraps(decode)
        def decode_with_math(*args, **kwargs):
            print("[decode] Video VAE using math SDPA", flush=True)
            with sdpa_kernel(SDPBackend.MATH):
                return decode(*args, **kwargs)

        pipe.vae.decode = decode_with_math
        return pipe, text_encoder, transformer

    example.load_pipeline = load_pipeline_with_math_decode
    example.main()


if __name__ == "__main__":
    main()
