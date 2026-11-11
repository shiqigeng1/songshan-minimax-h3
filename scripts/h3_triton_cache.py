"""Stage Triton cache entries next to their final paths."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from triton.runtime.cache import FileCacheManager


class SiblingFileCacheManager(FileCacheManager):
    def put(self, data, filename, binary=True):
        if not self.cache_dir:
            raise RuntimeError("Could not create or locate cache dir")

        filepath = Path(self._make_path(filename))
        binary = isinstance(data, bytes)
        temporary = None
        try:
            with NamedTemporaryFile(
                mode="wb" if binary else "w",
                dir=filepath.parent,
                prefix=".triton-tmp-",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(data if binary else str(data))
            os.replace(temporary, filepath)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return str(filepath)
