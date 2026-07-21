"""Runtime compatibility hooks for the local Fish Speech service."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path


def _patch_windows_inductor_atomic_write() -> None:
    """Use replace semantics for TorchInductor cache files on Torch < 2.6."""
    if sys.platform != "win32":
        return

    import torch

    version = tuple(
        int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2]
    )
    if version >= (2, 6):
        return

    from torch._inductor import codecache

    def write_atomic(
        path_: str,
        content: str | bytes,
        make_dirs: bool = False,
        encode_utf_8: bool = False,
    ) -> None:
        path = Path(path_)
        if make_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.parent / f".{os.getpid()}.{threading.get_ident()}.tmp"
        mode = "w" if isinstance(content, str) else "wb"
        with temp_path.open(
            mode,
            encoding="utf-8" if encode_utf_8 else None,
        ) as output:
            output.write(content)
        temp_path.replace(path)

    codecache.write_atomic = write_atomic


_patch_windows_inductor_atomic_write()
