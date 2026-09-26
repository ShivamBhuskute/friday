"""Make the CUDA runtime findable before CTranslate2 tries to load it.

CTranslate2 links against ``libcublas.so.12`` and the cuDNN 9 family, but the
system toolkit on this machine is CUDA 13. The ``nvidia-*-cu12`` wheels provide
the exact SONAMEs it wants inside the venv.

The dynamic linker resolves those lazily via ``dlopen`` and caches its search
path at process start, so exporting ``LD_LIBRARY_PATH`` from inside Python is
too late. Preloading each library by absolute path works regardless, and is a
no-op when the system libraries are already present.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger("friday.cuda")

# Ordered: dependencies before the libraries that consume them.
_SONAMES = (
    "libcublas.so.12",
    "libcublasLt.so.12",
    "libcudnn.so.9",
    "libcudnn_graph.so.9",
    "libcudnn_engines_precompiled.so.9",
    "libcudnn_engines_runtime_compiled.so.9",
    "libcudnn_heuristic.so.9",
    "libcudnn_ops.so.9",
    "libcudnn_adv.so.9",
    "libcudnn_cnn.so.9",
    "libnvrtc.so.12",
)

# Without these, CTranslate2 cannot run on CUDA at all.
_REQUIRED = frozenset({"libcublas.so.12", "libcudnn.so.9"})

_loaded = False


def _pip_lib_dirs() -> list[Path]:
    """Locate ``site-packages/nvidia/*/lib`` inside this environment."""
    dirs: list[Path] = []
    for entry in sys.path:
        if not entry:
            continue
        nvidia = Path(entry) / "nvidia"
        if not nvidia.is_dir():
            continue
        for child in nvidia.iterdir():
            lib = child / "lib"
            if lib.is_dir():
                dirs.append(lib)
    return dirs


def ensure_cuda_runtime() -> list[str]:
    """Preload the CUDA shared objects CTranslate2 needs.

    Returns the list of libraries that could not be found, for diagnostics.
    """
    global _loaded
    if _loaded:
        return []

    search = _pip_lib_dirs()
    # An explicit override always wins, for people who set LD_LIBRARY_PATH.
    extra = os.environ.get("FRIDAY_CUDA_LIB_DIR")
    if extra:
        search.insert(0, Path(p for p in extra.split(os.pathsep) if p))

    missing: list[str] = []
    for soname in _SONAMES:
        if _load_by_name(soname):
            continue
        path = _find(soname, search)
        if path is None:
            # Not every build needs every library; only report the core ones.
            if soname in _REQUIRED:
                missing.append(soname)
            continue
        try:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:  # pragma: no cover - environment specific
            if soname in _REQUIRED:
                log.debug("could not preload %s: %s", soname, exc)
                missing.append(soname)

    _loaded = True
    if search:
        log.debug("cuda runtime search path: %s", ", ".join(str(p) for p in search))
    return missing


def _load_by_name(soname: str) -> bool:
    """Load a SONAME through the default search path.

    Succeeds if the library is already resident (torch, a system toolkit) or
    reachable via ``LD_LIBRARY_PATH``, in which case no preload is needed.
    """
    try:
        ctypes.CDLL(soname, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        return False
    return True


def _find(soname: str, dirs: list[Path]) -> Path | None:
    for d in dirs:
        candidate = d / soname
        if candidate.exists():
            return candidate
    return None
