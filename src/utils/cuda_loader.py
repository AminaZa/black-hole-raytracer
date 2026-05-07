"""CuPy / CUDA DLL discovery and loading.

CuPy on Windows pip installs (``cupy-cuda12x`` plus the ``nvidia-*-cu12``
wheels) ships its DLLs under ``site-packages/nvidia/<lib>/bin``. Python and
NVRTC's transitive ``LoadLibraryEx`` calls do not search those paths by
default, so we have to register them with ``os.add_dll_directory`` *and*
prepend them to ``PATH`` before ``import cupy``. On Linux / Mac this whole
file is a no-op and ``import cupy`` just works.

Use :func:`load_cupy` to get a (cupy, available) tuple. The module also
exposes :data:`xp_module` and :data:`gpu_available` for callers that want a
single drop-in numpy-or-cupy choice.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from types import ModuleType


def _candidate_nvidia_dirs() -> list[Path]:
    """Return the ``site-packages/nvidia/<lib>/bin`` directories that exist."""
    out: list[Path] = []
    seen: set[str] = set()
    for entry in sys.path:
        if not entry:
            continue
        nvidia_root = Path(entry) / "nvidia"
        if not nvidia_root.is_dir():
            continue
        if str(nvidia_root) in seen:
            continue
        seen.add(str(nvidia_root))
        for child in nvidia_root.iterdir():
            bin_dir = child / "bin"
            if bin_dir.is_dir():
                out.append(bin_dir)
    return out


def _register_dll_paths() -> list[Path]:
    """Make the pip-installed CUDA DLL directories visible to the loader."""
    if os.name != "nt":
        return []
    dirs = _candidate_nvidia_dirs()
    path_extra = ";".join(str(d) for d in dirs)
    if path_extra:
        # Prepend so transitive LoadLibrary inside the DLLs (e.g. NVRTC →
        # nvrtc-builtins) finds siblings.
        os.environ["PATH"] = path_extra + ";" + os.environ.get("PATH", "")
    for d in dirs:
        try:
            os.add_dll_directory(str(d))
        except OSError:
            # Already added or path unreadable — harmless.
            pass

    runtime = next((d for d in dirs if d.parent.name == "cuda_runtime"), None)
    if runtime is not None and not os.environ.get("CUDA_PATH"):
        os.environ["CUDA_PATH"] = str(runtime.parent)
    return dirs


_REGISTERED_DIRS: list[Path] = _register_dll_paths()


def load_cupy() -> tuple[ModuleType | None, bool]:
    """Try to import CuPy with a working CUDA context.

    Returns ``(cupy_module, True)`` on success, ``(None, False)`` otherwise.
    A short warning is emitted on import failure so callers know they fell
    back to the CPU path.
    """
    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover - environment-dependent
        warnings.warn(f"CuPy not available ({exc}); falling back to CPU.", stacklevel=2)
        return None, False

    try:
        # Force a JIT to confirm NVRTC is reachable; without this we'd only
        # learn about a broken stack at the first kernel launch.
        probe = cp.arange(4, dtype=cp.float64) * 2.0
        cp.cuda.Stream.null.synchronize()
        _ = cp.asnumpy(probe)
    except Exception as exc:  # pragma: no cover - environment-dependent
        warnings.warn(
            f"CuPy is installed but a probe kernel failed ({exc}); falling back to CPU.",
            stacklevel=2,
        )
        return None, False

    return cp, True


_cupy, gpu_available = load_cupy()
cupy: ModuleType | None = _cupy

if _cupy is not None:
    xp_module: ModuleType = _cupy
else:
    import numpy as _np

    xp_module = _np


def device_summary() -> str:
    """One-line description of the active CUDA device, or 'CPU only'."""
    if not gpu_available or _cupy is None:
        return "CPU only (CuPy/CUDA not available)"
    props = _cupy.cuda.runtime.getDeviceProperties(0)
    name = props["name"].decode("utf-8", errors="replace")
    mem_gib = props["totalGlobalMem"] / (1024**3)
    cc = f"{props['major']}.{props['minor']}"
    return f"{name} ({mem_gib:.1f} GiB, CC {cc})"


def asnumpy(x):  # type: ignore[no-untyped-def]
    """``x`` → numpy ndarray. Works on numpy or cupy arrays."""
    if _cupy is not None and isinstance(x, _cupy.ndarray):
        return _cupy.asnumpy(x)
    return x


def get_xp(array) -> ModuleType:  # type: ignore[no-untyped-def]
    """Return the array module (numpy or cupy) the given array belongs to."""
    if _cupy is not None and isinstance(array, _cupy.ndarray):
        return _cupy
    import numpy as _np

    return _np
