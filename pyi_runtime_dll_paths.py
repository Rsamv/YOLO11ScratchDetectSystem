"""Prepare native DLL search paths before frozen imports run.

PyInstaller runtime hooks execute before the application imports ``ui.py``.
That timing matters for ONNX Runtime: its ``onnxruntime_pybind11_state.pyd``
is imported very early and needs the packaged CUDA/cuDNN/TensorRT DLL folders
to be visible before Python starts resolving native dependencies.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path


def _add_dll_dir(path: Path, seen: set[str]) -> None:
    if os.name != "nt" or not path.exists() or not path.is_dir():
        return
    resolved = str(path.resolve())
    key = resolved.lower()
    if key in seen:
        return
    seen.add(key)
    try:
        os.add_dll_directory(resolved)
    except (AttributeError, FileNotFoundError, OSError):
        pass
    current_path = os.environ.get("PATH", "")
    path_parts = {p.lower() for p in current_path.split(os.pathsep) if p}
    if key not in path_parts:
        os.environ["PATH"] = resolved + os.pathsep + current_path


if os.name == "nt":
    app_root = Path(sys.executable).resolve().parent
    bundle_root = Path(getattr(sys, "_MEIPASS", app_root)).resolve()
    candidates = [
        app_root,
        bundle_root,
        bundle_root / "onnxruntime" / "capi",
        bundle_root / "tensorrt_bindings",
        bundle_root / "tensorrt_libs",
        bundle_root / "torch" / "lib",
        bundle_root / "nvidia" / "cuda_runtime" / "bin",
        app_root / "_internal",
        app_root / "_internal" / "onnxruntime" / "capi",
        app_root / "_internal" / "tensorrt_bindings",
        app_root / "_internal" / "tensorrt_libs",
        app_root / "_internal" / "torch" / "lib",
        app_root / "_internal" / "nvidia" / "cuda_runtime" / "bin",
    ]
    seen_dirs: set[str] = set()
    for candidate in candidates:
        _add_dll_dir(candidate, seen_dirs)

    # Preload the core ORT DLLs from the packaged location so the extension
    # module does not accidentally bind to a stale system PATH copy.
    for root in (bundle_root, app_root / "_internal"):
        capi = root / "onnxruntime" / "capi"
        for name in ("onnxruntime.dll", "onnxruntime_providers_shared.dll"):
            dll = capi / name
            if dll.exists():
                try:
                    ctypes.WinDLL(str(dll))
                except OSError:
                    pass
