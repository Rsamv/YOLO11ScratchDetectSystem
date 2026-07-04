"""TensorRT-backed YOLO detector used by the Qt frontend.

The original project used an Ultralytics/PyTorch runtime directly. This file
keeps the public ``v5detect`` API stable while making the TensorRT path more
portable across Windows and Linux so the project can be packaged into Docker
images for supported NVIDIA GPU devices.
"""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
FILE = Path(__file__).resolve()
SOURCE_ROOT = FILE.parent
APP_ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else SOURCE_ROOT
_MEIPASS = Path(getattr(sys, "_MEIPASS", str(SOURCE_ROOT))).resolve()
_INTERNAL = APP_ROOT / "_internal"
BUNDLE_ROOT = _MEIPASS if (_MEIPASS / "weights").exists() else (
    _INTERNAL if _INTERNAL.exists() else APP_ROOT
)
ROOT = APP_ROOT
if str(SOURCE_ROOT) not in sys.path:
    sys.path.append(str(SOURCE_ROOT))


def _bootstrap_runtime_dll_dirs() -> None:
    if os.name != "nt":
        return
    candidates = [
        APP_ROOT,
        APP_ROOT / "runtime",
        APP_ROOT / "runtime" / "cuda" / "bin",
        APP_ROOT / "runtime" / "cudnn" / "bin",
        APP_ROOT / "runtime" / "tensorrt" / "bin",
        APP_ROOT / "_internal" / "onnxruntime" / "capi",
        APP_ROOT / "_internal" / "tensorrt_bindings",
        APP_ROOT / "_internal" / "tensorrt_libs",
        APP_ROOT / "_internal" / "torch" / "lib",
        APP_ROOT / "_internal" / "nvidia" / "cuda_runtime" / "bin",
        BUNDLE_ROOT,
        BUNDLE_ROOT / "onnxruntime" / "capi",
        BUNDLE_ROOT / "tensorrt_bindings",
        BUNDLE_ROOT / "tensorrt_libs",
        BUNDLE_ROOT / "torch" / "lib",
        BUNDLE_ROOT / "nvidia" / "cuda_runtime" / "bin",
        BUNDLE_ROOT / "runtime",
        BUNDLE_ROOT / "runtime" / "cuda" / "bin",
        BUNDLE_ROOT / "runtime" / "cudnn" / "bin",
        BUNDLE_ROOT / "runtime" / "tensorrt" / "bin",
    ]
    seen = set()
    for directory in candidates:
        path = str(Path(directory).resolve())
        if path in seen or not os.path.isdir(path):
            continue
        seen.add(path)
        try:
            os.add_dll_directory(path)
        except (AttributeError, FileNotFoundError, OSError):
            pass
        current_path = os.environ.get("PATH", "")
        if path.lower() not in {p.lower() for p in current_path.split(os.pathsep) if p}:
            os.environ["PATH"] = path + os.pathsep + current_path


_bootstrap_runtime_dll_dirs()

TRT_IMPORT_ERROR: Optional[BaseException] = None
ORT_IMPORT_ERROR: Optional[BaseException] = None

try:
    import tensorrt as trt
except Exception as exc:
    trt = None
    TRT_IMPORT_ERROR = exc

try:
    import onnxruntime as ort
except Exception as exc:
    ort = None
    ORT_IMPORT_ERROR = exc

TRT_AVAILABLE = trt is not None
ORT_AVAILABLE = ort is not None
DEFAULT_MODEL_PATH = BUNDLE_ROOT / "weights" / "best.trt"
DEFAULT_IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))
DEFAULT_OUTPUT_FORMAT = os.getenv("YOLO_OUTPUT_FORMAT", "auto").strip().lower()
DEFAULT_TRT_WORKSPACE_GIB = float(os.getenv("YOLO_TRT_WORKSPACE_GIB", "2"))


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


DEFAULT_TRT_FP16 = _env_flag("YOLO_TRT_FP16", True)
DEFAULT_AUTO_BUILD_ENGINE = _env_flag("YOLO_AUTO_BUILD_TRT", True)
DEFAULT_FORCE_ENGINE_REBUILD = _env_flag("YOLO_TRT_FORCE_REBUILD", False)
DEFAULT_TRT_PINNED_OUTPUT = _env_flag("YOLO_TRT_PINNED_OUTPUT", True)
DEFAULT_COPY_FRAME_OUTPUT = _env_flag("YOLO_COPY_FRAME_OUTPUT", False)
DEFAULT_MAX_NMS_CANDIDATES = int(os.getenv("YOLO_MAX_NMS_CANDIDATES", "300"))
DEFAULT_PROFILE_EVERY_N = int(os.getenv("YOLO_PROFILE_EVERY_N", "0"))
DEFAULT_TRT_MIN_IMGSZ = int(os.getenv("YOLO_TRT_MIN_IMGSZ", "416"))
DEFAULT_TRT_OPT_IMGSZ = int(os.getenv("YOLO_TRT_OPT_IMGSZ", os.getenv("YOLO_REALTIME_IMGSZ", "416")))


class DetectorError(RuntimeError):
    """Raised when the TensorRT detector cannot be initialized or executed."""


class CpuFallbackRequiredError(DetectorError):
    """Raised when GPU backends failed and the caller must decide on CPU fallback."""

    def __init__(
        self,
        message: str,
        requested_model_path: Path,
        onnx_path: Optional[Path],
        gpu_attempts: Sequence[str],
    ) -> None:
        super().__init__(message)
        self.requested_model_path = requested_model_path
        self.onnx_path = onnx_path
        self.gpu_attempts = list(gpu_attempts)


def _candidate_cuda_dirs() -> List[str]:
    dirs: List[str] = []
    app_dirs = [
        str(APP_ROOT),
        str(APP_ROOT / "runtime"),
        str(APP_ROOT / "runtime" / "cuda"),
        str(APP_ROOT / "runtime" / "cuda" / "bin"),
        str(APP_ROOT / "runtime" / "cudnn"),
        str(APP_ROOT / "runtime" / "cudnn" / "bin"),
        str(APP_ROOT / "runtime" / "tensorrt"),
        str(APP_ROOT / "runtime" / "tensorrt" / "bin"),
        str(APP_ROOT / "_internal"),
        str(APP_ROOT / "_internal" / "onnxruntime" / "capi"),
        str(APP_ROOT / "_internal" / "tensorrt_bindings"),
        str(APP_ROOT / "_internal" / "tensorrt_libs"),
        str(APP_ROOT / "_internal" / "torch" / "lib"),
        str(APP_ROOT / "_internal" / "nvidia" / "cuda_runtime" / "bin"),
        str(BUNDLE_ROOT),
        str(BUNDLE_ROOT / "onnxruntime" / "capi"),
        str(BUNDLE_ROOT / "tensorrt_bindings"),
        str(BUNDLE_ROOT / "tensorrt_libs"),
        str(BUNDLE_ROOT / "torch" / "lib"),
        str(BUNDLE_ROOT / "nvidia" / "cuda_runtime" / "bin"),
        str(BUNDLE_ROOT / "runtime"),
        str(BUNDLE_ROOT / "runtime" / "cuda"),
        str(BUNDLE_ROOT / "runtime" / "cuda" / "bin"),
        str(BUNDLE_ROOT / "runtime" / "cudnn"),
        str(BUNDLE_ROOT / "runtime" / "cudnn" / "bin"),
        str(BUNDLE_ROOT / "runtime" / "tensorrt"),
        str(BUNDLE_ROOT / "runtime" / "tensorrt" / "bin"),
        str(Path(sys.executable).resolve().parent),
    ]
    dirs.extend(app_dirs)
    env_candidates = (
        os.getenv("CUDA_PATH"),
        os.getenv("CUDA_HOME"),
        os.getenv("CUDA_ROOT"),
        os.getenv("NVIDIA_TENSORRT_ROOT"),
    )
    for value in env_candidates:
        if not value:
            continue
        if os.name == "nt":
            dirs.extend(
                [
                    os.path.join(value, "bin"),
                    os.path.join(value, "lib", "x64"),
                ]
            )
        else:
            dirs.extend(
                [
                    os.path.join(value, "lib64"),
                    os.path.join(value, "targets", "x86_64-linux", "lib"),
                ]
            )

    for path_item in sys.path:
        if os.name == "nt":
            dirs.extend(
                [
                    os.path.join(path_item, "Library", "bin"),
                    os.path.join(path_item, "torch", "lib"),
                ]
            )

    if os.name != "nt":
        dirs.extend(
            [
                "/usr/local/cuda/lib64",
                "/usr/local/cuda/targets/x86_64-linux/lib",
                "/usr/lib/x86_64-linux-gnu",
                "/usr/lib/wsl/lib",
            ]
        )

    seen = set()
    result: List[str] = []
    for directory in dirs:
        if directory and os.path.isdir(directory) and directory not in seen:
            seen.add(directory)
            result.append(directory)
    return result


def _candidate_cudart_names() -> List[str]:
    if os.name == "nt":
        return [
            "cudart64_12.dll",
            "cudart64_120.dll",
            "cudart64_11.dll",
            "cudart64_110.dll",
            "cudart64_101.dll",
            "cudart64_102.dll",
            "cudart64_100.dll",
            "cudart64_90.dll",
        ]
    return [
        "libcudart.so",
        "libcudart.so.12",
        "libcudart.so.11.0",
        "libcudart.so.10.2",
        "libcudart.so.10.1",
    ]


def _load_cudart_library() -> ctypes.CDLL:
    for directory in _candidate_cuda_dirs():
        if os.name == "nt":
            try:
                os.add_dll_directory(directory)
            except (AttributeError, FileNotFoundError, OSError):
                pass
        for name in _candidate_cudart_names():
            full_path = os.path.join(directory, name)
            if os.path.exists(full_path):
                return ctypes.CDLL(full_path)

    for name in _candidate_cudart_names():
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue

    raise DetectorError(
        "CUDA runtime library was not found. Install a CUDA-enabled TensorRT "
        "runtime in the container/host, or make sure libcudart/cudart64 is on "
        "the library search path."
    )


class CudaRuntime:
    CUDA_MEMCPY_HOST_TO_DEVICE = 1
    CUDA_MEMCPY_DEVICE_TO_HOST = 2

    def __init__(self) -> None:
        _prepare_windows_dll_search_dirs()
        self.lib = _load_cudart_library()
        self._bind()

    def _bind(self) -> None:
        self.lib.cudaMalloc.restype = int
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaFree.restype = int
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaMemcpyAsync.restype = int
        self.lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.lib.cudaStreamCreate.restype = int
        self.lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cudaStreamSynchronize.restype = int
        self.lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamDestroy.restype = int
        self.lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self._host_alloc_available = False
        try:
            self.lib.cudaHostAlloc.restype = int
            self.lib.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
            self.lib.cudaFreeHost.restype = int
            self.lib.cudaFreeHost.argtypes = [ctypes.c_void_p]
            self._host_alloc_available = True
        except AttributeError:
            self._host_alloc_available = False

    @staticmethod
    def _check(code: int, action: str) -> None:
        if code != 0:
            raise DetectorError(f"{action} failed with CUDA error code {code}")

    def malloc(self, nbytes: int) -> ctypes.c_void_p:
        ptr = ctypes.c_void_p()
        self._check(self.lib.cudaMalloc(ctypes.byref(ptr), int(nbytes)), f"cudaMalloc({nbytes})")
        return ptr

    def free(self, ptr: Optional[ctypes.c_void_p]) -> None:
        if ptr:
            self._check(self.lib.cudaFree(ptr), "cudaFree")

    def memcpy_htod_async(self, dst_ptr: ctypes.c_void_p, src_np: np.ndarray, stream: ctypes.c_void_p) -> None:
        self._check(
            self.lib.cudaMemcpyAsync(
                dst_ptr,
                src_np.ctypes.data_as(ctypes.c_void_p),
                int(src_np.nbytes),
                self.CUDA_MEMCPY_HOST_TO_DEVICE,
                stream,
            ),
            "cudaMemcpyAsync(HtoD)",
        )

    def memcpy_dtoh_async(self, dst_np: np.ndarray, src_ptr: ctypes.c_void_p, stream: ctypes.c_void_p) -> None:
        self._check(
            self.lib.cudaMemcpyAsync(
                dst_np.ctypes.data_as(ctypes.c_void_p),
                src_ptr,
                int(dst_np.nbytes),
                self.CUDA_MEMCPY_DEVICE_TO_HOST,
                stream,
            ),
            "cudaMemcpyAsync(DtoH)",
        )

    def stream_create(self) -> ctypes.c_void_p:
        stream = ctypes.c_void_p()
        self._check(self.lib.cudaStreamCreate(ctypes.byref(stream)), "cudaStreamCreate")
        return stream

    def stream_synchronize(self, stream: ctypes.c_void_p) -> None:
        self._check(self.lib.cudaStreamSynchronize(stream), "cudaStreamSynchronize")

    def stream_destroy(self, stream: Optional[ctypes.c_void_p]) -> None:
        if stream:
            self._check(self.lib.cudaStreamDestroy(stream), "cudaStreamDestroy")

    def host_empty(
        self,
        shape: Tuple[int, ...],
        dtype: np.dtype,
    ) -> Tuple[np.ndarray, Optional[ctypes.c_void_p], Optional[object]]:
        dtype = np.dtype(dtype)
        nbytes = int(np.prod(shape)) * dtype.itemsize
        if not self._host_alloc_available or nbytes <= 0:
            return np.empty(shape, dtype=dtype), None, None

        ptr = ctypes.c_void_p()
        code = self.lib.cudaHostAlloc(ctypes.byref(ptr), int(nbytes), 0)
        if code != 0 or not ptr.value:
            return np.empty(shape, dtype=dtype), None, None

        owner = (ctypes.c_byte * nbytes).from_address(ptr.value)
        return np.ndarray(shape=shape, dtype=dtype, buffer=owner), ptr, owner

    def free_host(self, ptr: Optional[ctypes.c_void_p]) -> None:
        if ptr and self._host_alloc_available:
            self._check(self.lib.cudaFreeHost(ptr), "cudaFreeHost")


def get_trt_logger() -> "trt.Logger":
    if not TRT_AVAILABLE:
        raise DetectorError(
            "TensorRT Python bindings are not installed. Install the TensorRT "
            "runtime that matches the target container/device."
        )
    return trt.Logger(trt.Logger.WARNING)


def get_ort_available_providers() -> List[str]:
    if not ORT_AVAILABLE:
        return []
    try:
        return list(ort.get_available_providers())
    except Exception:
        return []


def _prepare_windows_dll_search_dirs() -> None:
    if os.name != "nt":
        return
    for directory in _candidate_cuda_dirs():
        try:
            os.add_dll_directory(directory)
        except (AttributeError, FileNotFoundError, OSError):
            pass


def preload_ort_runtime_dlls() -> None:
    if not ORT_AVAILABLE:
        return
    _prepare_windows_dll_search_dirs()
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        return
    try:
        preload()
        return
    except Exception:
        pass

    for directory in _candidate_cuda_dirs():
        try:
            preload(directory=directory)
            return
        except Exception:
            continue


def _backup_existing_engine(engine_path: Path) -> None:
    if not engine_path.exists():
        return
    backup_path = engine_path.with_suffix(engine_path.suffix + ".old")
    shutil.copy2(engine_path, backup_path)


def _model_family_paths(model_path: Path) -> Tuple[Path, Path]:
    suffix = model_path.suffix.lower()
    if suffix == ".onnx":
        return model_path.with_suffix(".trt"), model_path
    if suffix == ".trt":
        return model_path, model_path.with_suffix(".onnx")
    raise DetectorError(f"Unsupported model file type: {model_path.name}. Use .trt or .onnx.")


def _engine_meta_path(engine_path: Path) -> Path:
    return engine_path.with_suffix(engine_path.suffix + ".meta.json")


def _engine_build_signature(
    onnx_path: Path,
    imgsz: int = DEFAULT_IMGSZ,
    opt_imgsz_min: Optional[int] = DEFAULT_TRT_MIN_IMGSZ,
    opt_imgsz: Optional[int] = DEFAULT_TRT_OPT_IMGSZ,
    enable_fp16: bool = DEFAULT_TRT_FP16,
) -> Dict[str, object]:
    try:
        onnx_mtime_ns = onnx_path.stat().st_mtime_ns
    except OSError:
        onnx_mtime_ns = 0
    return {
        "onnx": onnx_path.name,
        "onnx_mtime_ns": onnx_mtime_ns,
        "max_imgsz": int(_make_divisible(int(imgsz), 32)),
        "min_imgsz": int(_make_divisible(int(opt_imgsz_min or imgsz), 32)),
        "opt_imgsz": int(_make_divisible(int(opt_imgsz or imgsz), 32)),
        "fp16": bool(enable_fp16),
        "tensorrt": str(getattr(trt, "__version__", "")),
    }


def _engine_meta_matches(engine_path: Path, onnx_path: Path) -> bool:
    meta_path = _engine_meta_path(engine_path)
    if not meta_path.exists():
        return False
    try:
        with meta_path.open("r", encoding="utf-8") as fp:
            meta = json.load(fp)
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    expected = _engine_build_signature(onnx_path)
    return all(meta.get(key) == value for key, value in expected.items())


def _should_rebuild_engine(engine_path: Path, onnx_path: Path, force_rebuild: bool) -> bool:
    if not onnx_path.exists():
        return False
    if force_rebuild or not engine_path.exists():
        return True
    try:
        return onnx_path.stat().st_mtime > engine_path.stat().st_mtime or not _engine_meta_matches(engine_path, onnx_path)
    except OSError:
        return False


def _collect_parser_errors(parser: "trt.OnnxParser") -> str:
    messages: List[str] = []
    for index in range(parser.num_errors):
        messages.append(str(parser.get_error(index)))
    return "\n".join(messages) if messages else "Unknown ONNX parsing failure."


def _set_trt_builder_optimization_level(config: "trt.IBuilderConfig", level: int = 5) -> None:
    """TensorRT exposes optimization level as a property in newer bindings."""
    if hasattr(config, "set_builder_optimization_level"):
        config.set_builder_optimization_level(int(level))
        return
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = int(level)


def build_engine_from_onnx(
    onnx_path: Path,
    engine_path: Optional[Path] = None,
    imgsz: int = DEFAULT_IMGSZ,
    workspace_gib: float = DEFAULT_TRT_WORKSPACE_GIB,
    enable_fp16: bool = DEFAULT_TRT_FP16,
    force_rebuild: bool = False,
    logger: Optional["trt.Logger"] = None,
    opt_imgsz_min: Optional[int] = None,
    opt_imgsz: Optional[int] = None,
) -> Path:
    if not TRT_AVAILABLE:
        raise DetectorError("TensorRT is unavailable, so a .trt engine cannot be built from ONNX.")
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model does not exist: {onnx_path}")

    logger = logger or get_trt_logger()
    engine_path = engine_path or onnx_path.with_suffix(".trt")
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with onnx_path.open("rb") as fp:
        model_bytes = fp.read()
    if not parser.parse(model_bytes):
        raise DetectorError(
            f"Failed to parse ONNX model {onnx_path.name}.\n{_collect_parser_errors(parser)}"
        )

    if network.num_inputs != 1:
        raise DetectorError(f"Only single-input ONNX models are supported, got {network.num_inputs}.")

    config = builder.create_builder_config()
    if workspace_gib > 0:
        workspace_bytes = int(workspace_gib * (1 << 30))
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)

    can_use_fp16 = bool(enable_fp16 and getattr(builder, "platform_has_fast_fp16", False))
    if can_use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    _set_trt_builder_optimization_level(config, 5)

    timing_cache = config.create_timing_cache(b"")
    config.set_timing_cache(timing_cache, ignore_mismatch=False)

    input_tensor = network.get_input(0)
    input_shape = [int(dim) for dim in input_tensor.shape]
    if len(input_shape) != 4:
        raise DetectorError(f"Only 4D NCHW inputs are supported, got {tuple(input_shape)}.")

    input_batch = 1 if input_shape[0] <= 0 else input_shape[0]
    input_channels = 3 if input_shape[1] <= 0 else input_shape[1]
    input_height = _make_divisible(imgsz, 32) if input_shape[2] <= 0 else input_shape[2]
    input_width = _make_divisible(imgsz, 32) if input_shape[3] <= 0 else input_shape[3]

    if any(dim <= 0 for dim in input_shape):
        profile = builder.create_optimization_profile()

        min_hw = _make_divisible(opt_imgsz_min, 32) if opt_imgsz_min is not None else input_height
        min_hw = min(min_hw, input_height)

        opt_hw = _make_divisible(opt_imgsz, 32) if opt_imgsz is not None else input_height
        opt_hw = max(min_hw, min(opt_hw, input_height))

        min_shape = (input_batch, input_channels, min_hw, min_hw)
        opt_shape = (input_batch, input_channels, opt_hw, opt_hw)
        max_shape = (input_batch, input_channels, input_height, input_width)
        profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise DetectorError(
            "TensorRT failed to build a serialized engine from ONNX. "
            "This usually means the current GPU/runtime cannot compile this model."
        )

    if force_rebuild and engine_path.exists():
        _backup_existing_engine(engine_path)

    with engine_path.open("wb") as fp:
        fp.write(bytes(serialized_engine))
    try:
        with _engine_meta_path(engine_path).open("w", encoding="utf-8") as fp:
            json.dump(
                _engine_build_signature(
                    onnx_path,
                    imgsz=imgsz,
                    opt_imgsz_min=opt_imgsz_min,
                    opt_imgsz=opt_imgsz,
                    enable_fp16=enable_fp16,
                ),
                fp,
                ensure_ascii=True,
                indent=2,
            )
    except OSError:
        pass
    return engine_path


def _make_divisible(v: int, divisor: int = 32) -> int:
    return int(np.ceil(v / divisor) * divisor)


def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = np.array(boxes, copy=True, dtype=np.float32)
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return converted


def _numpy_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0.0)

        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    return keep


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    if len(boxes) == 0:
        return []

    try:
        boxes_xyxy = np.asarray(boxes, dtype=np.float32)
        boxes_cv = np.column_stack(
            (
                boxes_xyxy[:, 0],
                boxes_xyxy[:, 1],
                np.maximum(0.0, boxes_xyxy[:, 2] - boxes_xyxy[:, 0]),
                np.maximum(0.0, boxes_xyxy[:, 3] - boxes_xyxy[:, 1]),
            )
        ).tolist()
        scores_cv = np.asarray(scores, dtype=np.float32).tolist()
        indices = cv2.dnn.NMSBoxes(
            boxes_cv,
            scores_cv,
            score_threshold=0.0,
            nms_threshold=float(iou_threshold),
        )
        if indices is None or len(indices) == 0:
            return []
        return [int(i) for i in np.asarray(indices).reshape(-1)]
    except Exception as exc:
        logger.debug("OpenCV NMS failed, falling back to NumPy NMS: %s", exc)
        return _numpy_nms(boxes, scores, iou_threshold)


def letterbox(image: np.ndarray, imgsz: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    height, width = image.shape[:2]
    scale = min(imgsz / float(height), imgsz / float(width))
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    dw = imgsz - new_width
    dh = imgsz - new_height
    left = int(round(dw / 2.0 - 0.1))
    right = int(round(dw / 2.0 + 0.1))
    top = int(round(dh / 2.0 - 0.1))
    bottom = int(round(dh / 2.0 + 0.1))

    bordered = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return bordered, scale, (left, top)


def scale_boxes_to_original(
    boxes: np.ndarray,
    orig_shape: Tuple[int, int],
    network_shape: Tuple[int, int],
    scale: float,
    pad: Tuple[int, int],
) -> np.ndarray:
    if len(boxes) == 0:
        return boxes

    scaled = np.array(boxes, copy=True, dtype=np.float32)
    input_h, input_w = network_shape

    max_coord = float(np.max(np.abs(scaled[:, :4]))) if scaled.size else 0.0
    if max_coord <= 2.0:
        scaled[:, [0, 2]] *= float(input_w)
        scaled[:, [1, 3]] *= float(input_h)

    pad_w, pad_h = pad
    scaled[:, [0, 2]] -= float(pad_w)
    scaled[:, [1, 3]] -= float(pad_h)
    scaled[:, :4] /= max(scale, 1e-6)

    orig_h, orig_w = orig_shape
    scaled[:, [0, 2]] = np.clip(scaled[:, [0, 2]], 0, orig_w)
    scaled[:, [1, 3]] = np.clip(scaled[:, [1, 3]], 0, orig_h)
    return scaled


def _looks_like_xyxy(boxes: np.ndarray) -> bool:
    if len(boxes) == 0:
        return True
    return bool(np.mean(boxes[:, 2] >= boxes[:, 0]) > 0.7 and np.mean(boxes[:, 3] >= boxes[:, 1]) > 0.7)


def _parse_nms_plugin_outputs(
    outputs: Sequence[np.ndarray],
    conf_thres: float,
) -> Optional[List[Tuple[float, float, float, float, float, int]]]:
    if len(outputs) < 4:
        return None

    arrays = [np.asarray(out) for out in outputs]
    boxes = scores = classes = num_dets = None
    for arr in arrays:
        squeezed = np.squeeze(arr)
        if squeezed.ndim == 0:
            num_dets = int(squeezed)
        elif squeezed.ndim == 1:
            if np.issubdtype(squeezed.dtype, np.integer):
                classes = squeezed.astype(np.int32)
            elif scores is None:
                scores = squeezed.astype(np.float32)
        elif squeezed.ndim == 2 and squeezed.shape[-1] == 4:
            boxes = squeezed.astype(np.float32)

    if boxes is None or scores is None or classes is None:
        return None

    if num_dets is None:
        num_dets = min(len(boxes), len(scores), len(classes))

    detections: List[Tuple[float, float, float, float, float, int]] = []
    for idx in range(min(num_dets, len(boxes), len(scores), len(classes))):
        score = float(scores[idx])
        if score < conf_thres:
            continue
        x1, y1, x2, y2 = boxes[idx].tolist()
        detections.append((x1, y1, x2, y2, score, int(classes[idx])))
    return detections


def _normalize_dense_output(output: np.ndarray) -> np.ndarray:
    normalized = np.asarray(output)
    while normalized.ndim > 2 and normalized.shape[0] == 1:
        normalized = normalized[0]

    if normalized.ndim != 2:
        raise DetectorError(f"Unsupported TensorRT output shape: {tuple(output.shape)}")

    if normalized.shape[1] <= 7:
        return normalized.astype(np.float32, copy=False)

    if normalized.shape[0] < normalized.shape[1]:
        return normalized.transpose().astype(np.float32, copy=False)

    return normalized.astype(np.float32, copy=False)


def _parse_end2end_rows(rows: np.ndarray, conf_thres: float) -> List[Tuple[np.ndarray, float, int]]:
    if rows.shape[1] < 6:
        return []

    detections: List[Tuple[np.ndarray, float, int]] = []
    if rows.shape[1] >= 7 and np.all(np.abs(rows[:, 0] - np.round(rows[:, 0])) < 1e-3):
        candidate_boxes = rows[:, 3:7]
        if _looks_like_xyxy(candidate_boxes):
            for row in rows:
                score = float(row[2])
                if score < conf_thres:
                    continue
                detections.append((row[3:7].astype(np.float32), score, int(row[1])))
            return detections

    candidate_boxes = rows[:, :4]
    boxes = candidate_boxes if _looks_like_xyxy(candidate_boxes) else xywh2xyxy(candidate_boxes)
    scores = rows[:, 4].astype(np.float32)
    classes = rows[:, 5].astype(np.int32)
    keep = scores >= conf_thres
    for box, score, cls_id in zip(boxes[keep], scores[keep], classes[keep]):
        detections.append((box.astype(np.float32), float(score), int(cls_id)))
    return detections


def _parse_dense_rows(
    rows: np.ndarray,
    conf_thres: float,
    output_format: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rows.shape[1] <= 4:
        empty_boxes = np.empty((0, 4), dtype=np.float32)
        empty_scores = np.empty((0,), dtype=np.float32)
        empty_classes = np.empty((0,), dtype=np.int32)
        return empty_boxes, empty_scores, empty_classes

    assume_objectness = output_format == "yolov5" or rows.shape[1] == 85
    if output_format == "yolov8":
        assume_objectness = False

    if assume_objectness and rows.shape[1] <= 5:
        assume_objectness = False

    boxes = rows[:, :4].astype(np.float32)
    if assume_objectness:
        objectness = rows[:, 4].astype(np.float32)
        class_scores = rows[:, 5:].astype(np.float32)
        if class_scores.size == 0:
            empty_boxes = np.empty((0, 4), dtype=np.float32)
            empty_scores = np.empty((0,), dtype=np.float32)
            empty_classes = np.empty((0,), dtype=np.int32)
            return empty_boxes, empty_scores, empty_classes
        class_ids = np.argmax(class_scores, axis=1).astype(np.int32)
        best_scores = class_scores[np.arange(len(class_scores)), class_ids]
        scores = objectness * best_scores
    else:
        class_scores = rows[:, 4:].astype(np.float32)
        class_ids = np.argmax(class_scores, axis=1).astype(np.int32)
        scores = class_scores[np.arange(len(class_scores)), class_ids]

    keep = scores >= conf_thres
    return boxes[keep], scores[keep].astype(np.float32), class_ids[keep].astype(np.int32)


def _limit_nms_candidates(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    limit = int(DEFAULT_MAX_NMS_CANDIDATES)
    if limit <= 0 or len(scores) <= limit:
        return boxes, scores, class_ids

    idx = np.argpartition(scores, -limit)[-limit:]
    order = idx[np.argsort(scores[idx])[::-1]]
    return boxes[order], scores[order], class_ids[order]


def postprocess_yolo(
    outputs: Sequence[np.ndarray],
    conf_thres: float,
    iou_thres: float,
    orig_shape: Tuple[int, int],
    network_shape: Tuple[int, int],
    scale: float,
    pad: Tuple[int, int],
    output_format: str,
) -> List[Tuple[float, float, float, float, float, int]]:
    parsed_nms = _parse_nms_plugin_outputs(outputs, conf_thres)
    if parsed_nms is not None:
        boxes = np.array([det[:4] for det in parsed_nms], dtype=np.float32)
        boxes = scale_boxes_to_original(boxes, orig_shape, network_shape, scale, pad)
        return [
            (float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(score), int(cls_id))
            for box, (_, _, _, _, score, cls_id) in zip(boxes, parsed_nms)
        ]

    output = _normalize_dense_output(outputs[0] if isinstance(outputs, (list, tuple)) else outputs)
    if output.shape[1] in (6, 7):
        raw = _parse_end2end_rows(output, conf_thres)
        if not raw:
            return []
        boxes = np.array([item[0] for item in raw], dtype=np.float32)
        scores = np.array([item[1] for item in raw], dtype=np.float32)
        class_ids = np.array([item[2] for item in raw], dtype=np.int32)
        boxes = scale_boxes_to_original(boxes, orig_shape, network_shape, scale, pad)
        boxes, scores, class_ids = _limit_nms_candidates(boxes, scores, class_ids)
        keep = nms(boxes, scores, iou_thres)
        return [
            (
                float(boxes[idx][0]),
                float(boxes[idx][1]),
                float(boxes[idx][2]),
                float(boxes[idx][3]),
                float(scores[idx]),
                int(class_ids[idx]),
            )
            for idx in keep
        ]

    boxes_xywh, scores, class_ids = _parse_dense_rows(output, conf_thres, output_format)
    if len(boxes_xywh) == 0:
        return []

    boxes = xywh2xyxy(boxes_xywh)
    boxes = scale_boxes_to_original(boxes, orig_shape, network_shape, scale, pad)
    boxes, scores, class_ids = _limit_nms_candidates(boxes, scores, class_ids)
    keep = nms(boxes, scores, iou_thres)

    results: List[Tuple[float, float, float, float, float, int]] = []
    for idx in keep:
        x1, y1, x2, y2 = boxes[idx].tolist()
        results.append((x1, y1, x2, y2, float(scores[idx]), int(class_ids[idx])))
    return results


@dataclass
class TensorBuffer:
    name: str
    dtype: np.dtype
    mode: "trt.TensorIOMode"
    shape: Tuple[int, ...] = ()
    nbytes: int = 0
    device: Optional[ctypes.c_void_p] = None
    host: Optional[np.ndarray] = None
    host_ptr: Optional[ctypes.c_void_p] = None
    host_owner: Optional[object] = None


class TensorRTDetector:
    """TensorRT detector implementation."""

    def __init__(self, model_path: Optional[str] = None):
        if not TRT_AVAILABLE:
            detail = f" Original import error: {TRT_IMPORT_ERROR}" if TRT_IMPORT_ERROR else ""
            raise DetectorError(
                "TensorRT Python bindings are missing. Install them in the target "
                f"runtime, or use the provided Docker image on a supported NVIDIA host.{detail}"
            )

        requested_model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not requested_model_path.exists():
            raise FileNotFoundError(f"Model file does not exist: {requested_model_path}")

        self.requested_model_path = requested_model_path
        self.model_path, self.onnx_path = _model_family_paths(requested_model_path)
        self.load_notes: List[str] = []
        self.rebuilt_engine = False
        self.engine_generated_from_onnx = False
        self.backend_name = "TensorRT"
        self.device_name = "CUDA"
        self.output_format = DEFAULT_OUTPUT_FORMAT if DEFAULT_OUTPUT_FORMAT in {"auto", "yolov5", "yolov8"} else "auto"
        self.cuda = CudaRuntime()
        self.logger = get_trt_logger()
        self.runtime = trt.Runtime(self.logger)
        self._lock = threading.Lock()

        should_rebuild_first = False
        if DEFAULT_AUTO_BUILD_ENGINE and self.requested_model_path.suffix.lower() == ".onnx":
            should_rebuild_first = True
        elif DEFAULT_AUTO_BUILD_ENGINE and _should_rebuild_engine(self.model_path, self.onnx_path, DEFAULT_FORCE_ENGINE_REBUILD):
            should_rebuild_first = True

        if should_rebuild_first:
            if not self.onnx_path.exists():
                raise DetectorError(
                    f"Requested ONNX rebuild but companion ONNX file does not exist: {self.onnx_path}"
                )
            build_reason = "selected ONNX model" if self.requested_model_path.suffix.lower() == ".onnx" else "ONNX file is newer than engine or rebuild was forced"
            self.load_notes.append(f"Building TensorRT engine from {self.onnx_path.name} because {build_reason}.")
            build_engine_from_onnx(
                self.onnx_path,
                engine_path=self.model_path,
                imgsz=DEFAULT_IMGSZ,
                workspace_gib=DEFAULT_TRT_WORKSPACE_GIB,
                enable_fp16=DEFAULT_TRT_FP16,
                force_rebuild=True,
                opt_imgsz_min=DEFAULT_TRT_MIN_IMGSZ,
                opt_imgsz=DEFAULT_TRT_OPT_IMGSZ,
                logger=self.logger,
            )
            self.rebuilt_engine = True
            self.engine_generated_from_onnx = True

        self.engine = self._deserialize_engine(self.model_path)
        if self.engine is None and DEFAULT_AUTO_BUILD_ENGINE and self.onnx_path.exists() and not self.rebuilt_engine:
            self.load_notes.append(
                f"TensorRT engine {self.model_path.name} could not be deserialized; attempting rebuild from {self.onnx_path.name}."
            )
            build_engine_from_onnx(
                self.onnx_path,
                engine_path=self.model_path,
                imgsz=DEFAULT_IMGSZ,
                workspace_gib=DEFAULT_TRT_WORKSPACE_GIB,
                enable_fp16=DEFAULT_TRT_FP16,
                force_rebuild=True,
                opt_imgsz_min=DEFAULT_TRT_MIN_IMGSZ,
                opt_imgsz=DEFAULT_TRT_OPT_IMGSZ,
                logger=self.logger,
            )
            self.rebuilt_engine = True
            self.engine_generated_from_onnx = True
            self.engine = self._deserialize_engine(self.model_path)

        if self.engine is None:
            hint = ""
            if self.onnx_path.exists():
                hint = (
                    f" Auto-rebuild from {self.onnx_path.name} also failed or the rebuilt engine "
                    "still does not match the current TensorRT/CUDA/GPU environment."
                )
            raise DetectorError(
                "TensorRT engine deserialization failed. The .trt file likely does "
                f"not match the current TensorRT/CUDA/GPU environment.{hint}"
            )

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise DetectorError("TensorRT execution context could not be created.")

        self.input_name: Optional[str] = None
        self.output_names: List[str] = []
        self.buffers: Dict[str, TensorBuffer] = {}

        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            buffer = TensorBuffer(name=name, dtype=dtype, mode=mode)
            self.buffers[name] = buffer
            if mode == trt.TensorIOMode.INPUT:
                if self.input_name is not None:
                    raise DetectorError("Only single-input TensorRT engines are supported.")
                self.input_name = name
            else:
                self.output_names.append(name)

        if self.input_name is None:
            raise DetectorError("No input tensor was found in the TensorRT engine.")
        if not self.output_names:
            raise DetectorError("No output tensors were found in the TensorRT engine.")

        self.stream = self.cuda.stream_create()
        self._requested_imgsz = DEFAULT_IMGSZ
        self.imgsz = DEFAULT_IMGSZ
        self._last_input_shape: Optional[Tuple[int, int, int, int]] = None
        self._scale = 1.0
        self._pad = (0, 0)
        self.profile_every = max(0, int(DEFAULT_PROFILE_EVERY_N))
        self._profile_count = 0
        self.last_profile: Dict[str, float] = {}

        self._configure_input_shape(DEFAULT_IMGSZ)

        self._warmup()

    def _warmup(self) -> None:
        """Run a dummy inference so CUDA kernels are compiled before the first real frame."""
        try:
            warmup_shape = (1, 3, self.imgsz, self.imgsz)
            dummy = np.zeros(warmup_shape, dtype=self.buffers[self.input_name].dtype)
            dummy = np.ascontiguousarray(dummy)
            self.cuda.memcpy_htod_async(self.buffers[self.input_name].device, dummy, self.stream)
            self.context.set_tensor_address(self.input_name, int(self.buffers[self.input_name].device.value))
            for name in self.output_names:
                self.context.set_tensor_address(name, int(self.buffers[name].device.value))
            self.context.execute_async_v3(stream_handle=int(self.stream.value))
            self.cuda.stream_synchronize(self.stream)
        except Exception:
            pass

    def _deserialize_engine(self, engine_path: Path):
        with engine_path.open("rb") as fp:
            engine_bytes = fp.read()
        return self.runtime.deserialize_cuda_engine(engine_bytes)

    def _engine_input_shape(self) -> Tuple[int, ...]:
        return tuple(int(dim) for dim in self.engine.get_tensor_shape(self.input_name))

    def _context_tensor_shape(self, name: str) -> Tuple[int, ...]:
        return tuple(int(dim) for dim in self.context.get_tensor_shape(name))

    def _free_buffer(self, buffer: TensorBuffer) -> None:
        if buffer.device is not None:
            try:
                self.cuda.free(buffer.device)
            finally:
                buffer.device = None
        if buffer.host_ptr is not None:
            try:
                self.cuda.free_host(buffer.host_ptr)
            finally:
                buffer.host_ptr = None
                buffer.host_owner = None
        buffer.host = None
        buffer.shape = ()
        buffer.nbytes = 0

    def _configure_input_shape(self, requested_imgsz: int) -> None:
        engine_shape = self._engine_input_shape()
        if len(engine_shape) != 4:
            raise DetectorError(f"Unsupported input tensor rank: {engine_shape}")

        if engine_shape[2] > 0 and engine_shape[3] > 0:
            final_shape = (1, int(engine_shape[1]), int(engine_shape[2]), int(engine_shape[3]))
            self.imgsz = int(engine_shape[2])
        else:
            self.imgsz = _make_divisible(int(requested_imgsz), 32)
            final_shape = (1, 3, self.imgsz, self.imgsz)
            result = self.context.set_input_shape(self.input_name, final_shape)
            if result is False:
                raise DetectorError(f"TensorRT rejected dynamic input shape {final_shape} for {self.input_name}.")

        if any(dim <= 0 for dim in final_shape):
            raise DetectorError(f"Invalid input shape resolved from engine: {final_shape}")

        self._ensure_buffers(final_shape)

    def _ensure_buffers(self, input_shape: Tuple[int, int, int, int]) -> None:
        if self._last_input_shape == input_shape and all(
            buffer.device is not None and (buffer.mode == trt.TensorIOMode.INPUT or buffer.host is not None)
            for buffer in self.buffers.values()
        ):
            return

        if self._last_input_shape is not None and self._last_input_shape != input_shape:
            for buffer in self.buffers.values():
                self._free_buffer(buffer)

        current_context_shape = self._context_tensor_shape(self.input_name)
        if current_context_shape != input_shape and any(dim < 0 for dim in self._engine_input_shape()):
            result = self.context.set_input_shape(self.input_name, input_shape)
            if result is False:
                raise DetectorError(f"TensorRT rejected input shape {input_shape} for {self.input_name}.")

        for name, buffer in self.buffers.items():
            if name == self.input_name:
                shape = input_shape
            else:
                shape = self._context_tensor_shape(name)
            if any(dim <= 0 for dim in shape):
                raise DetectorError(f"Output tensor {name} has unresolved shape {shape}.")

            nbytes = int(np.prod(shape)) * buffer.dtype.itemsize
            if buffer.device is None or buffer.nbytes != nbytes or buffer.shape != shape:
                self._free_buffer(buffer)
                buffer.device = self.cuda.malloc(nbytes)
                buffer.nbytes = nbytes
                buffer.shape = shape
                if buffer.mode != trt.TensorIOMode.INPUT:
                    if DEFAULT_TRT_PINNED_OUTPUT:
                        buffer.host, buffer.host_ptr, buffer.host_owner = self.cuda.host_empty(shape, buffer.dtype)
                    else:
                        buffer.host = np.empty(shape, dtype=buffer.dtype)
                        buffer.host_ptr = None
                        buffer.host_owner = None
                else:
                    buffer.host = None
                    buffer.host_ptr = None
                    buffer.host_owner = None

        self._last_input_shape = input_shape

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        framed, scale, pad = letterbox(image, self.imgsz)
        tensor = cv2.dnn.blobFromImage(
            framed,
            scalefactor=1.0 / 255.0,
            size=(self.imgsz, self.imgsz),
            swapRB=True,
            crop=False,
        )
        input_dtype = self.buffers[self.input_name].dtype
        if tensor.dtype != input_dtype:
            tensor = tensor.astype(input_dtype, copy=False)
        tensor = np.ascontiguousarray(tensor)
        self._scale = scale
        self._pad = pad
        return tensor

    def _record_profile(self, profile: Dict[str, float]) -> None:
        if not self.profile_every:
            return
        self._profile_count += 1
        self.last_profile = profile
        if self._profile_count % self.profile_every == 0:
            summary = " ".join(f"{key}={value:.2f}ms" for key, value in profile.items())
            logger.debug("[TensorRT profile] frame=%s %s", self._profile_count, summary)

    def detect(self, img: np.ndarray):
        return self.run(img)

    def run(
        self,
        im0s: np.ndarray,
        imgsz: Optional[int] = None,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        line_thickness: int = 3,
    ):
        del line_thickness
        if im0s is None or getattr(im0s, "size", 0) == 0:
            raise DetectorError("Input image is empty.")
        if im0s.ndim != 3 or im0s.shape[2] != 3:
            raise DetectorError("Input image must be a BGR HxWx3 array.")

        requested = imgsz if imgsz is not None else self._requested_imgsz
        if requested is None:
            requested = DEFAULT_IMGSZ

        prof_enabled = self.profile_every > 0
        t0 = time.perf_counter() if prof_enabled else 0.0

        with self._lock:
            self._configure_input_shape(int(requested))
            t_config = time.perf_counter() if prof_enabled else 0.0
            blob = self._preprocess(im0s)
            t_pre = time.perf_counter() if prof_enabled else 0.0
            input_shape = tuple(int(dim) for dim in blob.shape)
            self._ensure_buffers(input_shape)

            input_buffer = self.buffers[self.input_name]
            self.cuda.memcpy_htod_async(input_buffer.device, blob, self.stream)
            t_htod = time.perf_counter() if prof_enabled else 0.0

            self.context.set_tensor_address(self.input_name, int(input_buffer.device.value))
            for name in self.output_names:
                self.context.set_tensor_address(name, int(self.buffers[name].device.value))

            ok = self.context.execute_async_v3(stream_handle=int(self.stream.value))
            if ok is False:
                raise DetectorError("TensorRT execution returned failure from execute_async_v3().")
            t_exec = time.perf_counter() if prof_enabled else 0.0

            for name in self.output_names:
                buffer = self.buffers[name]
                self.cuda.memcpy_dtoh_async(buffer.host, buffer.device, self.stream)
            t_dtoh = time.perf_counter() if prof_enabled else 0.0

            self.cuda.stream_synchronize(self.stream)
            t_sync = time.perf_counter() if prof_enabled else 0.0

            outputs = [self.buffers[name].host for name in self.output_names]
            results = postprocess_yolo(
                outputs=outputs,
                conf_thres=float(conf_thres),
                iou_thres=float(iou_thres),
                orig_shape=tuple(int(v) for v in im0s.shape[:2]),
                network_shape=(self.imgsz, self.imgsz),
                scale=self._scale,
                pad=self._pad,
                output_format=self.output_format,
            )
            t_post = time.perf_counter() if prof_enabled else 0.0

        if prof_enabled:
            self._record_profile(
                {
                    "shape": (t_config - t0) * 1000.0,
                    "pre": (t_pre - t_config) * 1000.0,
                    "htod": (t_htod - t_pre) * 1000.0,
                    "enqueue": (t_exec - t_htod) * 1000.0,
                    "dtoh": (t_dtoh - t_exec) * 1000.0,
                    "sync": (t_sync - t_dtoh) * 1000.0,
                    "post": (t_post - t_sync) * 1000.0,
                    "total": (t_post - t0) * 1000.0,
                }
            )
        return results, im0s.copy() if DEFAULT_COPY_FRAME_OUTPUT else im0s

    def close(self) -> None:
        with self._lock:
            for buffer in self.buffers.values():
                try:
                    self._free_buffer(buffer)
                except Exception:
                    pass

            if getattr(self, "stream", None) is not None:
                try:
                    self.cuda.stream_synchronize(self.stream)
                except Exception:
                    pass
                try:
                    self.cuda.stream_destroy(self.stream)
                except Exception:
                    pass
                self.stream = None

            if getattr(self, "context", None) is not None:
                self.context = None
            if getattr(self, "engine", None) is not None:
                self.engine = None
            if getattr(self, "runtime", None) is not None:
                self.runtime = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class OnnxRuntimeDetector:
    """ONNX Runtime detector implementation for CUDA or CPU inference."""

    def __init__(self, model_path: str, provider: str = "cuda"):
        if not ORT_AVAILABLE:
            detail = f" Original import error: {ORT_IMPORT_ERROR}" if ORT_IMPORT_ERROR else ""
            raise DetectorError(
                f"onnxruntime is not installed. Install onnxruntime-gpu to enable CUDA and CPU fallback inference.{detail}"
            )
        preload_ort_runtime_dlls()

        requested_model_path = Path(model_path)
        if not requested_model_path.exists():
            raise FileNotFoundError(f"Model file does not exist: {requested_model_path}")

        if requested_model_path.suffix.lower() == ".trt":
            onnx_path = requested_model_path.with_suffix(".onnx")
        elif requested_model_path.suffix.lower() == ".onnx":
            onnx_path = requested_model_path
        else:
            raise DetectorError(f"Unsupported model file type: {requested_model_path.name}. Use .onnx or .trt.")

        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model does not exist: {onnx_path}")

        self.requested_model_path = requested_model_path
        self.onnx_path = onnx_path
        self.model_path = onnx_path
        self.load_notes: List[str] = []
        self.rebuilt_engine = False
        self.engine_generated_from_onnx = False
        self.output_format = DEFAULT_OUTPUT_FORMAT if DEFAULT_OUTPUT_FORMAT in {"auto", "yolov5", "yolov8"} else "auto"
        self.provider = provider.lower()
        self.backend_name = "ONNX Runtime CUDA" if self.provider == "cuda" else "ONNX Runtime CPU"
        self.device_name = "CUDA" if self.provider == "cuda" else "CPU"
        self._lock = threading.Lock()

        available_providers = get_ort_available_providers()
        provider_name = "CUDAExecutionProvider" if self.provider == "cuda" else "CPUExecutionProvider"
        if provider_name not in available_providers:
            raise DetectorError(
                f"{provider_name} is unavailable in onnxruntime. Available providers: {available_providers or ['none']}"
            )

        try:
            self.session = ort.InferenceSession(str(self.onnx_path), providers=[provider_name])
        except Exception as exc:
            raise DetectorError(f"Failed to create ONNX Runtime session with {provider_name}: {exc}") from exc

        active_providers = list(self.session.get_providers())
        if provider_name not in active_providers:
            raise DetectorError(
                f"Requested {provider_name}, but ONNX Runtime activated {active_providers or ['none']} instead."
            )

        self.input_meta = self.session.get_inputs()[0]
        self.input_name = self.input_meta.name
        input_shape = list(self.input_meta.shape)
        if len(input_shape) != 4:
            raise DetectorError(f"Only 4D NCHW ONNX inputs are supported, got {tuple(input_shape)}.")

        self._requested_imgsz = DEFAULT_IMGSZ
        self.imgsz = int(input_shape[2]) if isinstance(input_shape[2], int) and input_shape[2] > 0 else DEFAULT_IMGSZ
        self.dynamic_input = not (isinstance(input_shape[2], int) and input_shape[2] > 0 and isinstance(input_shape[3], int) and input_shape[3] > 0)
        self._scale = 1.0
        self._pad = (0, 0)
        if self.provider == "cuda":
            self._warmup()

    def _warmup(self) -> None:
        warmup_size = _make_divisible(int(self.imgsz if self.imgsz > 0 else DEFAULT_IMGSZ), 32)
        dummy = np.zeros((1, 3, warmup_size, warmup_size), dtype=np.float32)
        try:
            self.session.run(None, {self.input_name: dummy})
        except Exception as exc:
            detail = str(exc)
            if "NoKernelImageForDevice" in detail or "no kernel image is available" in detail:
                detail += (
                    "\nThe installed onnxruntime-gpu package does not contain CUDA kernels "
                    "for this GPU architecture. Use TensorRT on this machine, install an "
                    "onnxruntime-gpu/CUDA build that supports the GPU, or use CPU fallback."
                )
            raise DetectorError(f"ONNX Runtime CUDA warmup failed: {detail}") from exc

    def _preprocess(self, image: np.ndarray, imgsz: int) -> np.ndarray:
        framed, scale, pad = letterbox(image, imgsz)
        tensor = cv2.dnn.blobFromImage(
            framed,
            scalefactor=1.0 / 255.0,
            size=(imgsz, imgsz),
            swapRB=True,
            crop=False,
        )
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)
        self._scale = scale
        self._pad = pad
        return tensor

    def detect(self, img: np.ndarray):
        return self.run(img)

    def run(
        self,
        im0s: np.ndarray,
        imgsz: Optional[int] = None,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        line_thickness: int = 3,
    ):
        del line_thickness
        if im0s is None or getattr(im0s, "size", 0) == 0:
            raise DetectorError("Input image is empty.")
        if im0s.ndim != 3 or im0s.shape[2] != 3:
            raise DetectorError("Input image must be a BGR HxWx3 array.")

        requested = _make_divisible(int(imgsz if imgsz is not None else self._requested_imgsz), 32)
        network_size = requested if self.dynamic_input else self.imgsz
        blob = self._preprocess(im0s, network_size)

        with self._lock:
            try:
                outputs = self.session.run(None, {self.input_name: blob})
            except Exception as exc:
                raise DetectorError(f"ONNX Runtime inference failed on {self.backend_name}: {exc}") from exc

        results = postprocess_yolo(
            outputs=outputs,
            conf_thres=float(conf_thres),
            iou_thres=float(iou_thres),
            orig_shape=tuple(int(v) for v in im0s.shape[:2]),
            network_shape=(network_size, network_size),
            scale=self._scale,
            pad=self._pad,
            output_format=self.output_format,
        )
        return results, im0s.copy() if DEFAULT_COPY_FRAME_OUTPUT else im0s

    def close(self) -> None:
        self.session = None


class OpenCVDnnDetector:
    """CPU-only ONNX fallback that does not depend on ONNX Runtime."""

    def __init__(self, model_path: str):
        requested_model_path = Path(model_path)
        if requested_model_path.suffix.lower() == ".trt":
            onnx_path = requested_model_path.with_suffix(".onnx")
        elif requested_model_path.suffix.lower() == ".onnx":
            onnx_path = requested_model_path
        else:
            raise DetectorError(f"Unsupported model file type: {requested_model_path.name}. Use .onnx or .trt.")
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model does not exist: {onnx_path}")

        try:
            self.net = cv2.dnn.readNetFromONNX(str(onnx_path))
        except Exception as exc:
            raise DetectorError(f"OpenCV DNN failed to load ONNX model {onnx_path}: {exc}") from exc

        try:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except Exception:
            pass

        self.requested_model_path = requested_model_path
        self.onnx_path = onnx_path
        self.model_path = onnx_path
        self.load_notes: List[str] = []
        self.rebuilt_engine = False
        self.engine_generated_from_onnx = False
        self.output_format = DEFAULT_OUTPUT_FORMAT if DEFAULT_OUTPUT_FORMAT in {"auto", "yolov5", "yolov8"} else "auto"
        self.backend_name = "OpenCV DNN CPU"
        self.device_name = "CPU"
        self.imgsz = DEFAULT_IMGSZ
        self._requested_imgsz = DEFAULT_IMGSZ
        self._scale = 1.0
        self._pad = (0, 0)
        self._lock = threading.Lock()
        try:
            self.output_names = list(self.net.getUnconnectedOutLayersNames())
        except Exception:
            self.output_names = []

    def _preprocess(self, image: np.ndarray, imgsz: int) -> np.ndarray:
        framed, scale, pad = letterbox(image, imgsz)
        tensor = cv2.dnn.blobFromImage(
            framed,
            scalefactor=1.0 / 255.0,
            size=(imgsz, imgsz),
            swapRB=True,
            crop=False,
        )
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)
        self._scale = scale
        self._pad = pad
        return tensor

    def detect(self, img: np.ndarray):
        return self.run(img)

    def run(
        self,
        im0s: np.ndarray,
        imgsz: Optional[int] = None,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        line_thickness: int = 3,
    ):
        del line_thickness
        if im0s is None or getattr(im0s, "size", 0) == 0:
            raise DetectorError("Input image is empty.")
        if im0s.ndim != 3 or im0s.shape[2] != 3:
            raise DetectorError("Input image must be a BGR HxWx3 array.")

        # OpenCV DNN is sensitive to the static ONNX input shape. Realtime tabs may
        # request a smaller size for speed, but many exported YOLO ONNX graphs here
        # expect 640 and will fail inside Reshape if the fallback backend receives
        # a different size.
        del imgsz
        network_size = _make_divisible(int(self.imgsz or self._requested_imgsz), 32)
        blob = self._preprocess(im0s, network_size)

        with self._lock:
            try:
                self.net.setInput(blob)
                raw_outputs = self.net.forward(self.output_names) if self.output_names else self.net.forward()
            except Exception as exc:
                raise DetectorError(f"OpenCV DNN CPU inference failed: {exc}") from exc

        if isinstance(raw_outputs, np.ndarray):
            outputs = [raw_outputs]
        else:
            outputs = [np.asarray(output) for output in raw_outputs]

        results = postprocess_yolo(
            outputs=outputs,
            conf_thres=float(conf_thres),
            iou_thres=float(iou_thres),
            orig_shape=tuple(int(v) for v in im0s.shape[:2]),
            network_shape=(network_size, network_size),
            scale=self._scale,
            pad=self._pad,
            output_format=self.output_format,
        )
        return results, im0s.copy() if DEFAULT_COPY_FRAME_OUTPUT else im0s

    def close(self) -> None:
        self.net = None


class v5detect:
    """Backend selector: TensorRT -> ONNX Runtime CUDA -> optional CPU fallback."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        allow_cpu_fallback: bool = False,
        backend_preference: str = "auto",
    ):
        requested_model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not requested_model_path.exists():
            raise FileNotFoundError(f"Model file does not exist: {requested_model_path}")

        self.requested_model_path = requested_model_path
        self.model_path, self.onnx_path = _model_family_paths(requested_model_path)
        self.allow_cpu_fallback = allow_cpu_fallback
        self.backend_preference = self._normalize_backend_preference(backend_preference)
        self.load_notes: List[str] = []
        self.gpu_attempts: List[str] = []
        self.cpu_attempts: List[str] = []
        self.backend_name = ""
        self.device_name = ""
        self.rebuilt_engine = False
        self.engine_generated_from_onnx = False
        self._impl = None

        if self.backend_preference != "auto":
            self._impl = self._load_selected_backend(requested_model_path, self.backend_preference)
            self._finish_init()
            return

        gpu_attempts: List[str] = []

        try:
            self._impl = TensorRTDetector(model_path=str(requested_model_path))
            self.load_notes.extend(getattr(self._impl, "load_notes", []))
        except Exception as exc:
            gpu_attempts.append(f"TensorRT: {exc}")

        if self._impl is None and self.onnx_path.exists():
            try:
                self._impl = OnnxRuntimeDetector(model_path=str(self.onnx_path), provider="cuda")
                self.load_notes.append(f"Falling back to ONNX Runtime CUDA with {self.onnx_path.name}.")
            except Exception as exc:
                gpu_attempts.append(f"ONNX Runtime CUDA: {exc}")

        if self._impl is None:
            if allow_cpu_fallback:
                if not self.onnx_path.exists():
                    raise DetectorError(
                        "GPU backends are unavailable and no ONNX model was found for CPU fallback.\n"
                        + "\n".join(gpu_attempts)
                    )
                cpu_attempts: List[str] = []
                try:
                    self._impl = OnnxRuntimeDetector(model_path=str(self.onnx_path), provider="cpu")
                    self.load_notes.append(f"User accepted ONNX Runtime CPU fallback with {self.onnx_path.name}.")
                except Exception as exc:
                    cpu_attempts.append(f"ONNX Runtime CPU: {exc}")
                enable_opencv_dnn = os.getenv("YOLO_ENABLE_OPENCV_DNN_FALLBACK", "1").strip().lower() in {"1", "true", "yes", "on"}
                if self._impl is None and enable_opencv_dnn:
                    try:
                        self._impl = OpenCVDnnDetector(model_path=str(self.onnx_path))
                        self.load_notes.append(f"User accepted OpenCV DNN CPU fallback with {self.onnx_path.name}.")
                    except Exception as exc:
                        cpu_attempts.append(f"OpenCV DNN CPU: {exc}")
                elif self._impl is None:
                    cpu_attempts.append(
                        "OpenCV DNN CPU fallback is disabled by YOLO_ENABLE_OPENCV_DNN_FALLBACK=0. "
                        "Package/use ONNX Runtime CPU instead, or enable OpenCV DNN fallback."
                    )
                if self._impl is None:
                    raise DetectorError(
                        "CPU fallback failed.\n"
                        + "\n".join(gpu_attempts + cpu_attempts)
                    )
                self.cpu_attempts = list(cpu_attempts)
            else:
                if self.onnx_path.exists():
                    detail = "\n".join(gpu_attempts) if gpu_attempts else "No GPU backend could be initialized."
                    raise CpuFallbackRequiredError(
                        "当前设备未能建立 CUDA 推理链路。\n\n"
                        f"{detail}\n\n是否切换到 CPU 推理继续运行？",
                        requested_model_path=requested_model_path,
                        onnx_path=self.onnx_path,
                        gpu_attempts=gpu_attempts,
                    )
                raise DetectorError("\n".join(gpu_attempts) if gpu_attempts else "No inference backend could be initialized.")

        self.gpu_attempts = list(gpu_attempts)
        self._finish_init()

    @staticmethod
    def _normalize_backend_preference(value: str) -> str:
        value = (value or "auto").strip().lower().replace("-", "_")
        aliases = {
            "gpu": "auto",
            "tensorrt_cuda": "tensorrt",
            "trt": "tensorrt",
            "ort_cuda": "onnx_cuda",
            "cuda": "onnx_cuda",
            "ort_cpu": "onnx_cpu",
            "cpu": "onnx_cpu",
            "opencv": "opencv_cpu",
            "opencv_dnn": "opencv_cpu",
        }
        value = aliases.get(value, value)
        if value not in {"auto", "tensorrt", "onnx_cuda", "onnx_cpu", "opencv_cpu"}:
            raise DetectorError(f"Unsupported backend preference: {value}")
        return value

    def _load_selected_backend(self, requested_model_path: Path, preference: str):
        if preference == "tensorrt":
            try:
                detector = TensorRTDetector(model_path=str(requested_model_path))
                self.load_notes.extend(getattr(detector, "load_notes", []))
                self.load_notes.append("User selected TensorRT backend.")
                return detector
            except Exception as exc:
                raise DetectorError(f"Selected TensorRT backend failed.\nTensorRT: {exc}") from exc

        if not self.onnx_path.exists():
            raise DetectorError(
                f"Selected backend {preference} requires companion ONNX model: {self.onnx_path}"
            )

        if preference == "onnx_cuda":
            try:
                detector = OnnxRuntimeDetector(model_path=str(self.onnx_path), provider="cuda")
                self.load_notes.append(f"User selected ONNX Runtime CUDA with {self.onnx_path.name}.")
                return detector
            except Exception as exc:
                raise DetectorError(f"Selected ONNX Runtime CUDA backend failed.\nONNX Runtime CUDA: {exc}") from exc

        if preference == "onnx_cpu":
            try:
                detector = OnnxRuntimeDetector(model_path=str(self.onnx_path), provider="cpu")
                self.load_notes.append(f"User selected ONNX Runtime CPU with {self.onnx_path.name}.")
                return detector
            except Exception as exc:
                raise DetectorError(f"Selected ONNX Runtime CPU backend failed.\nONNX Runtime CPU: {exc}") from exc

        if preference == "opencv_cpu":
            try:
                detector = OpenCVDnnDetector(model_path=str(self.onnx_path))
                self.load_notes.append(f"User selected OpenCV DNN CPU with {self.onnx_path.name}.")
                return detector
            except Exception as exc:
                raise DetectorError(f"Selected OpenCV DNN CPU backend failed.\nOpenCV DNN CPU: {exc}") from exc

        raise DetectorError(f"Unsupported backend preference: {preference}")

    def _finish_init(self) -> None:
        self.backend_name = getattr(self._impl, "backend_name", "Unknown")
        self.device_name = getattr(self._impl, "device_name", "Unknown")
        self.rebuilt_engine = getattr(self._impl, "rebuilt_engine", False)
        self.engine_generated_from_onnx = getattr(self._impl, "engine_generated_from_onnx", False)
        self.model_path = getattr(self._impl, "model_path", self.model_path)

    def __getattr__(self, item):
        return getattr(self._impl, item)

    def detect(self, img: np.ndarray):
        return self._impl.detect(img)

    def run(
        self,
        im0s: np.ndarray,
        imgsz: Optional[int] = None,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        line_thickness: int = 3,
    ):
        return self._impl.run(im0s, imgsz=imgsz, conf_thres=conf_thres, iou_thres=iou_thres, line_thickness=line_thickness)

    def close(self) -> None:
        if self._impl is not None and hasattr(self._impl, "close"):
            self._impl.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GPU-first YOLO smoke test")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path to a .trt or .onnx model file")
    parser.add_argument("--image", required=True, help="Path to a test image")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="Inference size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--allow-cpu-fallback", action="store_true", help="Allow ONNX Runtime CPU fallback if CUDA backends fail")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations before timing")
    parser.add_argument("--repeat", type=int, default=1, help="Timed iterations for a quick benchmark")
    args = parser.parse_args()

    image_data = np.fromfile(args.image, dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR) if image_data.size else None
    if image is None:
        raise SystemExit(f"Failed to read test image: {args.image}")

    detector = v5detect(model_path=args.model, allow_cpu_fallback=args.allow_cpu_fallback)
    for note in detector.load_notes:
        logger.info(note)
    logger.info("backend=%s device=%s", detector.backend_name, detector.device_name)
    for _ in range(max(0, int(args.warmup))):
        detector.run(image, imgsz=args.imgsz, conf_thres=args.conf, iou_thres=args.iou)

    timings: List[float] = []
    detections = []
    for _ in range(max(1, int(args.repeat))):
        start = time.perf_counter()
        detections, _ = detector.run(image, imgsz=args.imgsz, conf_thres=args.conf, iou_thres=args.iou)
        timings.append((time.perf_counter() - start) * 1000.0)

    if timings:
        logger.info(
            "timing_ms avg=%.2f min=%.2f max=%.2f repeat=%s warmup=%s",
            float(np.mean(timings)),
            float(np.min(timings)),
            float(np.max(timings)),
            len(timings),
            max(0, int(args.warmup)),
        )
    logger.info("detections=%s", len(detections))
    for det in detections[:10]:
        logger.debug(det)
