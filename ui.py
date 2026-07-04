from utils import *
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
划痕缺陷检测系统 v6.1（检测对象选择 + 铝合金零件标准 + 距离计算 + 聚集检测）
"""

import time
import os
import csv
import sys
import math
import io
import shutil
import logging
import multiprocessing
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from queue import Empty, Full, Queue
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QPoint, QUrl, QMutex, QMutexLocker
# 全局检测器锁
detector_mutex = QMutex()
from PyQt5.QtGui import QIcon, QPixmap, QImage, QFont, QColor, QDesktopServices, QFontMetrics
from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QTabWidget, QLabel, QPushButton,
    QSlider, QFileDialog, QMessageBox, QVBoxLayout, QHBoxLayout,
    QGroupBox, QTextEdit, QStatusBar, QProgressBar,
    QGridLayout, QFrame, QComboBox, QDialog, QFormLayout, QSpinBox,
    QDoubleSpinBox, QListWidget, QListWidgetItem, QLineEdit, QCheckBox,
    QSplitter, QScrollArea, QSizePolicy, QStackedWidget
)

from detect_qt5 import CpuFallbackRequiredError, DetectorError, v5detect
import distance_calc as dc

logger = logging.getLogger(__name__)

try:
    from scipy.optimize import linear_sum_assignment
except Exception as exc:
    linear_sum_assignment = None
    logger.debug("scipy linear_sum_assignment unavailable; using greedy tracking fallback: %s", exc)

SOURCE_ROOT = Path(__file__).resolve().parent
APP_ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else SOURCE_ROOT
_MEIPASS = Path(getattr(sys, "_MEIPASS", str(SOURCE_ROOT))).resolve()
# onedir 模式下数据在 _internal 目录，兼容 _MEIPASS 指向临时目录的情况
_INTERNAL = APP_ROOT / "_internal"
BUNDLE_ROOT = _MEIPASS if (_MEIPASS / "weights").exists() else (
    _INTERNAL if _INTERNAL.exists() else APP_ROOT
)


def _find_resource(relative_path: str) -> Path:
    """在多个可能位置查找资源文件，返回第一个存在的路径。"""
    candidates = [
        BUNDLE_ROOT / relative_path,
        _MEIPASS / relative_path,
        _INTERNAL / relative_path,
        APP_ROOT / relative_path,
        SOURCE_ROOT / relative_path,
    ]
    seen = set()
    for p in candidates:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            if p.exists():
                return p
    raise FileNotFoundError(
        f"找不到资源文件: {relative_path}\n已搜索: {[str(c) for c in candidates]}"
    )


# ──────────────────────────── 全局样式（Windows 11 明亮轻快风格 + 工业美）────────────────────────
with open(str(_find_resource("theme.qss")), "r", encoding="utf-8") as f:
    LIGHT_STYLE = f.read()

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def find_existing_path(*candidates):
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return ""


def open_local_path(path: str):
    if not path:
        return False
    return QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(path)))


class UnavailableDetector:
    def __init__(self, reason="视觉分析引擎尚未加载"):
        self.reason = reason

    def run(self, *args, **kwargs):
        raise DetectorError(self.reason)

    def detect(self, *args, **kwargs):
        raise DetectorError(self.reason)

    def close(self):
        return None


MODEL_LOAD_MIN_VISIBLE_MS_GPU = 1600
MODEL_LOAD_MIN_VISIBLE_MS_CPU = 900

BACKEND_CHOICES = [
    ("自动优先", "auto"),
    ("TensorRT", "tensorrt"),
    ("ONNX CUDA", "onnx_cuda"),
    ("ONNX CPU", "onnx_cpu"),
    ("OpenCV CPU", "opencv_cpu"),
]
BACKEND_LABELS = {value: label for label, value in BACKEND_CHOICES}


class ModelLoadWorker(QThread):
    sig_loaded = pyqtSignal(str, object)
    sig_failed = pyqtSignal(str, object)

    def __init__(
        self,
        path: str,
        allow_cpu_fallback: bool = False,
        backend_preference: str = "auto",
        min_visible_ms: int = 0,
        parent=None,
    ):
        super().__init__(parent)
        self.path = path
        self.allow_cpu_fallback = allow_cpu_fallback
        self.backend_preference = backend_preference
        self.min_visible_ms = max(0, int(min_visible_ms))

    def _keep_loading_visible(self, started_at: float):
        remaining = self.min_visible_ms / 1000.0 - (time.perf_counter() - started_at)
        if remaining > 0:
            time.sleep(remaining)

    def run(self):
        started_at = time.perf_counter()
        try:
            detector = v5detect(
                model_path=self.path,
                allow_cpu_fallback=self.allow_cpu_fallback,
                backend_preference=self.backend_preference,
            )
            self._keep_loading_visible(started_at)
            self.sig_loaded.emit(self.path, detector)
        except BaseException as exc:
            self._keep_loading_visible(started_at)
            self.sig_failed.emit(self.path, exc)


def ensure_detector_available(parent, detector, title="模型未就绪"):
    if detector is None or isinstance(detector, UnavailableDetector):
        message = getattr(detector, "reason", "尚未加载可用的 TensorRT 引擎")
        QMessageBox.warning(parent, title, message)
        return False
    return True

# ──────────────────────────── 安全版核心判定逻辑（2mm致命缺陷 + 防闪退）────────────────────────
def evaluate_defects_detailed(dets, pixel_per_cm, standard, min_defect_mm=0.2, image_width=None, ref_width=None, detect_height=None):
    """
    评估缺陷详情
    image_width: 实际图像宽度，用于动态计算像素当量
    ref_width: 保留兼容字段
    detect_height: 画面横向实际视野宽度(cm)，保留旧变量名以兼容配置
    """
    # 如果提供了实际图像宽度和视野宽度，根据当前输入帧动态计算像素当量
    calc_log = ""
    if image_width is not None and ref_width is not None and detect_height is not None and detect_height > 0:
        actual_pixel_per_cm = image_width / detect_height
        calc_log = f" [动态换算: 图像宽{image_width}px/视野宽{detect_height}cm={actual_pixel_per_cm:.2f}像素/cm]"
        mm_per_pixel = 10.0 / actual_pixel_per_cm
    else:
        mm_per_pixel = 10.0 / pixel_per_cm

    oversized_count = 0
    defect_details = []
    force_fail = False
    lethal_count = 0  # 致命缺陷计数（超过 lethal_mm 阈值）

    for det in dets:
        try:
            if len(det) < 4:
                continue
            x1 = float(det[0])
            y1 = float(det[1])
            x2 = float(det[2])
            y2 = float(det[3])

            w_px = abs(x2 - x1)
            h_px = abs(y2 - y1)
            w_mm = w_px * mm_per_pixel
            h_mm = h_px * mm_per_pixel

            # 使用 dynamic_px_cm 统一像素/cm标定
            dynamic_px_cm = image_width / detect_height if (image_width and detect_height) else pixel_per_cm
            size_info = dc.calculate_defect_size((x1, y1, x2, y2), dynamic_px_cm)
            equiv_diameter = size_info['equiv_diameter_mm']

            if equiv_diameter < min_defect_mm:
                continue

            # 计算缺陷中心点距离
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            defect_details.append({
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'center': (cx, cy),
                'w_px': w_px,
                'h_px': h_px,
                'w_mm': w_mm,
                'h_mm': h_mm,
                'diameter': equiv_diameter
            })

            # 致命缺陷判定（超过 lethal_mm 阈值直接判定为不合格）
            if equiv_diameter > standard.lethal_mm:
                lethal_count += 1
                force_fail = True

            if equiv_diameter > standard.size_threshold_mm:
                oversized_count += 1

        except (TypeError, ValueError, KeyError, IndexError) as e:
            logger.debug("Defect size calculation skipped: %s", e)
            continue

    # 统一依据标称的 GB 标准进行合格判定（超标数量不超过限制，且无致命缺陷）
    is_passed = (oversized_count <= standard.max_defects) and not force_fail

    return oversized_count, is_passed, defect_details, lethal_count, calc_log


def calculate_defect_distance(d1, d2, pixel_per_cm):
    """计算两个缺陷中心点之间的距离(mm)"""
    cx1, cy1 = d1['center']
    cx2, cy2 = d2['center']
    dist_px = math.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
    dist_mm = dist_px * 10.0 / pixel_per_cm
    return dist_mm


def check_defect_clustering(defect_details, pixel_per_cm, cluster_threshold_mm=10.0):
    """检测缺陷聚集情况，返回聚集簇信息"""
    clusters = []
    used = set()

    for i, d1 in enumerate(defect_details):
        if i in used:
            continue
        cluster = [i]
        used.add(i)

        for j, d2 in enumerate(defect_details):
            if j in used:
                continue
            dist = calculate_defect_distance(d1, d2, pixel_per_cm)
            if dist < cluster_threshold_mm:
                cluster.append(j)
                used.add(j)

        if len(cluster) > 1:
            clusters.append(cluster)

    return clusters


def draw_defect_labels(image, defect_details, standard, results=None):
    """绘制缺陷标签和置信度
    results: 原始检测结果列表，每个元素为 [x1, y1, x2, y2, conf, ...]
    """
    if image is None:
        return image
    img = image.copy()
    lethal_mm = getattr(standard, 'lethal_mm', 2.0)

    # 如果没有defect_details但有results，至少绘制检测框
    if not defect_details and results:
        for det in results:
            if len(det) >= 5:
                x1, y1, x2, y2, conf = [float(v) for v in det[:5]]
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # 绘制置信度 - 醒目的黄色黑边气泡框
                conf_text = f"{conf:.2f}"
                font_scale = 0.8  # 放大一倍
                thickness = 2
                (tw, th), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                # 背景框 - 高对比度黄色
                cv2.rectangle(img, (int(x1), int(y1) - th - 10), (int(x1) + tw + 8, int(y1)), (0, 255, 255), -1)
                # 黑色边框
                cv2.rectangle(img, (int(x1), int(y1) - th - 10), (int(x1) + tw + 8, int(y1)), (0, 0, 0), 2)
                # 文字
                cv2.putText(img, conf_text, (int(x1) + 2, int(y1) - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        return img

    if len(defect_details) == 0:
        return img

    for i, d in enumerate(defect_details):
        x1, y1, x2, y2 = d['bbox']
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        # 获取置信度
        conf = 0.0
        if results and i < len(results) and len(results[i]) >= 5:
            conf = float(results[i][4])

        # 绘制直径标签
        text = f"{d['diameter']:.1f}mm"
        color = (0, 0, 255) if d['diameter'] > lethal_mm else (0, 255, 255)
        font_scale = 0.5
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(img, (int(x1), int(y1) - th - 4), (int(x1) + tw + 4, int(y1)), color, -1)
        cv2.putText(img, text, (int(x1) + 2, int(y1) - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)

        # 绘制置信度 - 醒目的黄色黑边气泡框（在直径后面）
        if conf > 0:
            conf_text = f"{conf:.2f}"
            font_scale_conf = 0.8  # 放大一倍
            thickness_conf = 2
            (cw, ch), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale_conf, thickness_conf)
            # 在直径标签右侧绘制置信度
            conf_x = int(x1) + tw + 10
            conf_y_top = int(y1) - ch - 6
            conf_y_bottom = int(y1)
            # 黄色高亮背景
            cv2.rectangle(img, (conf_x, conf_y_top), (conf_x + cw + 6, conf_y_bottom), (0, 255, 255), -1)
            # 黑色边框
            cv2.rectangle(img, (conf_x, conf_y_top), (conf_x + cw + 6, conf_y_bottom), (0, 0, 0), 2)
            # 置信度文字
            cv2.putText(img, conf_text, (conf_x + 2, int(y1) - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale_conf, (0, 0, 0), thickness_conf)

    return img


def default_filter_settings():
    return {
        "roi_enabled": True,
        "roi_margin_pct": 10.0,
        "roi_rule": "center",
        "stable_enabled": False,
        "stable_frames": 20,
        "track_iou": 0.30,
        "count_enabled": False,
        "count_orientation": "vertical",
        "count_line_pct": 50.0,
        "count_min_displacement": 30.0,
        "count_min_interval_ms": 800,
        "speed_enabled": False,
        "line_speed": 1.0,
        "base_speed": 1.0,
    }


def merge_filter_settings(settings=None):
    merged = default_filter_settings()
    if settings:
        merged.update(settings)
    return merged


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def bbox_center(det):
    x1, y1, x2, y2 = [float(v) for v in det[:4]]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def draw_filter_overlay(image, filter_info):
    if image is None or not filter_info:
        return image
    img = image.copy()
    rect = filter_info.get("roi_rect")
    if rect:
        x1, y1, x2, y2 = [int(v) for v in rect]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 180, 0), 2)
        cv2.putText(img, "ROI", (x1 + 8, max(24, y1 + 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 0), 2)
    line = filter_info.get("count_line")
    if line:
        orientation, value = line
        h, w = img.shape[:2]
        if orientation == "vertical":
            x = int(value)
            cv2.line(img, (x, 0), (x, h), (255, 140, 0), 2)
            cv2.putText(img, "COUNT", (min(w - 120, x + 8), 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 140, 0), 2)
        else:
            y = int(value)
            cv2.line(img, (0, y), (w, y), (255, 140, 0), 2)
            cv2.putText(img, "COUNT", (12, max(28, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 140, 0), 2)
    return img


def realtime_timing_text(payload):
    if os.getenv("UI_SHOW_PIPELINE_TIMING", "1").strip().lower() not in {"1", "true", "yes", "on"}:
        return ""
    timing = payload.get("timing") or {}
    if not timing:
        return ""
    parts = [
        f"总{timing.get('total', 0.0):.0f}ms",
        f"推理{timing.get('infer', 0.0):.0f}ms",
        f"后处理{(timing.get('filter', 0.0) + timing.get('eval', 0.0)):.0f}ms",
    ]
    fps_parts = []
    if timing.get("capture_fps", 0.0) > 0:
        fps_parts.append(f"采集{timing.get('capture_fps', 0.0):.1f}")
    if timing.get("calc_fps", 0.0) > 0:
        fps_parts.append(f"计算{timing.get('calc_fps', 0.0):.1f}")
    if timing.get("worker_fps", 0.0) > 0:
        fps_parts.append(f"输出{timing.get('worker_fps', 0.0):.1f}")
    if fps_parts:
        parts.append("FPS " + "/".join(fps_parts))
    if timing.get("render", 0.0) > 1.0:
        parts.append(f"绘制{timing.get('render', 0.0):.0f}ms")
    backend_profile = payload.get("backend_profile") or {}
    if backend_profile:
        parts.append(f"TRT{backend_profile.get('total', 0.0):.0f}ms")
    return " | " + " / ".join(parts)


def realtime_perf_line(payload):
    timing = payload.get("timing") or {}
    if not timing:
        return ""
    capture = timing.get("capture_fps", 0.0)
    calc = timing.get("calc_fps", 0.0)
    worker = timing.get("worker_fps", 0.0)
    total = timing.get("total", 0.0)
    infer = timing.get("infer", 0.0)
    return (
        f"FPS 采集 {capture:.1f} | 计算 {calc:.1f} | 输出 {worker:.1f}    "
        f"耗时 总 {total:.0f}ms | 推理 {infer:.0f}ms"
    )


class ProductionFilterEngine:
    def __init__(self, settings=None):
        self.settings = merge_filter_settings(settings)
        self._tracks = {}
        self._frame_idx = {}
        self._next_track_id = 1
        self.total_count = 0

    def update_settings(self, settings):
        self.settings = merge_filter_settings(settings)
        self.reset()

    def reset(self, source=None):
        if source is None:
            self._tracks.clear()
            self._frame_idx.clear()
            self.total_count = 0
        else:
            self._tracks.pop(source, None)
            self._frame_idx.pop(source, None)

    def roi_rect(self, image_shape):
        h, w = image_shape[:2]
        margin = max(0.0, min(45.0, float(self.settings.get("roi_margin_pct", 10.0)))) / 100.0
        return (
            int(w * margin),
            int(h * margin),
            int(w * (1.0 - margin)),
            int(h * (1.0 - margin)),
        )

    def _inside_roi(self, det, rect):
        x1, y1, x2, y2 = [float(v) for v in det[:4]]
        rx1, ry1, rx2, ry2 = rect
        if self.settings.get("roi_rule") == "corners":
            points = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
            return all(rx1 <= x <= rx2 and ry1 <= y <= ry2 for x, y in points)
        cx, cy = bbox_center(det)
        return rx1 <= cx <= rx2 and ry1 <= cy <= ry2

    def _effective_stable_frames(self):
        base_frames = max(1, int(self.settings.get("stable_frames", 20)))
        if not self.settings.get("speed_enabled", False):
            return base_frames
        speed = max(0.1, float(self.settings.get("line_speed", 1.0)))
        base_speed = max(0.1, float(self.settings.get("base_speed", 1.0)))
        return max(1, min(base_frames * 2, int(round(base_frames * base_speed / speed))))

    def _effective_interval_ms(self):
        interval = max(0, int(self.settings.get("count_min_interval_ms", 800)))
        if not self.settings.get("speed_enabled", False):
            return interval
        speed = max(0.1, float(self.settings.get("line_speed", 1.0)))
        base_speed = max(0.1, float(self.settings.get("base_speed", 1.0)))
        return max(120, int(round(interval * base_speed / speed)))

    def _match_distance(self):
        speed_factor = 1.0
        if self.settings.get("speed_enabled", False):
            speed = max(0.1, float(self.settings.get("line_speed", 1.0)))
            base_speed = max(0.1, float(self.settings.get("base_speed", 1.0)))
            speed_factor = max(1.0, min(3.0, speed / base_speed))
        return 80.0 * speed_factor

    def _track(self, results, image_shape, source):
        frame_idx = self._frame_idx.get(source, 0) + 1
        self._frame_idx[source] = frame_idx
        tracks = self._tracks.setdefault(source, [])
        for tr in tracks:
            tr["matched"] = False

        matched = []
        max_dist = self._match_distance()
        min_iou = float(self.settings.get("track_iou", 0.30))

        assignments = {}
        if linear_sum_assignment is not None and results and tracks:
            try:
                cost = np.full((len(results), len(tracks)), 1e6, dtype=np.float32)
                valid = np.zeros((len(results), len(tracks)), dtype=bool)
                for det_idx, det in enumerate(results):
                    cx, cy = bbox_center(det)
                    for track_idx, tr in enumerate(tracks):
                        iou = bbox_iou(det, tr["bbox"])
                        tx, ty = tr["center"]
                        dist = math.hypot(cx - tx, cy - ty)
                        if iou >= min_iou or dist <= max_dist:
                            score = iou + max(0.0, 1.0 - dist / max_dist)
                            cost[det_idx, track_idx] = -float(score)
                            valid[det_idx, track_idx] = True
                row_idx, col_idx = linear_sum_assignment(cost)
                for det_idx, track_idx in zip(row_idx, col_idx):
                    if valid[det_idx, track_idx]:
                        assignments[int(det_idx)] = tracks[int(track_idx)]
            except Exception:
                logger.exception("Hungarian tracking failed; falling back to greedy matching")
                assignments = {}

        for det_idx, det in enumerate(results):
            cx, cy = bbox_center(det)
            best = assignments.get(det_idx)
            if best is None:
                best_score = -1.0
                for tr in tracks:
                    if tr.get("matched"):
                        continue
                    iou = bbox_iou(det, tr["bbox"])
                    tx, ty = tr["center"]
                    dist = math.hypot(cx - tx, cy - ty)
                    if iou >= min_iou or dist <= max_dist:
                        score = iou + max(0.0, 1.0 - dist / max_dist)
                        if score > best_score:
                            best_score = score
                            best = tr
            if best is None:
                best = {
                    "id": self._next_track_id,
                    "hits": 0,
                    "bbox": det,
                    "center": (cx, cy),
                    "prev_center": None,
                    "last_seen": frame_idx,
                    "last_count_time": 0.0,
                    "counted": False,
                }
                self._next_track_id += 1
                tracks.append(best)
            prev_center = best.get("center")
            best["prev_center"] = prev_center
            best["center"] = (cx, cy)
            best["bbox"] = det
            best["hits"] = int(best.get("hits", 0)) + 1
            best["last_seen"] = frame_idx
            best["matched"] = True
            matched.append((det, best))

        keep_after = max(4, self._effective_stable_frames() * 2)
        self._tracks[source] = [tr for tr in tracks if frame_idx - tr.get("last_seen", 0) <= keep_after]
        return matched

    def _count_crossings(self, matched, image_shape):
        if not self.settings.get("count_enabled", False):
            return 0
        h, w = image_shape[:2]
        orientation = self.settings.get("count_orientation", "vertical")
        line_pct = max(5.0, min(95.0, float(self.settings.get("count_line_pct", 50.0)))) / 100.0
        line_value = w * line_pct if orientation == "vertical" else h * line_pct
        min_move = max(0.0, float(self.settings.get("count_min_displacement", 30.0)))
        interval = self._effective_interval_ms() / 1000.0
        now = time.time()
        events = 0
        for _, tr in matched:
            if tr.get("counted"):
                continue
            prev = tr.get("prev_center")
            cur = tr.get("center")
            if not prev or not cur:
                continue
            prev_value = prev[0] if orientation == "vertical" else prev[1]
            cur_value = cur[0] if orientation == "vertical" else cur[1]
            crossed = (prev_value - line_value) * (cur_value - line_value) <= 0 and prev_value != cur_value
            moved = math.hypot(cur[0] - prev[0], cur[1] - prev[1]) >= min_move
            enough_time = now - float(tr.get("last_count_time", 0.0)) >= interval
            if crossed and moved and enough_time:
                tr["counted"] = True
                tr["last_count_time"] = now
                events += 1
        self.total_count += events
        return events

    def filter(self, results, image_shape, source="image", temporal=True):
        results = list(results or [])
        info = {
            "raw_count": len(results),
            "roi_removed": 0,
            "unstable_removed": 0,
            "count_events": 0,
            "total_count": self.total_count,
            "effective_stable_frames": self._effective_stable_frames(),
            "roi_rect": None,
            "count_line": None,
        }

        filtered = results
        if self.settings.get("roi_enabled", False):
            rect = self.roi_rect(image_shape)
            info["roi_rect"] = rect
            filtered = [det for det in filtered if self._inside_roi(det, rect)]
            info["roi_removed"] = len(results) - len(filtered)
        if self.settings.get("count_enabled", False):
            h, w = image_shape[:2]
            orientation = self.settings.get("count_orientation", "vertical")
            pct = max(5.0, min(95.0, float(self.settings.get("count_line_pct", 50.0)))) / 100.0
            info["count_line"] = (orientation, w * pct if orientation == "vertical" else h * pct)

        needs_tracking = temporal and (
            self.settings.get("stable_enabled", False) or self.settings.get("count_enabled", False)
        )
        if not needs_tracking:
            info["final_count"] = len(filtered)
            info["total_count"] = self.total_count
            return filtered, info

        matched = self._track(filtered, image_shape, source)
        info["count_events"] = self._count_crossings(matched, image_shape)
        if self.settings.get("stable_enabled", False):
            stable_frames = self._effective_stable_frames()
            stable = [det for det, tr in matched if int(tr.get("hits", 0)) >= stable_frames]
            info["unstable_removed"] = len(filtered) - len(stable)
            filtered = stable
        info["final_count"] = len(filtered)
        info["total_count"] = self.total_count
        return filtered, info


# ──────────────────────────── 自定义标题栏 ─────────────────────────
class TitleBar(QWidget):

    def __init__(self, parent: QMainWindow):
        super().__init__(parent)
        self._win = parent
        self._drag_pos = None
        self.setFixedHeight(60)
        self.setStyleSheet("TitleBar { background: #ffffff; border-bottom: 3px solid #c0c0c0; border-bottom-left-radius: 0px; border-bottom-right-radius: 0px; }")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 0, 12, 0)
        lay.setSpacing(6)

        self.title_lbl = QLabel("划痕缺陷检测系统 v6.1")
        self.title_lbl.setStyleSheet("""
            color:#0078d4;
            font-size: 26px;
            font-weight: bold;
            background: transparent;
            letter-spacing: 2px;
            padding: 4px 10px;
            border-radius: 8px;
        """)
        lay.addWidget(self.title_lbl)
        lay.addStretch()

        self.time_lbl = QLabel()
        self.time_lbl.setStyleSheet("color:#7a7a7a; font-size:12px; background:transparent;")
        lay.addWidget(self.time_lbl)
        self._clock = QTimer(self)
        self._clock.timeout.connect(self._tick_clock)
        self._clock.start(1000)
        self._tick_clock()

        lay.addSpacing(12)

        self.btn_min = QPushButton("─")
        self.btn_min.setObjectName("titlebar_btn")
        self.btn_min.clicked.connect(parent.showMinimized)

        self.btn_restore = QPushButton("❐")
        self.btn_restore.setObjectName("titlebar_btn")
        self.btn_restore.clicked.connect(self._toggle_fullscreen)

        self.btn_close = QPushButton("✕")
        self.btn_close.setObjectName("titlebar_btn_close")
        self.btn_close.clicked.connect(parent.close)

        for b in (self.btn_min, self.btn_restore, self.btn_close):
            lay.addWidget(b)

    def _tick_clock(self):
        self.time_lbl.setText(datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))

    def _toggle_fullscreen(self):
        if self._win.isFullScreen():
            self._win.showNormal()
            self.btn_restore.setText("❐")
        else:
            self._win.showFullScreen()
            self.btn_restore.setText("⊡")

    def mouseDoubleClickEvent(self, e):
        self._toggle_fullscreen()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag_pos = e.globalPos() - self._win.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if e.buttons() == Qt.LeftButton and self._drag_pos:
            if self._win.isFullScreen():
                self._win.showNormal()
                self.btn_restore.setText("❐")
            self._win.move(e.globalPos() - self._drag_pos)

    def mouseReleaseEvent(self, e):
        self._drag_pos = None


# ──────────────────────────── 检测标准配置数据 ─────────────────────────
class InspectionStandard:
    def __init__(self, name="GB 11614-2022 平板玻璃", max_defects=3, size_threshold_mm=5.0, min_defect_mm=0.2,
                 detect_object="scratch", lethal_mm=2.0):
        self.name = name
        self.max_defects = max_defects
        self.size_threshold_mm = size_threshold_mm
        self.min_defect_mm = min_defect_mm
        self.detect_object = detect_object  # "scratch" 或 "aluminum_alloy"
        self.lethal_mm = lethal_mm  # 致命缺陷阈值(mm)

    def to_dict(self):
        return {"name": self.name, "max_defects": self.max_defects,
                "size_threshold_mm": self.size_threshold_mm, "min_defect_mm": self.min_defect_mm,
                "detect_object": self.detect_object, "lethal_mm": self.lethal_mm}

    @staticmethod
    def from_dict(d):
        return InspectionStandard(d["name"], d["max_defects"],
                                  d["size_threshold_mm"], d.get("min_defect_mm", 0.2),
                                  d.get("detect_object", "scratch"), d.get("lethal_mm", 2.0))


class StandardConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("检测标准配置")
        self.setMinimumWidth(400)
        self.setMinimumHeight(350)
        self.setModal(True)
        self.standards = []
        self.load_defaults()
        self.init_ui()

    def load_defaults(self):
        # 屏幕划痕标准 (GB平板玻璃)
        self.standards = [
            InspectionStandard("GB 11614-2022 平板玻璃", max_defects=3, size_threshold_mm=2.0,
                             min_defect_mm=0.2, detect_object="scratch", lethal_mm=5.0),
            # 精密铝合金零件标准 (参考GB/T 6892-2023)
            InspectionStandard("GB/T 6892-2023 铝合金零件", max_defects=2, size_threshold_mm=1.0,
                             min_defect_mm=0.2, detect_object="aluminum_alloy", lethal_mm=2.0)
        ]

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.SingleSelection)
        lbl = QLabel("检测标准列表（最多3个）：")
        lbl.setWordWrap(True)
        layout.addWidget(lbl)
        layout.addWidget(self.list_widget)

        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("添加")
        self.btn_edit = QPushButton("编辑")
        self.btn_delete = QPushButton("删除")
        self.btn_add.clicked.connect(self.add_standard)
        self.btn_edit.clicked.connect(self.edit_standard)
        self.btn_delete.clicked.connect(self.delete_standard)
        for b in (self.btn_add, self.btn_edit, self.btn_delete):
            btn_layout.addWidget(b)
        layout.addLayout(btn_layout)

        group = QGroupBox("检测装置设置")
        group.setStyleSheet("QGroupBox { border: 2px solid #c0c0c0; border-radius: 16px; margin-top: 28px; padding-top: 24px; background-color: #ffffff; } QGroupBox::title { subcontrol-origin: padding; subcontrol-position: top left; padding: 0 12px; margin-top: -14px; color: #0078d4; font-weight: bold; font-size: 18px; background-color: #ffffff; }")
        form = QFormLayout(group)
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(1.0, 200.0)
        self.height_spin.setValue(19.2)
        self.height_spin.setSuffix(" cm")
        self.pixel_per_cm_spin = QDoubleSpinBox()
        self.pixel_per_cm_spin.setRange(1.0, 1000.0)
        self.pixel_per_cm_spin.setValue(100.0)
        self.pixel_per_cm_spin.setSuffix(" 像素/cm")
        self.height_spin.valueChanged.connect(self.on_height_changed)

        self.auto_pixel_cb = QCheckBox("自动判断像素当量（按当前图像宽度动态换算）")
        self.auto_pixel_cb.setChecked(True)
        self.auto_pixel_cb.toggled.connect(self.on_auto_pixel_toggled)

        form.addRow("画面实际宽度:", self.height_spin)
        form.addRow("像素当量:", self.pixel_per_cm_spin)
        form.addRow("", self.auto_pixel_cb)
        layout.addWidget(group)

        info = QLabel("注：这里填写整张输入画面从左到右覆盖的实物宽度。\n自动模式下会按 当前图像宽度/画面实际宽度 重新计算像素当量。")
        info.setStyleSheet("color:#7a7a7a; font-size:13px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        btn_box = QHBoxLayout()
        self.btn_ok = QPushButton("确定")
        self.btn_cancel = QPushButton("取消")
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        btn_box.addStretch()
        btn_box.addWidget(self.btn_ok)
        btn_box.addWidget(self.btn_cancel)
        layout.addLayout(btn_box)

        self.refresh_list()
        self.on_auto_pixel_toggled(True)

    def on_auto_pixel_toggled(self, checked):
        self.pixel_per_cm_spin.setEnabled(not checked)
        if not checked:
            # 取消自动时，恢复正常样式
            self.pixel_per_cm_spin.setStyleSheet("""
                QDoubleSpinBox {
                    background: #ffffff;
                    color: #1a1a1a;
                    border: 2px solid #c0c0c0;
                    border-radius: 8px;
                    padding: 8px;
                    font-size: 17px;
                }
            """)
        else:
            # 自动模式时，灰色不可编辑样式
            self.pixel_per_cm_spin.setStyleSheet("""
                QDoubleSpinBox {
                    background: #f0f2f5;
                    color: #7a7a7a;
                    border: 2px solid #c0c0c0;
                    border-radius: 8px;
                    padding: 8px;
                    font-size: 17px;
                }
            """)
            self.on_height_changed()

    def on_height_changed(self):
        if self.auto_pixel_cb.isChecked():
            h = self.height_spin.value()
            if h > 0:
                # 配置界面用1920宽度给出参考像素当量；实际检测时会按当前帧宽度重新计算。
                self.pixel_per_cm_spin.setValue(1920.0 / h)

    def refresh_list(self):
        self.list_widget.clear()
        for std in self.standards:
            item = QListWidgetItem(f"{std.name}  (超标数≤{std.max_defects}, 阈值{std.size_threshold_mm:.1f}mm, 最小过滤{std.min_defect_mm:.1f}mm)")
            item.setData(Qt.UserRole, std)
            self.list_widget.addItem(item)
        self.btn_edit.setEnabled(len(self.standards) > 0)
        self.btn_delete.setEnabled(len(self.standards) > 0)
        self.btn_add.setEnabled(len(self.standards) < 3)

    def add_standard(self):
        if len(self.standards) >= 3:
            QMessageBox.warning(self, "限制", "最多只能添加3个标准")
            return
        dlg = StandardEditDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            name, max_def, size_th, min_def = dlg.get_data()
            self.standards.append(InspectionStandard(name, max_def, size_th, min_def))
            self.refresh_list()

    def edit_standard(self):
        cur = self.list_widget.currentItem()
        if not cur: return
        std = cur.data(Qt.UserRole)
        dlg = StandardEditDialog(self, std.name, std.max_defects, std.size_threshold_mm, std.min_defect_mm)
        if dlg.exec_() == QDialog.Accepted:
            name, max_def, size_th, min_def = dlg.get_data()
            std.name = name
            std.max_defects = max_def
            std.size_threshold_mm = size_th
            std.min_defect_mm = min_def
            self.refresh_list()

    def delete_standard(self):
        cur = self.list_widget.currentItem()
        if not cur: return
        if len(self.standards) <= 1:
            QMessageBox.warning(self, "警告", "至少保留一个标准")
            return
        self.standards.remove(cur.data(Qt.UserRole))
        self.refresh_list()

    def get_config(self):
        selected_row = self.list_widget.currentRow()
        if selected_row < 0 and self.standards:
            selected_row = 0
        return (self.standards, selected_row, self.pixel_per_cm_spin.value(),
                self.height_spin.value())

    def set_config(self, standards, current_idx, pixel_per_cm, detect_height=19.2):
        self.standards = standards[:]
        self.pixel_per_cm_spin.setValue(pixel_per_cm)
        self.height_spin.setValue(detect_height)
        self.refresh_list()
        if 0 <= current_idx < self.list_widget.count():
            self.list_widget.setCurrentRow(current_idx)


class StandardEditDialog(QDialog):
    def __init__(self, parent=None, name="", max_defects=3, size_threshold=5.0, min_defect=0.2):
        super().__init__(parent)
        self.setWindowTitle("编辑标准")
        self.setMinimumWidth(300)
        layout = QFormLayout(self)
        self.name_edit = QLineEdit(name)
        self.max_defects_spin = QSpinBox()
        self.max_defects_spin.setRange(0, 999)
        self.max_defects_spin.setValue(max_defects)
        self.size_threshold_spin = QDoubleSpinBox()
        self.size_threshold_spin.setRange(0.1, 1000.0)
        self.size_threshold_spin.setValue(size_threshold)
        self.size_threshold_spin.setSuffix(" mm")
        self.min_defect_spin = QDoubleSpinBox()
        self.min_defect_spin.setRange(0.0, 10.0)
        self.min_defect_spin.setValue(min_defect)
        self.min_defect_spin.setSuffix(" mm")
        layout.addRow("标准名称:", self.name_edit)
        layout.addRow("允许超标缺陷数量:", self.max_defects_spin)
        layout.addRow("缺陷大小阈值:", self.size_threshold_spin)
        layout.addRow("最小缺陷过滤:", self.min_defect_spin)
        btn_box = QHBoxLayout()
        ok_btn = QPushButton("确定")
        cancel_btn = QPushButton("取消")
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        btn_box.addWidget(ok_btn)
        btn_box.addWidget(cancel_btn)
        layout.addRow(btn_box)

    def get_data(self):
        return (self.name_edit.text().strip(), self.max_defects_spin.value(),
                self.size_threshold_spin.value(), self.min_defect_spin.value())


# ──────────────────────────── 工具函数 ────────────────────────────
def apply_preprocess(img: np.ndarray, grayscale: bool, exposure: float) -> np.ndarray:
    out = img.copy()
    if grayscale:
        g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        out = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    if abs(exposure - 1.0) > 0.005:
        out = np.clip(out.astype(np.float32) * exposure, 0, 255).astype(np.uint8)
    return out


def cv2_to_qpixmap(img: np.ndarray, target_w=None, target_h=None) -> QPixmap:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if target_w and target_h:
        scale = min(target_w / w, target_h / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        qi = QImage(rgb.data, nw, nh, 3 * nw, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qi)
    qi = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qi)


def read_image_unicode(path: str, retries: int = 3, delay: float = 0.08) -> tuple[Optional[np.ndarray], Optional[str]]:
    """Read images robustly from Windows paths, including removable drives and CJK paths."""
    path_obj = Path(path)
    for attempt in range(max(1, retries)):
        try:
            if not path_obj.exists():
                return None, f"文件不存在: {path}"
            size = path_obj.stat().st_size
            if size <= 0:
                return None, f"文件大小为 0: {path}"
            data = np.fromfile(str(path_obj), dtype=np.uint8)
            if data.size == 0:
                return None, f"文件读取为空: {path}"
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is not None:
                return image, None
            last_error = f"OpenCV 解码失败: {path} ({size} bytes)"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt + 1 < retries:
            time.sleep(delay)
    return None, last_error


def write_image_unicode(path: str, image: np.ndarray) -> bool:
    path_obj = Path(path)
    ext = path_obj.suffix or ".jpg"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return False
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path_obj))
    return True


def read_image_unicode(path: str, retries: int = 3, delay: float = 0.08) -> tuple[Optional[np.ndarray], Optional[str]]:
    path_obj = Path(path)
    last_error = ""
    for attempt in range(max(1, retries)):
        try:
            if not path_obj.exists():
                return None, f"File does not exist: {path}"
            size = path_obj.stat().st_size
            if size <= 0:
                return None, f"File size is 0: {path}"
            data = np.fromfile(str(path_obj), dtype=np.uint8)
            if data.size == 0:
                return None, f"File read returned 0 bytes: {path}"
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is not None:
                return image, None

            qimg = QImage(str(path_obj))
            if not qimg.isNull():
                qimg = qimg.convertToFormat(QImage.Format_RGB888)
                width, height = qimg.width(), qimg.height()
                bytes_per_line = qimg.bytesPerLine()
                ptr = qimg.bits()
                ptr.setsize(qimg.byteCount())
                arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, bytes_per_line))
                rgb = arr[:, : width * 3].reshape((height, width, 3)).copy()
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), None

            last_error = f"OpenCV and Qt both failed to decode: {path} ({size} bytes)"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt + 1 < retries:
            time.sleep(delay)
    return None, last_error


def write_image_unicode(path: str, image: np.ndarray) -> bool:
    path_obj = Path(path)
    ext = path_obj.suffix or ".jpg"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return False
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path_obj))
    return True


class AspectRatioDisplay(QtWidgets.QWidget):
    """Visible image frame that always stays at a 16:9 ratio."""

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self._placeholder_text = text
        self._source_pixmap = QPixmap()
        self._max_height = int(os.getenv("UI_DISPLAY_MAX_HEIGHT", str(self._default_max_height())))
        self._frame = QLabel(text, self)
        self._frame.setAlignment(Qt.AlignCenter)
        self._frame.setScaledContents(False)
        self._frame.setObjectName("display_label")
        self._frame.setStyleSheet("""
            color: #9a9a9a;
            font-size: 24px;
            font-weight: bold;
            border: 2px solid #d0d0d0;
            border-radius: 8px;
            background: #fafafa;
            padding: 0;
        """)
        policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.setMinimumSize(400, 225)
        self.setMaximumHeight(self._max_height)

    @staticmethod
    def _default_max_height():
        screen = QApplication.primaryScreen()
        if screen is None:
            return 420
        available_h = screen.availableGeometry().height()
        if available_h <= 950:
            return 300
        if available_h <= 1100:
            return 330
        if available_h <= 1250:
            return 420
        return 520

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return min(self._max_height, max(225, int(width * 9 / 16)))

    def setText(self, text: str):
        self._placeholder_text = text
        if self._source_pixmap.isNull():
            self._frame.setText(text)

    def setPixmap(self, pixmap: QPixmap):
        self._source_pixmap = QPixmap(pixmap)
        self._apply_ratio_height()
        self._refresh_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_ratio_height()
        self._update_frame_geometry()
        self._refresh_pixmap()

    def _apply_ratio_height(self):
        if self.width() <= 0:
            return
        desired = self.heightForWidth(self.width())
        parent_height = self.parentWidget().height() if self.parentWidget() else 0
        if parent_height > 0:
            screen = QApplication.primaryScreen()
            available_h = screen.availableGeometry().height() if screen else parent_height
            reserved = 320 if available_h <= 1100 else 220
            desired = min(desired, max(180, parent_height - reserved))
        if abs(self.height() - desired) > 1:
            self.setFixedHeight(desired)
            self.setMaximumHeight(desired)

    def _update_frame_geometry(self):
        rect = self.contentsRect()
        if rect.width() <= 0 or rect.height() <= 0:
            return
        self._frame.setGeometry(rect)

    def _refresh_pixmap(self):
        if self._source_pixmap.isNull():
            self._frame.setPixmap(QPixmap())
            self._frame.setText(self._placeholder_text)
            return

        self._frame.setText("")
        scaled = self._source_pixmap.scaled(
            self._frame.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self._frame.setPixmap(scaled)


class MarkerSlider(QSlider):
    """Horizontal video timeline with clickable frame markers."""

    seek_requested = pyqtSignal(int)
    marker_requested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self._markers = []
        self.setMouseTracking(True)
        self.setFixedHeight(30)
        self.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 8px;
                background: #d7dde5;
                border-radius: 4px;
            }
            QSlider::sub-page:horizontal {
                background: #0078d4;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                width: 16px;
                height: 16px;
                background: #ffffff;
                border: 2px solid #0078d4;
                border-radius: 8px;
                margin: -5px 0;
            }
        """)

    def set_markers(self, markers):
        self._markers = sorted({int(m) for m in markers if int(m) >= self.minimum()})
        self.update()

    def _value_to_x(self, value):
        span = max(1, self.maximum() - self.minimum())
        usable = max(1, self.width() - 20)
        return 10 + int((int(value) - self.minimum()) / span * usable)

    def _x_to_value(self, x):
        span = max(1, self.maximum() - self.minimum())
        usable = max(1, self.width() - 20)
        ratio = min(1.0, max(0.0, (int(x) - 10) / usable))
        return int(round(self.minimum() + ratio * span))

    def _nearest_marker(self, x):
        if not self._markers:
            return None
        candidates = [(abs(self._value_to_x(m) - x), m) for m in self._markers]
        distance, marker = min(candidates, key=lambda item: item[0])
        return marker if distance <= 10 else None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            marker = self._nearest_marker(event.x())
            if marker is not None:
                self.marker_requested.emit(marker)
                event.accept()
                return
            value = self._x_to_value(event.x())
            self.setValue(value)
            self.seek_requested.emit(value)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.seek_requested.emit(self.value())
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._markers or self.maximum() <= self.minimum():
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        pen = QtGui.QPen(QColor("#d13438"), 3)
        painter.setPen(pen)
        painter.setBrush(QColor("#d13438"))
        y = self.height() // 2
        for marker in self._markers:
            if marker < self.minimum() or marker > self.maximum():
                continue
            x = self._value_to_x(marker)
            painter.drawLine(x, max(3, y - 11), x, min(self.height() - 3, y + 11))
            painter.drawEllipse(QPoint(x, y), 4, 4)


def auto_save_defect(image, save_dir, prefix="defect", extra_info=""):
    if not save_dir or not os.path.exists(save_dir):
        return None
    # 使用完整微秒数和时间计数器确保文件名唯一
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filepath = os.path.join(save_dir, f"{prefix}_{timestamp}{extra_info}.jpg")
    write_image_unicode(filepath, image)
    return filepath


def fit_label_font(lbl: QLabel, min_size=10):
    """自动缩小标签字体使文字不超过标签可用宽度"""
    base_style = lbl.property("_fit_base_style")
    if base_style is None:
        base_style = lbl.styleSheet()
        lbl.setProperty("_fit_base_style", base_style)

    for size in range(36, min_size - 1, -1):
        f = QFont(lbl.font())
        f.setPixelSize(size)
        metrics = QFontMetrics(f)
        if metrics.horizontalAdvance(lbl.text()) <= max(1, lbl.width() - 8):
            if lbl.property("_fit_last_size") != size:
                lbl.setStyleSheet(f"{base_style}; font-size:{size}px;")
                lbl.setProperty("_fit_last_size", size)
            return
    if lbl.property("_fit_last_size") != min_size:
        lbl.setStyleSheet(f"{base_style}; font-size:{min_size}px;")
        lbl.setProperty("_fit_last_size", min_size)

    font = lbl.font()
    start_size = font.pointSize() if font.pointSize() > 0 else min_size
    for size in range(int(start_size), min_size - 1, -1):
        f = QFont(font)
        f.setPointSize(size)
        metrics = QFontMetrics(f)
        w = metrics.horizontalAdvance(lbl.text())
        if w <= lbl.width():
            lbl.setFont(f)
            return


# ──────────────────────────── 批量检测线程 ─────────────────────────
def _preprocess_batch_image(args):
    path, grayscale, exposure = args
    img, read_error = read_image_unicode(path)
    if img is None:
        return path, None, None, read_error
    processed = apply_preprocess(img, grayscale, exposure)
    return path, img, processed, None


class BatchWorker(QThread):
    sig_status = pyqtSignal(str)
    sig_prepare_progress = pyqtSignal(int, int)
    sig_progress = pyqtSignal(int, int)
    sig_result = pyqtSignal(str, list, object, bool, int, list, int, str, float)
    sig_done = pyqtSignal(str, float, float, float)
    sig_aborted = pyqtSignal(str)

    def __init__(self, detector, paths, conf, iou, grayscale, exposure, save_dir,
                 pixel_per_cm, standard, auto_save=False, auto_save_dir=None, auto_save_type=0,
                 detect_height=None, filter_settings=None):
        super().__init__()
        self.detector = detector
        self.paths = paths
        self.conf = conf
        self.iou = iou
        self.grayscale = grayscale
        self.exposure = exposure
        self.save_dir = save_dir
        self.pixel_per_cm = pixel_per_cm
        self.standard = standard
        self.auto_save = auto_save
        self.auto_save_dir = auto_save_dir
        self.auto_save_type = auto_save_type  # 0=不合格, 1=合格, 2=所有
        self.detect_height = detect_height  # 画面横向实际视野宽度(cm)，变量名保留兼容旧配置
        self.filter_engine = ProductionFilterEngine(filter_settings)
        self._abort = False
        self._paused = False
        self._executor = None

    def _should_save(self, passed):
        """根据留存类型判断是否应该保存"""
        if not self.auto_save or not self.auto_save_dir:
            return False
        if self.auto_save_type == 0:  # 留存不合格
            return passed == False
        elif self.auto_save_type == 1:  # 留存合格
            return passed == True
        else:  # 留存所有
            return True

    def _wait_if_paused(self):
        while self._paused and not self._abort:
            time.sleep(0.05)

    def abort(self):
        self._abort = True
        self._paused = False

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def toggle_pause(self):
        self._paused = not self._paused
        return self._paused

    def _write_report(self, log_rows, status, note=""):
        csv_path = os.path.join(self.save_dir, "batch_report.csv")
        rows = list(log_rows)
        rows.append([
            "任务状态", "", "", status, "",
            datetime.now().strftime("%H:%M:%S"),
            note
        ])
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerows(rows)
        return csv_path

    def run(self):
        batch_start_time = time.time()
        os.makedirs(self.save_dir, exist_ok=True)
        if self.auto_save and self.auto_save_dir:
            os.makedirs(self.auto_save_dir, exist_ok=True)
        total = len(self.paths)
        log_rows = [["文件名", "缺陷数", "超标缺陷数", "是否通过", "均值置信度", "时间", "高级过滤"]]
        preprocess_args = [(p, self.grayscale, self.exposure) for p in self.paths]
        preprocess_results = [None] * total
        done_count = 0
        detected_count = 0
        self.sig_status.emit(f"正在准备批量检测数据... 0/{total}")
        self.sig_prepare_progress.emit(0, total)
        try:
            max_workers = min((os.cpu_count() or 1), 4, max(1, len(preprocess_args)))
            self._executor = ThreadPoolExecutor(max_workers=max_workers)
            pending = {}
            next_idx = 0
            while (next_idx < total or pending) and not self._abort:
                self._wait_if_paused()
                while (
                    next_idx < total
                    and len(pending) < max_workers
                    and not self._paused
                    and not self._abort
                ):
                    future = self._executor.submit(_preprocess_batch_image, preprocess_args[next_idx])
                    pending[future] = next_idx
                    next_idx += 1
                if self._abort:
                    break
                if self._paused:
                    continue
                done, _ = wait(pending, timeout=0.05, return_when=FIRST_COMPLETED)
                for future in done:
                    idx = pending.pop(future)
                    preprocess_results[idx] = future.result()
                    done_count += 1
                    self.sig_prepare_progress.emit(done_count, total)
                    self.sig_status.emit(f"正在准备批量检测数据... {done_count}/{total}")
        except Exception:
            logger.exception("Batch threaded preprocessing failed; falling back to serial")
            preprocess_results = [None] * total
            done_count = 0
            self.sig_prepare_progress.emit(0, total)
            self.sig_status.emit(f"正在使用兼容模式准备批量检测数据... 0/{total}")
            for idx, arg in enumerate(preprocess_args):
                if self._abort:
                    break
                self._wait_if_paused()
                if self._abort:
                    break
                preprocess_results[idx] = _preprocess_batch_image(arg)
                done_count = idx + 1
                self.sig_prepare_progress.emit(done_count, total)
                self.sig_status.emit(f"正在使用兼容模式准备批量检测数据... {done_count}/{total}")
        finally:
            if self._executor:
                self._executor.shutdown(wait=False, cancel_futures=True)
                self._executor = None

        if self._abort:
            self._write_report(
                log_rows,
                "已中止",
                f"准备完成 {done_count}/{total}，检测完成 {detected_count}/{total}"
            )
            self.sig_aborted.emit(self.save_dir)
            return

        self.sig_status.emit(f"图像准备完成，开始模型检测... 0/{total}")
        self.sig_progress.emit(0, total)
        inference_times = []
        for i, result in enumerate(preprocess_results):
            if result is None:
                continue
            p, original_img, img, read_error = result
            if self._abort:
                break
            self._wait_if_paused()
            if self._abort:
                break
            try:
                if img is None:
                    logger.warning("Batch image read failed: %s: %s", p, read_error)
                    continue
                t_image_start = time.time()
                with QMutexLocker(detector_mutex):
                    t_infer_start = time.time()
                    dets, im_out = self.detector.run(img, conf_thres=self.conf, iou_thres=self.iou)
                    inference_times.append(time.time() - t_infer_start)
                # 获取实际图像宽度用于动态像素当量计算
                img_h, img_w = img.shape[:2]
                raw_count = len(dets)
                dets, filter_info = self.filter_engine.filter(dets, img.shape, source="batch", temporal=True)
                oversized, passed, details, lethal, calc_log = evaluate_defects_detailed(
                    dets, self.pixel_per_cm, self.standard, self.standard.min_defect_mm,
                    image_width=img_w, ref_width=1920, detect_height=self.detect_height)
                logger.debug("Batch detect %s: pixel_per_cm=%.2f%s", Path(p).name, self.pixel_per_cm, calc_log)
                im_out = draw_defect_labels(im_out, details, self.standard, dets)
                im_out = draw_filter_overlay(im_out, filter_info)
                # 不再无条件保存所有结果 — 由 auto_save 根据条件决定
                saved_path = ""
                if self._should_save(passed):
                    result_str = "qualified" if passed else "unqualified"
                    saved_path = auto_save_defect(im_out, self.auto_save_dir, prefix=f"batch_{Path(p).stem}_{result_str}", extra_info=f"_oversized{oversized}")
                avg_c = f"{sum(float(r[4]) for r in dets) / len(dets):.2f}" if dets else "—"
                filter_note = f"raw={raw_count}, roi_drop={filter_info.get('roi_removed', 0)}, unstable_drop={filter_info.get('unstable_removed', 0)}"
                safe_name = os.path.basename(p)
                if safe_name.startswith(('=', '+', '-', '@')):
                    safe_name = "'" + safe_name
                log_rows.append([safe_name, len(dets), oversized, "通过" if passed else "不通过", avg_c, datetime.now().strftime("%H:%M:%S"), filter_note])
                detected_count += 1
                image_time_ms = (time.time() - t_image_start) * 1000
                self.sig_result.emit(p, dets, im_out, passed, oversized, details, lethal, saved_path, image_time_ms)
                self.sig_progress.emit(i + 1, total)
            except Exception:
                logger.exception("BatchWorker skipped %s", p)
                continue
        if self._abort:
            self._write_report(
                log_rows,
                "已中止",
                f"准备完成 {done_count}/{total}，检测完成 {detected_count}/{total}"
            )
            self.sig_aborted.emit(self.save_dir)
            return
        self._write_report(log_rows, "已完成", f"检测完成 {detected_count}/{total}")

        total_time = time.time() - batch_start_time
        avg_per_image_io = (total_time / detected_count * 1000) if detected_count > 0 else 0.0
        avg_per_image_infer = (sum(inference_times) / len(inference_times) * 1000) if inference_times else 0.0
        self.sig_done.emit(self.save_dir, total_time, avg_per_image_io, avg_per_image_infer)


# ──────────────────────────── 模型面板 ────────────────────────────
class ModelPanel(QGroupBox):
    model_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__("视觉检测模型", parent)
        self.setAcceptDrops(True)
        self.setStyleSheet("QGroupBox { border: 2px solid #c0c0c0; border-radius: 10px; margin-top: 4px; padding-top: 10px; background-color: #ffffff; } QGroupBox::title { subcontrol-origin: padding; subcontrol-position: top left; padding: 0 9px; margin-top: -1px; color: #0078d4; font-weight: bold; font-size: 16px; background-color: #f0f2f5; }")
        self._history = []
        self.weights_dir = BUNDLE_ROOT / "weights"
        self.weights_dir.mkdir(exist_ok=True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 3, 6, 4)
        lay.setSpacing(2)
        self.combo = QComboBox()
        self.combo.setFixedHeight(34)
        self.combo.setStyleSheet("QComboBox { min-height: 0px; max-height: 34px; padding: 2px 8px; font-size: 15px; }")
        self.combo.setToolTip("切换历史模型")
        self.combo.currentIndexChanged.connect(self._on_combo)
        lay.addWidget(self.combo)
        btn = QPushButton("浏览选择 .trt/.onnx 文件")
        self.browse_btn = btn
        btn.setObjectName("btn_side_compact")
        btn.setFixedHeight(36)
        btn.clicked.connect(self.browse)
        lay.addWidget(btn)
        self.runtime_label = QLabel("后端: 未加载")
        self.runtime_label.setWordWrap(True)
        self.runtime_label.setStyleSheet(
            "color:#5a5a5a; font-size:14px; font-weight:bold; padding:4px 6px; background:#f5f7fa; border:1px solid #d7dde5; border-radius:6px;"
        )
        self.runtime_label.setVisible(False)
        lay.addWidget(self.runtime_label)
        self._scan_weights_dir()

    def apply_density(self, profile):
        if profile == "roomy":
            title_size, combo_h, btn_h = 20, 58, 72
            margin_top, padding_top = 6, 12
            margins, spacing = (9, 8, 9, 8), 6
            combo_font, btn_font = 19, 26
        elif profile == "ultra":
            title_size, combo_h, btn_h = 16, 34, 36
            margin_top, padding_top = 2, 6
            margins, spacing = (6, 3, 6, 4), 2
            combo_font, btn_font = 15, 18
        elif profile == "balanced":
            title_size, combo_h, btn_h = 17, 38, 42
            margin_top, padding_top = 4, 8
            margins, spacing = (7, 5, 7, 5), 4
            combo_font, btn_font = 16, 20
        else:
            title_size, combo_h, btn_h = 16, 34, 36
            margin_top, padding_top = 2, 6
            margins, spacing = (6, 3, 6, 4), 2
            combo_font, btn_font = 15, 18
        title_offset = -1 if profile in {"ultra", "compact"} else (-2 if profile == "balanced" else -3)
        self.setStyleSheet(f"QGroupBox {{ border: 2px solid #c0c0c0; border-radius: 10px; margin-top: {margin_top + 2}px; padding-top: {padding_top + 4}px; background-color: #ffffff; }} QGroupBox::title {{ subcontrol-origin: padding; subcontrol-position: top left; padding: 0 9px; margin-top: {title_offset}px; color: #0078d4; font-weight: bold; font-size: {title_size}px; background-color: #f0f2f5; }}")
        self.layout().setContentsMargins(*margins)
        self.layout().setSpacing(spacing)
        self.combo.setFixedHeight(combo_h)
        self.combo.setStyleSheet(f"QComboBox {{ min-height: 0px; max-height: {combo_h}px; padding: 2px 8px; font-size: {combo_font}px; }}")
        self.browse_btn.setFixedHeight(btn_h)
        self.browse_btn.setStyleSheet(f"font-size:{btn_font}px; padding:3px 8px;")
        runtime_font = max(13, combo_font - 1)
        self.runtime_label.setStyleSheet(
            f"color:#5a5a5a; font-size:{runtime_font}px; font-weight:bold; padding:4px 6px; background:#f5f7fa; border:1px solid #d7dde5; border-radius:6px;"
        )

    def _scan_weights_dir(self):
        if not self.weights_dir.exists():
            return
        model_files = sorted(self.weights_dir.glob("*.trt")) + sorted(self.weights_dir.glob("*.onnx"))
        for pt_path in model_files:
            path_str = str(pt_path.absolute())
            if path_str not in self._history:
                self._history.append(path_str)
                self.combo.blockSignals(True)
                self.combo.addItem(pt_path.name, path_str)
                self.combo.blockSignals(False)
        if model_files and self.combo.count() > 0:
            self.combo.blockSignals(True)
            self.combo.setCurrentIndex(0)
            self.combo.blockSignals(False)
            self.model_changed.emit(self._history[0])

    def set_initial(self, path: str):
        path = os.path.normpath(path)
        name = os.path.basename(path)
        if path not in self._history:
            self._history.append(path)
            self.combo.blockSignals(True)
            self.combo.addItem(name, path)
            self.combo.blockSignals(False)
        idx = self._history.index(path) if path in self._history else 0
        self.combo.blockSignals(True)
        self.combo.setCurrentIndex(idx)
        self.combo.blockSignals(False)

    def load_model(self, path: str):
        path = os.path.normpath(path)
        name = os.path.basename(path)
        if path not in self._history:
            self._history.append(path)
            self.combo.blockSignals(True)
            self.combo.addItem(name, path)
            self.combo.blockSignals(False)
        self.combo.blockSignals(True)
        self.combo.setCurrentIndex(self._history.index(path))
        self.combo.blockSignals(False)
        self.model_changed.emit(path)

    def browse(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "模型文件 (*.trt *.onnx)")
        if p:
            self.load_model(p)

    def _on_combo(self, idx):
        if 0 <= idx < len(self._history):
            p = self._history[idx]
            self.model_changed.emit(p)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() and any(u.toLocalFile().lower().endswith(('.trt', '.onnx')) for u in e.mimeData().urls()):
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            p = u.toLocalFile()
            if p.lower().endswith(('.trt', '.onnx')):
                self.load_model(p)
                break

    def set_runtime_status(self, text: str, color: str = "#5a5a5a"):
        self.runtime_label.setText(text)
        self.runtime_label.setStyleSheet(
            f"color:{color}; font-size:14px; font-weight:bold; padding:4px 6px; background:#f5f7fa; border:1px solid #d7dde5; border-radius:6px;"
        )


# ──────────────────────────── 检测设置弹出对话框 ─────────────────────────
class AdvancedFilterSettingsDialog(QDialog):
    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.setWindowTitle("高级产线过滤设置")
        self.setMinimumWidth(720)
        self.setMinimumHeight(620)
        self.setModal(True)
        self._settings = merge_filter_settings(settings)
        self.init_ui()
        self.load_settings(self._settings)

    def init_ui(self):
        layout = QVBoxLayout(self)

        self.status_label = QLabel()
        self.status_label.setTextFormat(Qt.RichText)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(
            "font-size:16px; font-weight:bold; background:#f5f7fa; "
            "border:2px solid #d1ddea; border-radius:10px; padding:10px;"
        )
        layout.addWidget(self.status_label)

        roi_group = QGroupBox("ROI屏蔽")
        roi_layout = QFormLayout(roi_group)
        self.roi_enabled_cb = QCheckBox("启用ROI屏蔽")
        self.roi_margin_spin = QDoubleSpinBox()
        self.roi_margin_spin.setRange(0.0, 45.0)
        self.roi_margin_spin.setSingleStep(1.0)
        self.roi_margin_spin.setSuffix(" %")
        self.roi_rule_combo = QComboBox()
        self.roi_rule_combo.addItem("检测框中心点必须在ROI内", "center")
        self.roi_rule_combo.addItem("检测框四个角点都必须在ROI内", "corners")
        roi_hint = QLabel("默认边距10%，即仅检测画面中心80%区域；1920×1080时ROI为1536×864。")
        roi_hint.setWordWrap(True)
        roi_hint.setStyleSheet("color:#5a5a5a; font-size:14px;")
        roi_layout.addRow(self.roi_enabled_cb)
        roi_layout.addRow("四周屏蔽边距:", self.roi_margin_spin)
        roi_layout.addRow("保留规则:", self.roi_rule_combo)
        roi_layout.addRow(roi_hint)
        layout.addWidget(roi_group)

        stable_group = QGroupBox("多帧确认 / 帧稳定性滤波")
        stable_layout = QFormLayout(stable_group)
        self.stable_enabled_cb = QCheckBox("启用连续帧确认")
        self.stable_frames_spin = QSpinBox()
        self.stable_frames_spin.setRange(1, 120)
        self.stable_frames_spin.setSuffix(" 帧")
        self.track_iou_spin = QDoubleSpinBox()
        self.track_iou_spin.setRange(0.05, 0.90)
        self.track_iou_spin.setSingleStep(0.05)
        self.track_iou_spin.setDecimals(2)
        stable_hint = QLabel("目标连续出现达到设定帧数后才输出，可过滤震动、飞虫、瞬时反光造成的闪框。")
        stable_hint.setWordWrap(True)
        stable_hint.setStyleSheet("color:#5a5a5a; font-size:14px;")
        stable_layout.addRow(self.stable_enabled_cb)
        stable_layout.addRow("确认帧数N:", self.stable_frames_spin)
        stable_layout.addRow("跟踪匹配IoU:", self.track_iou_spin)
        stable_layout.addRow(stable_hint)
        layout.addWidget(stable_group)

        count_group = QGroupBox("计数去抖逻辑")
        count_layout = QFormLayout(count_group)
        self.count_enabled_cb = QCheckBox("启用计数线去抖")
        self.count_orientation_combo = QComboBox()
        self.count_orientation_combo.addItem("垂直计数线 X=比例位置", "vertical")
        self.count_orientation_combo.addItem("水平计数线 Y=比例位置", "horizontal")
        self.count_line_spin = QDoubleSpinBox()
        self.count_line_spin.setRange(5.0, 95.0)
        self.count_line_spin.setSingleStep(1.0)
        self.count_line_spin.setSuffix(" %")
        self.count_move_spin = QDoubleSpinBox()
        self.count_move_spin.setRange(0.0, 2000.0)
        self.count_move_spin.setSingleStep(5.0)
        self.count_move_spin.setSuffix(" px")
        self.count_interval_spin = QSpinBox()
        self.count_interval_spin.setRange(0, 10000)
        self.count_interval_spin.setSingleStep(100)
        self.count_interval_spin.setSuffix(" ms")
        count_hint = QLabel("同一跟踪目标只在穿越计数线且满足最小位移/时间间隔时计数一次，避免同一产品反复累计。")
        count_hint.setWordWrap(True)
        count_hint.setStyleSheet("color:#5a5a5a; font-size:14px;")
        count_layout.addRow(self.count_enabled_cb)
        count_layout.addRow("计数线方向:", self.count_orientation_combo)
        count_layout.addRow("计数线位置:", self.count_line_spin)
        count_layout.addRow("最小位移:", self.count_move_spin)
        count_layout.addRow("最小间隔:", self.count_interval_spin)
        count_layout.addRow(count_hint)
        layout.addWidget(count_group)

        speed_group = QGroupBox("基于速度的自适应")
        speed_layout = QFormLayout(speed_group)
        self.speed_enabled_cb = QCheckBox("启用速度自适应")
        self.line_speed_spin = QDoubleSpinBox()
        self.line_speed_spin.setRange(0.1, 300.0)
        self.line_speed_spin.setSingleStep(0.1)
        self.line_speed_spin.setDecimals(1)
        self.line_speed_spin.setSuffix(" m/min")
        self.base_speed_spin = QDoubleSpinBox()
        self.base_speed_spin.setRange(0.1, 300.0)
        self.base_speed_spin.setSingleStep(0.1)
        self.base_speed_spin.setDecimals(1)
        self.base_speed_spin.setSuffix(" m/min")
        speed_hint = QLabel("速度高于基准时自动降低确认帧数并放宽跟踪预测步长；速度低时提高稳定性。当前速度由用户手动输入。")
        speed_hint.setWordWrap(True)
        speed_hint.setStyleSheet("color:#5a5a5a; font-size:14px;")
        speed_layout.addRow(self.speed_enabled_cb)
        speed_layout.addRow("当前线速:", self.line_speed_spin)
        speed_layout.addRow("基准线速:", self.base_speed_spin)
        speed_layout.addRow(speed_hint)
        layout.addWidget(speed_group)

        btn_layout = QHBoxLayout()
        btn_default = QPushButton("恢复默认")
        btn_ok = QPushButton("确定")
        btn_ok.setObjectName("btn_active")
        btn_cancel = QPushButton("取消")
        btn_default.clicked.connect(lambda: self.load_settings(default_filter_settings()))
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_default)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_ok)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)
        self._connect_status_updates()

    def _badge(self, name, enabled, extra=""):
        bg = "#107c10" if enabled else "#6b7280"
        text = "已开启" if enabled else "已关闭"
        return (
            f"<span style='background:{bg}; color:white; padding:4px 9px; "
            f"border-radius:8px; margin-right:6px;'>{name}: {text}{extra}</span>"
        )

    def _status_html(self):
        return " ".join([
            self._badge("ROI", self.roi_enabled_cb.isChecked(), f" {100 - self.roi_margin_spin.value() * 2:.0f}%区域"),
            self._badge("稳帧", self.stable_enabled_cb.isChecked(), f" N={self.stable_frames_spin.value()}"),
            self._badge("计数", self.count_enabled_cb.isChecked(), f" {self.count_line_spin.value():.0f}%线"),
            self._badge("变速", self.speed_enabled_cb.isChecked(), f" {self.line_speed_spin.value():.1f}m/min"),
        ])

    def refresh_status(self):
        self.status_label.setText(self._status_html())

    def _connect_status_updates(self):
        widgets = [
            self.roi_enabled_cb, self.stable_enabled_cb, self.count_enabled_cb, self.speed_enabled_cb,
            self.roi_margin_spin, self.stable_frames_spin, self.count_line_spin, self.line_speed_spin,
        ]
        for widget in widgets:
            if hasattr(widget, "stateChanged"):
                widget.stateChanged.connect(lambda *_: self.refresh_status())
            elif hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(lambda *_: self.refresh_status())

    def load_settings(self, settings):
        s = merge_filter_settings(settings)
        self.roi_enabled_cb.setChecked(bool(s["roi_enabled"]))
        self.roi_margin_spin.setValue(float(s["roi_margin_pct"]))
        self.roi_rule_combo.setCurrentIndex(max(0, self.roi_rule_combo.findData(s["roi_rule"])))
        self.stable_enabled_cb.setChecked(bool(s["stable_enabled"]))
        self.stable_frames_spin.setValue(int(s["stable_frames"]))
        self.track_iou_spin.setValue(float(s["track_iou"]))
        self.count_enabled_cb.setChecked(bool(s["count_enabled"]))
        self.count_orientation_combo.setCurrentIndex(max(0, self.count_orientation_combo.findData(s["count_orientation"])))
        self.count_line_spin.setValue(float(s["count_line_pct"]))
        self.count_move_spin.setValue(float(s["count_min_displacement"]))
        self.count_interval_spin.setValue(int(s["count_min_interval_ms"]))
        self.speed_enabled_cb.setChecked(bool(s["speed_enabled"]))
        self.line_speed_spin.setValue(float(s["line_speed"]))
        self.base_speed_spin.setValue(float(s["base_speed"]))
        self.refresh_status()

    def get_settings(self):
        return {
            "roi_enabled": self.roi_enabled_cb.isChecked(),
            "roi_margin_pct": self.roi_margin_spin.value(),
            "roi_rule": self.roi_rule_combo.currentData(),
            "stable_enabled": self.stable_enabled_cb.isChecked(),
            "stable_frames": self.stable_frames_spin.value(),
            "track_iou": self.track_iou_spin.value(),
            "count_enabled": self.count_enabled_cb.isChecked(),
            "count_orientation": self.count_orientation_combo.currentData(),
            "count_line_pct": self.count_line_spin.value(),
            "count_min_displacement": self.count_move_spin.value(),
            "count_min_interval_ms": self.count_interval_spin.value(),
            "speed_enabled": self.speed_enabled_cb.isChecked(),
            "line_speed": self.line_speed_spin.value(),
            "base_speed": self.base_speed_spin.value(),
        }


class DetectionSettingsDialog(QDialog):
    """检测模型设置和图像预处理的弹出式设置面板"""
    settings_changed = pyqtSignal(float, float, bool, float, dict)

    def __init__(self, parent=None, conf=0.5, iou=0.5, grayscale=False, exposure=1.0, filter_settings=None):
        super().__init__(parent)
        self.setWindowTitle("检测设置")
        self.setMinimumWidth(700)
        self.setMinimumHeight(600)
        self.setModal(True)
        self._conf = conf
        self._iou = iou
        self._grayscale = grayscale
        self._exposure = exposure
        self._filter_settings = merge_filter_settings(filter_settings)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 检测参数设置组
        detect_group = QGroupBox("模型检测参数")
        detect_group.setStyleSheet("""
            QGroupBox { border: 2px solid #0078d4; border-radius: 16px; margin-top: 28px; padding-top: 24px; background-color: #ffffff; }
            QGroupBox::title { subcontrol-origin: padding; subcontrol-position: top left; padding: 0 12px; margin-top: -14px; color: #0078d4; font-weight: bold; font-size: 18px; background-color: #ffffff; }
        """)
        detect_layout = QVBoxLayout(detect_group)

        # 置信度阈值
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("置信度阈值 (Confidence)"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 99)
        self.conf_slider.setValue(int(self._conf * 100))
        self.lbl_conf = QLabel(f"{self._conf:.2f}")
        self.lbl_conf.setStyleSheet("color:#0078d4; font-weight:bold; font-size:17px; min-width:60px;")
        self.lbl_conf.setAlignment(Qt.AlignCenter)
        self.conf_slider.valueChanged.connect(lambda v: self.lbl_conf.setText(f"{v / 100:.2f}"))
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.lbl_conf)
        detect_layout.addLayout(conf_layout)

        # IoU阈值
        iou_layout = QHBoxLayout()
        iou_layout.addWidget(QLabel("交并比阈值 (IoU)"))
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(1, 99)
        self.iou_slider.setValue(int(self._iou * 100))
        self.lbl_iou = QLabel(f"{self._iou:.2f}")
        self.lbl_iou.setStyleSheet("color:#0078d4; font-weight:bold; font-size:17px; min-width:60px;")
        self.lbl_iou.setAlignment(Qt.AlignCenter)
        self.iou_slider.valueChanged.connect(lambda v: self.lbl_iou.setText(f"{v / 100:.2f}"))
        iou_layout.addWidget(self.iou_slider)
        iou_layout.addWidget(self.lbl_iou)
        detect_layout.addLayout(iou_layout)

        layout.addWidget(detect_group)

        # 图像预处理组
        preprocess_group = QGroupBox("图像预处理")
        preprocess_group.setStyleSheet("""
            QGroupBox { border: 2px solid #0078d4; border-radius: 16px; margin-top: 28px; padding-top: 24px; background-color: #ffffff; }
            QGroupBox::title { subcontrol-origin: padding; subcontrol-position: top left; padding: 0 12px; margin-top: -14px; color: #0078d4; font-weight: bold; font-size: 18px; background-color: #ffffff; }
        """)
        preprocess_layout = QVBoxLayout(preprocess_group)

        # 灰度模式
        self.gray_cb = QCheckBox("灰度模式")
        self.gray_cb.setChecked(self._grayscale)
        preprocess_layout.addWidget(self.gray_cb)

        # 曝光补偿
        exp_layout = QHBoxLayout()
        exp_layout.addWidget(QLabel("曝光补偿因子"))
        self.exp_slider = QSlider(Qt.Horizontal)
        self.exp_slider.setRange(10, 300)
        self.exp_slider.setValue(int(self._exposure * 100))
        self.lbl_exp = QLabel(f"{self._exposure:.2f}×")
        self.lbl_exp.setStyleSheet("color:#0078d4; font-weight:bold; font-size:17px; min-width:60px;")
        self.lbl_exp.setAlignment(Qt.AlignCenter)
        self.exp_slider.valueChanged.connect(lambda v: self.lbl_exp.setText(f"{v / 100:.2f}×"))
        exp_layout.addWidget(self.exp_slider)
        exp_layout.addWidget(self.lbl_exp)
        preprocess_layout.addLayout(exp_layout)

        # 重置曝光按钮
        btn_rst_exp = QPushButton("重置曝光补偿")
        btn_rst_exp.clicked.connect(lambda: (self.exp_slider.setValue(100), self.lbl_exp.setText("1.00×")))
        preprocess_layout.addWidget(btn_rst_exp)

        layout.addWidget(preprocess_group)

        filter_group = QGroupBox("产线过滤与稳定输出")
        filter_group.setStyleSheet("""
            QGroupBox { border: 2px solid #0078d4; border-radius: 16px; margin-top: 28px; padding-top: 24px; background-color: #ffffff; }
            QGroupBox::title { subcontrol-origin: padding; subcontrol-position: top left; padding: 0 12px; margin-top: -14px; color: #0078d4; font-weight: bold; font-size: 18px; background-color: #ffffff; }
        """)
        filter_layout = QVBoxLayout(filter_group)
        self.lbl_filter_summary = QLabel(self._filter_summary_html())
        self.lbl_filter_summary.setTextFormat(Qt.RichText)
        self.lbl_filter_summary.setWordWrap(True)
        self.lbl_filter_summary.setStyleSheet("font-size:16px; font-weight:bold; background:#f5f7fa; border:2px solid #d1ddea; border-radius:8px; padding:10px;")
        btn_filter = QPushButton("打开高级产线过滤设置")
        btn_filter.setObjectName("btn_active")
        btn_filter.clicked.connect(self.open_filter_settings)
        filter_layout.addWidget(self.lbl_filter_summary)
        filter_layout.addWidget(btn_filter)
        layout.addWidget(filter_group)
        layout.addStretch()

        # 按钮行
        btn_layout = QHBoxLayout()
        btn_ok = QPushButton("确定")
        btn_ok.setObjectName("btn_active")
        btn_cancel = QPushButton("取消")
        btn_ok.clicked.connect(self.on_ok)
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_ok)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)

    def on_ok(self):
        self.settings_changed.emit(
            self.conf_slider.value() / 100,
            self.iou_slider.value() / 100,
            self.gray_cb.isChecked(),
            self.exp_slider.value() / 100,
            dict(self._filter_settings)
        )
        self.accept()

    def _filter_summary(self):
        s = merge_filter_settings(self._filter_settings)
        items = [f"ROI {'开启' if s['roi_enabled'] else '关闭'}({100 - s['roi_margin_pct'] * 2:.0f}%区域)"]
        items.append(f"稳帧 {'开启' if s['stable_enabled'] else '关闭'}(N={s['stable_frames']})")
        items.append(f"计数去抖 {'开启' if s['count_enabled'] else '关闭'}")
        items.append(f"速度自适应 {'开启' if s['speed_enabled'] else '关闭'}({s['line_speed']:.1f}m/min)")
        return " | ".join(items)

    def _filter_badge(self, name, enabled, extra=""):
        bg = "#107c10" if enabled else "#6b7280"
        text = "已开启" if enabled else "已关闭"
        return (
            f"<span style='background:{bg}; color:white; padding:4px 9px; "
            f"border-radius:8px; margin-right:6px;'>{name}: {text}{extra}</span>"
        )

    def _filter_summary_html(self):
        s = merge_filter_settings(self._filter_settings)
        return " ".join([
            self._filter_badge("ROI", s["roi_enabled"], f" {100 - s['roi_margin_pct'] * 2:.0f}%区域"),
            self._filter_badge("稳帧", s["stable_enabled"], f" N={s['stable_frames']}"),
            self._filter_badge("计数", s["count_enabled"]),
            self._filter_badge("变速", s["speed_enabled"], f" {s['line_speed']:.1f}m/min"),
        ])

    def open_filter_settings(self):
        dlg = AdvancedFilterSettingsDialog(self, self._filter_settings)
        if dlg.exec_() == QDialog.Accepted:
            self._filter_settings = merge_filter_settings(dlg.get_settings())
            self.lbl_filter_summary.setText(self._filter_summary_html())

    def get_settings(self):
        return (
            self.conf_slider.value() / 100,
            self.iou_slider.value() / 100,
            self.gray_cb.isChecked(),
            self.exp_slider.value() / 100,
            dict(self._filter_settings)
        )


# ──────────────────────────── 侧边控制面板（新增总结计数 + 默认IoU=0.5）────────────────────────
class ControlPanel(QWidget):
    model_changed = pyqtSignal(str)
    standard_changed = pyqtSignal(object, float)
    backend_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        root = QVBoxLayout(self)
        self.root_layout = root
        root.setContentsMargins(4, 0, 4, 3)  # 顶部贴齐，避免模型块上方留白
        root.setSpacing(2)  # 紧凑间距

        self.model_panel = ModelPanel()
        self.model_panel.model_changed.connect(self.model_changed)
        root.addWidget(self.model_panel)

        # 综合检测统计 - 2行3列卡片式布局，避免窄侧栏挤压
        stat_box = QGroupBox("综合检测统计")
        self.stat_box = stat_box
        stat_box.setStyleSheet("QGroupBox { border: 2px solid #c0c0c0; border-radius: 10px; margin-top: 6px; padding-top: 8px; background-color: #ffffff; } QGroupBox::title { subcontrol-origin: padding; subcontrol-position: top left; padding: 0 8px; margin-top: -5px; color: #0078d4; font-weight: bold; font-size: 15px; background-color: #f0f2f5; }")
        sg = QGridLayout(stat_box)
        self.stat_layout = sg
        sg.setContentsMargins(5, 7, 5, 5)
        sg.setSpacing(3)

        def card(title, val="0"):
            box = QWidget()
            box.setStyleSheet("background:#f0f8ff; border-radius:10px; border:1px solid #c0d8f0;")
            lay = QVBoxLayout(box)
            lay.setContentsMargins(3, 3, 3, 3)
            lay.setSpacing(1)
            v = QLabel(val)
            v.setObjectName("stat_value")
            v.setAlignment(Qt.AlignCenter)
            v.setMinimumHeight(24)
            value_font = QFont("Microsoft YaHei UI", 13)
            value_font.setBold(True)
            v.setFont(value_font)
            v.setStyleSheet("color: #0078d4; background:transparent;")
            t = QLabel(title)
            t.setObjectName("stat_title")
            t.setAlignment(Qt.AlignCenter)
            t.setStyleSheet("font-size: 13px; color: #5a5a5a; font-weight:bold; background:transparent;")
            lay.addWidget(v)
            lay.addWidget(t)
            return box, v, t

        (self.card_total, self.lv_total, self.lt_total) = card("检视总数")
        (self.card_defect, self.lv_defect, self.lt_defect) = card("缺陷样本")
        (self.card_count, self.lv_count, self.lt_count) = card("缺陷累计")
        (self.card_fps, self.lv_fps, self.lt_fps) = card("输出FPS", "—")
        (self.card_avg, self.lbl_avg_time, self.lt_avg_time) = card("平均耗时", "—")
        (self.card_last, self.lbl_last_time, self.lt_last_time) = card("上次耗时", "—")
        self.stat_value_labels = [
            self.lv_total, self.lv_defect, self.lv_count, self.lv_fps,
            self.lbl_avg_time, self.lbl_last_time
        ]
        self.stat_title_labels = [
            self.lt_total, self.lt_defect, self.lt_count, self.lt_fps,
            self.lt_avg_time, self.lt_last_time
        ]

        # 2行3列网格：每个卡片为独立整体
        sg.addWidget(self.card_total, 0, 0)
        sg.addWidget(self.card_defect, 0, 1)
        sg.addWidget(self.card_count, 0, 2)
        sg.addWidget(self.card_fps, 1, 0)
        sg.addWidget(self.card_avg, 1, 1)
        sg.addWidget(self.card_last, 1, 2)
        root.addWidget(stat_box)

        analysis_group = QGroupBox("产线分析")
        self.analysis_group = analysis_group
        analysis_group.setStyleSheet("QGroupBox { border: 2px solid #b8c7d9; border-radius: 11px; margin-top: 11px; padding-top: 14px; background-color: #ffffff; } QGroupBox::title { subcontrol-origin: padding; subcontrol-position: top left; padding: 0 10px; margin-top: -8px; color: #0078d4; font-weight: bold; font-size: 16px; background-color: #f0f2f5; }")
        analysis_layout = QGridLayout(analysis_group)
        self.analysis_layout = analysis_layout
        analysis_layout.setContentsMargins(6, 10, 6, 6)
        analysis_layout.setHorizontalSpacing(5)
        analysis_layout.setVerticalSpacing(4)

        def mini_metric(title, value="—"):
            lbl = QLabel(f"{title}: {value}")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setMinimumHeight(28)
            lbl.setStyleSheet("font-size:14px; font-weight:bold; color:#172033; padding:3px 5px; background:#f7faff; border:1px solid #d1ddea; border-radius:6px;")
            return lbl

        self.lbl_yield_rate = mini_metric("合格率")
        self.lbl_fail_streak = mini_metric("连续异常")
        self.lbl_avg_defects = mini_metric("平均缺陷")
        self.lbl_line_health = mini_metric("工位评分")
        self.lbl_recent_fail = mini_metric("近20异常")
        self.lbl_size_avg = mini_metric("尺寸均值")
        self.analysis_metric_labels = [
            self.lbl_yield_rate, self.lbl_fail_streak, self.lbl_avg_defects,
            self.lbl_line_health, self.lbl_recent_fail, self.lbl_size_avg
        ]
        analysis_layout.addWidget(self.lbl_yield_rate, 0, 0)
        analysis_layout.addWidget(self.lbl_fail_streak, 0, 1)
        analysis_layout.addWidget(self.lbl_avg_defects, 1, 0)
        analysis_layout.addWidget(self.lbl_line_health, 1, 1)
        analysis_layout.addWidget(self.lbl_recent_fail, 2, 0)
        analysis_layout.addWidget(self.lbl_size_avg, 2, 1)
        self.btn_analysis_summary = QPushButton("写入分析摘要")
        self.btn_analysis_summary.setObjectName("btn_side_compact")
        self.btn_analysis_summary.setFixedHeight(36)
        self.btn_analysis_mark = QPushButton("标记异常批次")
        self.btn_analysis_mark.setObjectName("btn_side_compact")
        self.btn_analysis_mark.setFixedHeight(36)
        self.btn_analysis_summary.clicked.connect(self.append_analysis_snapshot)
        self.btn_analysis_mark.clicked.connect(self.mark_abnormal_batch)
        analysis_layout.addWidget(self.btn_analysis_summary, 3, 0)
        analysis_layout.addWidget(self.btn_analysis_mark, 3, 1)
        root.addWidget(analysis_group)

        # 检测评判标准
        std_group = QGroupBox("检测评判标准")
        self.std_group = std_group
        std_group.setStyleSheet("QGroupBox { border: 2px solid #c0c0c0; border-radius: 10px; margin-top: 8px; padding-top: 10px; background-color: #ffffff; } QGroupBox::title { subcontrol-origin: padding; subcontrol-position: top left; padding: 0 8px; margin-top: -6px; color: #0078d4; font-weight: bold; font-size: 14px; background-color: #f0f2f5; }")
        std_layout = QVBoxLayout(std_group)
        self.std_layout = std_layout
        std_layout.setContentsMargins(5, 6, 5, 5)
        std_layout.setSpacing(3)
        self.std_combo = QComboBox()
        self.std_combo.setFixedHeight(34)
        self.std_combo.setStyleSheet("QComboBox { min-height: 0px; max-height: 34px; padding: 2px 8px; font-size: 15px; }")
        self.std_combo.currentIndexChanged.connect(self._on_std_changed)
        self.btn_config = QPushButton("配置标准参数")
        self.btn_config.setObjectName("btn_side_compact")
        self.btn_config.setFixedHeight(36)
        self.btn_config.clicked.connect(self.open_std_config)
        std_layout.addWidget(QLabel("应用标准:"))
        std_layout.addWidget(self.std_combo)
        std_layout.addWidget(self.btn_config)
        root.addWidget(std_group)

        # 检测对象快速选择
        obj_group = QGroupBox("检测对象选择")
        self.obj_group = obj_group
        obj_group.setStyleSheet("QGroupBox { border: 2px solid #c0c0c0; border-radius: 10px; margin-top: 8px; padding-top: 10px; background-color: #ffffff; } QGroupBox::title { subcontrol-origin: padding; subcontrol-position: top left; padding: 0 8px; margin-top: -6px; color: #0078d4; font-weight: bold; font-size: 14px; background-color: #f0f2f5; }")
        obj_layout = QHBoxLayout(obj_group)
        self.obj_layout = obj_layout
        obj_layout.setContentsMargins(5, 6, 5, 5)
        obj_layout.setSpacing(5)
        self.btn_scratch = QPushButton("屏幕划痕")
        self.btn_scratch.setCheckable(True)
        self.btn_scratch.setChecked(True)
        self.btn_scratch.setObjectName("btn_side_compact")
        self.btn_scratch.setFixedHeight(36)
        self.btn_bolt = QPushButton("铝合金零件")
        self.btn_bolt.setCheckable(True)
        self.btn_bolt.setObjectName("btn_side_compact")
        self.btn_bolt.setFixedHeight(36)
        self.btn_scratch.clicked.connect(lambda: self._on_object_btn_clicked(0))
        self.btn_bolt.clicked.connect(lambda: self._on_object_btn_clicked(1))
        obj_layout.addWidget(self.btn_scratch)
        obj_layout.addWidget(self.btn_bolt)
        root.addWidget(obj_group)

        auto_group = QGroupBox("自动化数据记录")
        self.auto_group = auto_group
        auto_group.setStyleSheet("QGroupBox { border: 2px solid #c0c0c0; border-radius: 10px; margin-top: 8px; padding-top: 10px; background-color: #ffffff; } QGroupBox::title { subcontrol-origin: padding; subcontrol-position: top left; padding: 0 8px; margin-top: -6px; color: #0078d4; font-weight: bold; font-size: 14px; background-color: #f0f2f5; }")
        auto_layout = QVBoxLayout(auto_group)
        self.auto_layout = auto_layout
        auto_layout.setContentsMargins(5, 6, 5, 5)
        auto_layout.setSpacing(3)
        self.auto_save_cb = QCheckBox("自动留存样本")
        self.auto_save_cb.toggled.connect(self.on_auto_save_toggled)
        self.auto_report_cb = QCheckBox("自动生成结果报告")
        # 延迟到 __init__ 末尾再 setChecked，避免控件未就绪时触发 handler
        self.auto_save_type_combo = QComboBox()
        self.auto_save_type_combo.setFixedHeight(34)
        self.auto_save_type_combo.setStyleSheet("QComboBox { min-height: 0px; max-height: 34px; padding: 2px 8px; font-size: 15px; }")
        self.auto_save_type_combo.addItems(["留存不合格样本", "留存合格样本", "留存所有样本"])
        self.auto_save_type_combo.setEnabled(False)
        self.auto_save_path_btn = QPushButton("指定留存目录")
        self.auto_save_path_btn.setEnabled(False)
        self.auto_save_path_btn.setObjectName("disabled_btn")
        self.auto_save_path_btn.setStyleSheet("min-height:0px; max-height:36px; font-size:18px; padding:3px 8px;")
        self.auto_save_path_btn.setFixedHeight(36)
        self.auto_save_path_label = QLabel("未选择目录")
        self.auto_save_path_label.setMinimumHeight(22)
        self.auto_save_path_label.setStyleSheet("color:#5a5a5a; font-size:15px; font-weight:bold; padding:1px 4px; background:#f5f7fa; border-radius:4px;")
        self.auto_save_path_label.setWordWrap(True)
        self.auto_save_type_combo.currentIndexChanged.connect(self._on_auto_save_type_changed)
        self.auto_save_path_btn.clicked.connect(self.select_auto_save_dir)
        auto_layout.addWidget(self.auto_save_cb)
        auto_layout.addWidget(self.auto_report_cb)
        auto_layout.addWidget(self.auto_save_type_combo)
        auto_layout.addWidget(self.auto_save_path_btn)
        auto_layout.addWidget(self.auto_save_path_label)
        root.addWidget(auto_group)

        # 检测设置按钮 - 弹出式设置面板
        settings_group = QGroupBox("检测参数配置")
        self.settings_group = settings_group
        settings_group.setStyleSheet("QGroupBox { border: 2px solid #c0c0c0; border-radius: 10px; margin-top: 8px; padding-top: 10px; background-color: #ffffff; } QGroupBox::title { subcontrol-origin: padding; subcontrol-position: top left; padding: 0 8px; margin-top: -6px; color: #0078d4; font-weight: bold; font-size: 14px; background-color: #f0f2f5; }")
        settings_layout = QVBoxLayout(settings_group)
        self.settings_layout = settings_layout
        settings_layout.setContentsMargins(5, 6, 5, 5)
        settings_layout.setSpacing(3)

        # 当前设置显示
        self.lbl_current_settings = QLabel("置信度: 0.50 | IoU: 0.50")
        self.lbl_current_settings.setTextFormat(Qt.RichText)
        self.lbl_current_settings.setWordWrap(True)
        self.lbl_current_settings.setStyleSheet("color:#5a5a5a; font-size:15px; font-weight:bold; padding:3px; background:#f5f5f5; border-radius:6px;")
        self.lbl_current_settings.setAlignment(Qt.AlignCenter)
        settings_layout.addWidget(self.lbl_current_settings)

        self.btn_open_settings = QPushButton("打开配置面板")
        self.btn_open_settings.setObjectName("btn_active")
        self.btn_open_settings.setStyleSheet("font-size:20px; min-height:0px; max-height:40px; padding:3px 8px;")
        self.btn_open_settings.setFixedHeight(40)
        self.btn_open_settings.clicked.connect(self.open_settings_dialog)
        settings_layout.addWidget(self.btn_open_settings)
        root.addWidget(settings_group)

        self._conf = 0.50
        self._iou = 0.50
        self._gray = False
        self._exposure = 1.0
        self.filter_settings = merge_filter_settings()
        self.filter_engine = ProductionFilterEngine(self.filter_settings)
        self._last_filter_info = {}
        self.lbl_current_settings.setText(self.settings_label_text())
        self._total = 0
        self._defect_frames = 0
        self._total_defects = 0
        self._fps_start_time = None
        self._fps_frame_count = 0
        self._fps_times = []
        self._fps_last_time = None
        self._log_rows = []
        self._event_rows = []
        self._batch_stats = {}
        self._batch_size = int(os.getenv("QC_BATCH_SIZE", "50"))
        self._last_realtime_log_time = 0.0
        self._last_realtime_stats_ui_time = 0.0
        self._recent_passed = []
        self._fail_streak = 0
        self._defect_diameters = []
        self.qualified = 0          # 新增合格计数
        self.unqualified = 0        # 新增不合格计数
        self.standards = []
        self.current_standard = None
        self.pixel_per_cm = 100.0
        self.detect_height = 19.2   # 画面横向实际视野宽度(cm)，1920/19.2=100px/cm
        self.load_default_standards()
        self.auto_save_dir = ""
        self.auto_save_type = 0  # 0=不合格, 1=合格, 2=所有
        self.log_edit = None
        self.refresh_analysis()

        # 安装事件过滤器，拦截滚轮事件
        self.installEventFilter(self)

        # 最后才设置默认勾选，确保所有控件已就绪
        self.auto_save_cb.setChecked(True)
        self.auto_report_cb.setChecked(True)

    def apply_density(self, viewport_height):
        if viewport_height >= 1280:
            profile = "roomy"
        elif viewport_height >= 1080:
            profile = "balanced"
        else:
            profile = "compact"
        if getattr(self, "_density_profile", None) == profile:
            return
        self._density_profile = profile

        cfg = {
            "compact": {
                "root_spacing": 2,
                "model": "compact",
                "group_title": 14,
                "group_mt": 8,
                "group_pt": 10,
                "stat_title": 15,
                "stat_card_h": 44,
                "stat_title_h": 15,
                "stat_value_font": 13,
                "stat_label_font": 13,
                "stat_min_h": 22,
                "analysis_title": 16,
                "analysis_mt": 6,
                "analysis_pt": 6,
                "analysis_margins": (6, 2, 6, 4),
                "analysis_vspace": 1,
                "metric_font": 14,
                "metric_h": 17,
                "side_font": 18,
                "side_h": 36,
                "combo_h": 34,
                "combo_font": 15,
                "active_font": 20,
                "active_h": 40,
                "path_font": 15,
                "path_h": 22,
                "settings_label_h": 40,
            },
            "balanced": {
                "root_spacing": 3,
                "model": "balanced",
                "group_title": 15,
                "group_mt": 9,
                "group_pt": 11,
                "stat_title": 16,
                "stat_card_h": 56,
                "stat_title_h": 18,
                "stat_value_font": 15,
                "stat_label_font": 15,
                "stat_min_h": 30,
                "analysis_title": 17,
                "analysis_mt": 10,
                "analysis_pt": 10,
                "analysis_margins": (7, 5, 7, 6),
                "analysis_vspace": 3,
                "metric_font": 16,
                "metric_h": 26,
                "side_font": 21,
                "side_h": 44,
                "combo_h": 38,
                "combo_font": 16,
                "active_font": 22,
                "active_h": 46,
                "path_font": 17,
                "path_h": 25,
                "settings_label_h": 48,
            },
            "roomy": {
                "root_spacing": 5,
                "model": "roomy",
                "group_title": 18,
                "group_mt": 14,
                "group_pt": 19,
                "stat_title": 19,
                "stat_card_h": 86,
                "stat_title_h": 24,
                "stat_value_font": 20,
                "stat_label_font": 18,
                "stat_min_h": 50,
                "analysis_title": 20,
                "analysis_mt": 14,
                "analysis_pt": 18,
                "analysis_margins": (10, 10, 10, 10),
                "analysis_vspace": 6,
                "metric_font": 20,
                "metric_h": 42,
                "side_font": 28,
                "side_h": 78,
                "combo_h": 58,
                "combo_font": 19,
                "active_font": 28,
                "active_h": 78,
                "path_font": 21,
                "path_h": 38,
                "settings_label_h": 70,
            },
        }[profile]

        def group_style(title_size, margin_top=None, padding_top=None, border="#c0c0c0", radius=10, title_offset=None):
            margin_top = cfg["group_mt"] if margin_top is None else margin_top
            padding_top = cfg["group_pt"] if padding_top is None else padding_top
            title_offset = (1 if profile in ("compact", "balanced") else -3) if title_offset is None else title_offset
            return (
                f"QGroupBox {{ border: 2px solid {border}; border-radius: {radius}px; "
                f"margin-top: {margin_top}px; padding-top: {padding_top}px; background-color: #ffffff; }} "
                f"QGroupBox::title {{ subcontrol-origin: padding; subcontrol-position: top left; "
                f"padding: 0 9px; margin-top: {title_offset}px; color: #0078d4; font-weight: bold; "
                f"font-size: {title_size}px; background-color: #f0f2f5; }}"
            )

        def combo_style(height, font_size):
            return f"QComboBox {{ min-height: 0px; max-height: {height}px; padding: 2px 8px; font-size: {font_size}px; }}"

        def button_style(font_size, height):
            return f"font-size:{font_size}px; min-height:0px; max-height:{height}px; padding:3px 8px;"

        self.root_layout.setSpacing(cfg["root_spacing"])
        self.model_panel.apply_density(cfg["model"])

        self.stat_box.setStyleSheet(group_style(cfg["stat_title"], 8, 10, title_offset=1))
        self.stat_layout.setContentsMargins(5, 14 if profile == "compact" else 10, 5, 5)
        self.stat_layout.setSpacing(3 if profile == "compact" else 4)
        for card in (self.card_total, self.card_defect, self.card_count, self.card_fps, self.card_avg, self.card_last):
            card.setMinimumHeight(cfg["stat_card_h"])
            card.setMaximumHeight(cfg["stat_card_h"])
        for lbl in self.stat_value_labels:
            lbl.setMinimumHeight(cfg["stat_min_h"])
            lbl.setMaximumHeight(cfg["stat_min_h"])
            font = QFont("Microsoft YaHei UI", cfg["stat_value_font"])
            font.setBold(True)
            lbl.setFont(font)
            lbl.setStyleSheet(f"font-size:{cfg['stat_value_font']}px; color:#0078d4; background:transparent;")
        for lbl in self.stat_title_labels:
            lbl.setMinimumHeight(cfg["stat_title_h"])
            lbl.setMaximumHeight(cfg["stat_title_h"])
            lbl.setStyleSheet(f"font-size:{cfg['stat_label_font']}px; color:#5a5a5a; font-weight:bold; background:transparent;")

        self.analysis_group.setStyleSheet(group_style(
            cfg["analysis_title"], cfg["analysis_mt"], cfg["analysis_pt"], "#b8c7d9", 11
        ))
        self.analysis_layout.setContentsMargins(*cfg["analysis_margins"])
        self.analysis_layout.setHorizontalSpacing(5 if profile == "compact" else 7)
        self.analysis_layout.setVerticalSpacing(cfg["analysis_vspace"])
        for lbl in self.analysis_metric_labels:
            lbl.setMinimumHeight(cfg["metric_h"])
            lbl.setMaximumHeight(cfg["metric_h"])
            lbl.setStyleSheet(f"font-size:{cfg['metric_font']}px; font-weight:bold; color:#172033; padding:1px 6px; background:#f7faff; border:1px solid #d1ddea; border-radius:6px;")
        for btn in (self.btn_analysis_summary, self.btn_analysis_mark, self.btn_config, self.btn_scratch, self.btn_bolt):
            btn.setFixedHeight(cfg["side_h"])
            btn.setStyleSheet(button_style(cfg["side_font"], cfg["side_h"]))

        for group in (self.std_group, self.obj_group, self.auto_group, self.settings_group):
            group.setStyleSheet(group_style(cfg["group_title"]))
        for layout in (self.std_layout, self.auto_layout, self.settings_layout):
            layout.setContentsMargins(5, 6, 5, 5)
            layout.setSpacing(3 if profile == "compact" else 5)
        self.obj_layout.setContentsMargins(5, 6, 5, 5)
        self.obj_layout.setSpacing(5 if profile == "compact" else 7)

        for combo in (self.std_combo, self.auto_save_type_combo):
            combo.setFixedHeight(cfg["combo_h"])
            combo.setStyleSheet(combo_style(cfg["combo_h"], cfg["combo_font"]))
        self.auto_save_path_btn.setFixedHeight(cfg["side_h"])
        self.auto_save_path_btn.setStyleSheet(button_style(cfg["side_font"], cfg["side_h"]))
        self.auto_save_path_label.setMinimumHeight(cfg["path_h"])
        self.auto_save_path_label.setStyleSheet(f"color:#5a5a5a; font-size:{cfg['path_font']}px; font-weight:bold; padding:2px 4px; background:#f5f7fa; border-radius:4px;")
        self.lbl_current_settings.setStyleSheet(f"color:#5a5a5a; font-size:{cfg['path_font']}px; font-weight:bold; padding:3px; background:#f5f5f5; border-radius:6px;")
        self.lbl_current_settings.setMinimumHeight(cfg["settings_label_h"])
        self.lbl_current_settings.setMaximumHeight(cfg["settings_label_h"])
        self.btn_open_settings.setFixedHeight(cfg["active_h"])
        self.btn_open_settings.setStyleSheet(button_style(cfg["active_font"], cfg["active_h"]))
        self.updateGeometry()

    def eventFilter(self, obj, event):
        """拦截滚轮事件，阻止其传递到滑块控件"""
        if event.type() == QtCore.QEvent.Wheel:
            # 拦截滚轮事件，不传递给滑块
            return True
        return super().eventFilter(obj, event)

    def on_auto_save_toggled(self, checked):
        self.auto_save_path_btn.setEnabled(checked)
        self.auto_save_type_combo.setEnabled(checked)
        if checked:
            self.auto_save_path_btn.setObjectName("")
            self.auto_save_path_btn.setStyle(self.auto_save_path_btn.style())
            # 如果没有指定目录，显示自动路径提示
            if not self.auto_save_dir:
                self.auto_save_path_label.setText("自动: 源目录同级/dect_result")
        else:
            self.auto_save_path_btn.setObjectName("disabled_btn")
            self.auto_save_path_btn.setStyle(self.auto_save_path_btn.style())
        # 联动：自动保存未勾选时，禁止产线分析按钮
        self.btn_analysis_summary.setEnabled(checked)
        self.btn_analysis_mark.setEnabled(checked)
        if not checked:
            self._set_button_inline_active(self.btn_analysis_summary, False)
            self._set_button_inline_active(self.btn_analysis_mark, False)
            self.btn_analysis_summary.setStyleSheet("")
            self.btn_analysis_summary.setStyle(self.btn_analysis_summary.style())
            self.btn_analysis_mark.setStyleSheet("")
            self.btn_analysis_mark.setStyle(self.btn_analysis_mark.style())

    def _on_auto_save_type_changed(self, index):
        """自动留存类型切换: 0=不合格, 1=合格, 2=所有"""
        self.auto_save_type = index  # 0=不合格, 1=合格, 2=所有

    def attach_log_widget(self, log_edit, btn_export, btn_clear):
        self.log_edit = log_edit
        btn_export.clicked.connect(self.export_csv)
        btn_clear.clicked.connect(self.clear_log)

    def clear_log(self):
        """清除日志"""
        if self.log_edit:
            self.log_edit.clear()
            self._log_rows = []
            self._event_rows = []

    def _analysis_text(self):
        total = max(0, self._total)
        pass_count = max(0, self.qualified)
        yield_rate = (pass_count / total * 100) if total else 0.0
        avg_defects = (self._total_defects / total) if total else 0.0
        recent = self._recent_passed[-20:]
        recent_fail_rate = ((len(recent) - sum(recent)) / len(recent) * 100) if recent else 0.0
        recent_diameters = self._defect_diameters[-50:]
        avg_diameter = (sum(recent_diameters) / len(recent_diameters)) if recent_diameters else 0.0
        health = max(0, min(100, 100 - recent_fail_rate * 0.6 - self._fail_streak * 8 - max(0, avg_defects - 1) * 5))
        if health >= 85:
            level = "稳定"
        elif health >= 70:
            level = "关注"
        else:
            level = "预警"
        return yield_rate, avg_defects, recent_fail_rate, avg_diameter, health, level

    def _current_batch_no(self):
        return max(1, (max(1, self._total) - 1) // max(1, self._batch_size) + 1)

    def _ensure_batch_stats(self, batch_no):
        return self._batch_stats.setdefault(batch_no, {
            "samples": 0,
            "failed": 0,
            "defects": 0,
            "manual_review": False,
            "suggested_review": False,
        })

    def _append_event_row(self, event_type, batch_no, note):
        self._event_rows.append([
            datetime.now().strftime("%H:%M:%S"),
            event_type,
            batch_no,
            note,
        ])

    def _set_button_inline_active(self, button, active, color="#d13438"):
        if active:
            button.setStyleSheet(
                f"background:{color}; color:white; border:2px solid {color}; "
                "border-radius:8px; font-size:16px; font-weight:bold;"
            )
        else:
            button.setStyleSheet("")
            button.setStyle(button.style())

    def _pulse_summary_button(self):
        self._set_button_inline_active(self.btn_analysis_summary, True, "#107c10")
        QTimer.singleShot(1600, lambda: self._set_button_inline_active(self.btn_analysis_summary, False))

    def _refresh_batch_mark_button(self):
        batch_no = self._current_batch_no()
        batch = self._ensure_batch_stats(batch_no)
        self._set_button_inline_active(self.btn_analysis_mark, bool(batch.get("manual_review")), "#d13438")

    def refresh_analysis(self):
        yield_rate, avg_defects, recent_fail_rate, avg_diameter, health, level = self._analysis_text()
        self.lbl_yield_rate.setText(f"合格率: {yield_rate:.1f}%")
        self.lbl_fail_streak.setText(f"连续异常: {self._fail_streak}")
        self.lbl_avg_defects.setText(f"平均缺陷: {avg_defects:.2f}")
        self.lbl_line_health.setText(f"工位评分: {health:.0f}/{level}")
        self.lbl_recent_fail.setText(f"近20异常: {recent_fail_rate:.1f}%")
        self.lbl_size_avg.setText(f"尺寸均值: {avg_diameter:.1f}mm")

    def append_analysis_snapshot(self):
        yield_rate, avg_defects, recent_fail_rate, avg_diameter, health, level = self._analysis_text()
        batch_no = self._current_batch_no()
        text = (
            f"[产线分析] 样本 {self._total} | 合格率 {yield_rate:.1f}% | "
            f"连续异常 {self._fail_streak} | 近20异常 {recent_fail_rate:.1f}% | "
            f"平均缺陷 {avg_defects:.2f} | 尺寸均值 {avg_diameter:.1f}mm | 工位评分 {health:.0f}/{level}"
        )
        self._append_event_row("分析摘要", batch_no, text)
        if self.log_edit:
            self.log_edit.append(f"<span style='color:#0078d4; font-weight:bold;'>{text}</span>")
        self._pulse_summary_button()
        # 弹出图表化摘要对话框
        dlg = AnalysisSummaryDialog(self.window() if self.window() else self, self)
        dlg.exec_()

    def mark_abnormal_batch(self):
        batch_no = self._current_batch_no()
        batch = self._ensure_batch_stats(batch_no)
        batch["manual_review"] = not bool(batch.get("manual_review"))
        review = batch["manual_review"]
        status = "需人工复检" if review else "正常"
        text = (
            f"[人工标记] {datetime.now().strftime('%H:%M:%S')} "
            f"批次 {batch_no} {status}: 连续异常 {self._fail_streak}, "
            f"批内异常 {batch.get('failed', 0)}, 批内缺陷 {batch.get('defects', 0)}"
        )
        batch_folder_name = f"Batch_{batch_no:03d}_{status}"
        old_folder = batch.get("batch_folder", "")

        if review and not old_folder:
            # 标记 ON：整理已保存文件到批次文件夹
            saved_files = batch.get("saved_files", [])
            if saved_files:
                dect_dir = os.path.dirname(saved_files[0])
                batch_dir = os.path.join(dect_dir, batch_folder_name)
                os.makedirs(batch_dir, exist_ok=True)
                for f in saved_files:
                    if os.path.exists(f):
                        shutil.copy2(f, os.path.join(batch_dir, os.path.basename(f)))
                batch["batch_folder"] = batch_dir
                if self.log_edit:
                    self.log_edit.append(
                        f"<span style='color:#d13438;'>  → 已整理至: {batch_dir}</span>"
                    )
        elif old_folder and os.path.exists(old_folder):
            # 标记 OFF：重命名文件夹（需人工复检 → 正常）
            parent = os.path.dirname(old_folder)
            new_dir = os.path.join(parent, batch_folder_name)
            if old_folder != new_dir:
                os.rename(old_folder, new_dir)
                batch["batch_folder"] = new_dir
                if self.log_edit:
                    self.log_edit.append(
                        f"<span style='color:#107c10;'>  → 已更新为: {new_dir}</span>"
                    )

        self._append_event_row("异常批次标记", batch_no, text)
        if self.log_edit:
            self.log_edit.append(f"<span style='color:#d13438; font-weight:bold;'>{text}</span>")
        self._refresh_batch_mark_button()

    def select_auto_save_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "指定留存目录")
        if dir_path:
            self.auto_save_dir = dir_path
            self.auto_save_path_label.setText(dir_path)
        else:
            self.auto_save_dir = ""
            self.auto_save_path_label.setText("未选择目录")

    def get_auto_save_dir(self, batch_source_dir=None):
        """获取有效的自动留存目录。
        如果用户已指定目录则用指定目录，否则自动生成。
        """
        if self.auto_save_dir:
            return self.auto_save_dir
        base = batch_source_dir or os.getcwd()
        return os.path.join(base, "dect_result")

    def get_effective_save_dir_label(self, batch_source_dir=None):
        """返回目录描述文本（用于界面显示）"""
        if self.auto_save_dir:
            return self.auto_save_dir
        base = batch_source_dir or os.getcwd()
        return os.path.join(base, "dect_result") + " (自动)"

    def is_auto_save_enabled(self, passed=None, batch_source_dir=None):
        """检查是否应该保存样本
        passed: True=合格, False=不合格, None=未判定
        """
        if not self.auto_save_cb.isChecked():
            return False
        # 有显式目录，或 batch_source_dir 可自动生成时，才视为可用
        save_dir = self.get_auto_save_dir(batch_source_dir)
        if not save_dir or not os.path.exists(save_dir):
            return False
        # 根据留存类型判断
        if self.auto_save_type == 0:  # 留存不合格
            return passed == False
        elif self.auto_save_type == 1:  # 留存合格
            return passed == True
        else:  # 留存所有
            return True

    def open_settings_dialog(self):
        """打开检测设置弹出对话框"""
        dlg = DetectionSettingsDialog(
            self,
            conf=self._conf,
            iou=self._iou,
            grayscale=self._gray,
            exposure=self._exposure,
            filter_settings=self.filter_settings
        )
        dlg.settings_changed.connect(self.on_settings_changed)
        dlg.exec_()

    def on_settings_changed(self, conf, iou, grayscale, exposure, filter_settings=None):
        """设置变更回调"""
        self._conf = conf
        self._iou = iou
        self._gray = grayscale
        self._exposure = exposure
        if filter_settings is not None:
            self.filter_settings = merge_filter_settings(filter_settings)
            self.filter_engine.update_settings(self.filter_settings)
            self._last_filter_info = {}
        self.lbl_current_settings.setText(self.settings_label_text())
        if self.log_edit:
            gray_str = "开启" if grayscale else "关闭"
            filter_str = self.filter_summary(short=False)
            self.log_edit.append(
                f"<span style='color:#107c10'>[系统] 检测设置已更新: 置信度={conf:.2f}, IoU={iou:.2f}, 灰度={gray_str}, 曝光={exposure:.2f}× | {filter_str}</span>"
            )

    def load_default_standards(self):
        # 默认使用屏幕划痕标准
        default_std = InspectionStandard("GB 11614-2022 平板玻璃", 3, 5.0, 0.2, "scratch", 2.0)
        self.standards = [default_std]
        self.current_standard = default_std
        self.refresh_std_combo()

    def refresh_std_combo(self):
        self.std_combo.blockSignals(True)
        self.std_combo.clear()
        for s in self.standards:
            self.std_combo.addItem(s.name)
        if self.current_standard and self.current_standard in self.standards:
            self.std_combo.setCurrentIndex(self.standards.index(self.current_standard))
        else:
            self.std_combo.setCurrentIndex(0)
            self.current_standard = self.standards[0]
        self.std_combo.blockSignals(False)
        self.standard_changed.emit(self.current_standard, self.pixel_per_cm)

    def _on_std_changed(self, idx):
        if 0 <= idx < len(self.standards):
            self.current_standard = self.standards[idx]
            self.standard_changed.emit(self.current_standard, self.pixel_per_cm)

    def _on_object_btn_clicked(self, idx):
        """检测对象切换: 0=划痕, 1=铝合金零件"""
        if idx == 0:
            self.btn_scratch.setChecked(True)
            self.btn_scratch.setObjectName("btn_active")
            self.btn_bolt.setChecked(False)
            self.btn_bolt.setObjectName("btn_side_compact")
            # 屏幕划痕标准
            scratch_std = InspectionStandard("GB 11614-2022 平板玻璃", 3, 2.0, 0.2, "scratch", 5.0)
            self.standards = [scratch_std]
            self.current_standard = scratch_std
            obj_name = "划痕"
        else:
            self.btn_bolt.setChecked(True)
            self.btn_bolt.setObjectName("btn_active")
            self.btn_scratch.setChecked(False)
            self.btn_scratch.setObjectName("btn_side_compact")
            # 铝合金零件标准
            alloy_std = InspectionStandard("GB/T 6892-2023 铝合金零件", 2, 1.0, 0.2, "aluminum_alloy", 2.0)
            self.standards = [alloy_std]
            self.current_standard = alloy_std
            obj_name = "铝合金零件"
        # 更新按钮样式
        self.btn_scratch.setStyle(self.btn_scratch.style())
        self.btn_bolt.setStyle(self.btn_bolt.style())
        self.refresh_std_combo()
        if self.log_edit:
            self.log_edit.append(
                f"<span style='color:#107c10'>[系统] 检测对象已切换至：{obj_name}</span>")

    def open_std_config(self):
        dlg = StandardConfigDialog(self)
        dlg.set_config(self.standards, self.standards.index(self.current_standard) if self.current_standard else 0,
                       self.pixel_per_cm, self.detect_height)
        if dlg.exec_() == QDialog.Accepted:
            self.standards, sel_idx, self.pixel_per_cm, self.detect_height = dlg.get_config()
            self.current_standard = self.standards[sel_idx] if 0 <= sel_idx < len(self.standards) else self.standards[0]
            self.refresh_std_combo()
            if self.log_edit:
                self.log_edit.append(
                    f"<span style='color:#107c10'>[系统] 评判标准已更新至：{self.current_standard.name}</span>")

    def get_conf(self):
        return self._conf

    def get_iou(self):
        return self._iou

    def get_grayscale(self):
        return self._gray

    def get_exposure(self):
        return self._exposure

    def get_filter_settings(self):
        return dict(self.filter_settings)

    def filter_summary(self, short=True):
        s = merge_filter_settings(self.filter_settings)
        if short:
            return (
                f"过滤: ROI{'开' if s['roi_enabled'] else '关'} / "
                f"稳帧{'开' if s['stable_enabled'] else '关'} / "
                f"计数{'开' if s['count_enabled'] else '关'} / "
                f"变速{'开' if s['speed_enabled'] else '关'}"
            )
        parts = [
            f"ROI={'开' if s['roi_enabled'] else '关'}",
            f"稳帧={'开' if s['stable_enabled'] else '关'}(N={s['stable_frames']})",
            f"计数={'开' if s['count_enabled'] else '关'}",
            f"速度={'开' if s['speed_enabled'] else '关'}({s['line_speed']:.1f}m/min)",
        ]
        return "高级过滤 " + " / ".join(parts)

    def _settings_badge(self, name, enabled):
        bg = "#107c10" if enabled else "#6b7280"
        text = "开" if enabled else "关"
        return f"<span style='background:{bg}; color:white; padding:2px 6px; border-radius:5px;'>{name}{text}</span>"

    def settings_label_text(self):
        s = merge_filter_settings(self.filter_settings)
        badges = " ".join([
            self._settings_badge("ROI", s["roi_enabled"]),
            self._settings_badge("稳帧", s["stable_enabled"]),
            self._settings_badge("计数", s["count_enabled"]),
            self._settings_badge("变速", s["speed_enabled"]),
        ])
        return f"置信度: {self._conf:.2f} | IoU: {self._iou:.2f}<br>{badges}"

    def preprocess(self, img):
        return apply_preprocess(img, self._gray, self._exposure)

    def apply_detection_filters(self, results, image_shape, source="image", temporal=True):
        filtered, info = self.filter_engine.filter(results, image_shape, source=source, temporal=temporal)
        self._last_filter_info = info
        return filtered, info

    def reset_filter_state(self, source=None):
        self.filter_engine.reset(source)
        self._last_filter_info = {}

    def filter_status_text(self, info=None):
        info = info or self._last_filter_info or {}
        if not info:
            return ""
        parts = []
        if info.get("roi_removed", 0):
            parts.append(f"ROI丢弃 {info['roi_removed']}")
        if info.get("unstable_removed", 0):
            parts.append(f"稳帧过滤 {info['unstable_removed']}")
        if info.get("count_events", 0):
            parts.append(f"新增计数 {info['count_events']}")
        if self.filter_settings.get("count_enabled", False):
            parts.append(f"累计计数 {info.get('total_count', 0)}")
        if self.filter_settings.get("stable_enabled", False):
            parts.append(f"确认帧 {info.get('effective_stable_frames', self.filter_settings.get('stable_frames', 20))}")
        return " | ".join(parts)

    def update_stats(self, results, passed=None, oversized=None, defect_details=None, lethal_count=0, realtime=False, fps_override=None):
        self._total += 1
        now = time.time()
        log_interval = max(0.1, float(os.getenv("UI_REALTIME_LOG_INTERVAL", "1.0")))
        stats_ui_interval = max(0.03, float(os.getenv("UI_REALTIME_STATS_INTERVAL", "0.25")))
        should_log_realtime = (not realtime) or (now - self._last_realtime_log_time >= log_interval)
        should_refresh_stats_ui = (not realtime) or (now - self._last_realtime_stats_ui_time >= stats_ui_interval)
        if realtime and should_log_realtime:
            self._last_realtime_log_time = now
        if realtime and should_refresh_stats_ui:
            self._last_realtime_stats_ui_time = now

        # FPS显示为滑动平均检测帧率，避免切换模式或瞬时抖动导致误判。
        if self._fps_last_time is None:
            self._fps_last_time = now
            fps = 0.0
        else:
            interval = max(0.001, now - self._fps_last_time)
            self._fps_last_time = now
            self._fps_times.append(interval)
            self._fps_times = self._fps_times[-30:]
            fps = len(self._fps_times) / max(0.001, sum(self._fps_times))
        if fps_override is not None and fps_override > 0:
            fps = float(fps_override)

        n = len(results)
        ts = datetime.now().strftime("%H:%M:%S")

        valid_defects = len(defect_details) if defect_details else 0
        if defect_details:
            self._defect_diameters.extend(float(d.get('diameter', 0.0)) for d in defect_details)
            self._defect_diameters = self._defect_diameters[-300:]

        if passed is not None:
            self._recent_passed.append(bool(passed))
            self._recent_passed = self._recent_passed[-50:]
            if passed:
                self.qualified += 1
                self._fail_streak = 0
            else:
                self.unqualified += 1
                self._fail_streak += 1

        avg = (sum(r[4] for r in results) / n) if n else 0.0
        batch_no = self._current_batch_no()
        batch = self._ensure_batch_stats(batch_no)
        batch["samples"] += 1
        batch["defects"] += valid_defects
        if passed is False:
            batch["failed"] += 1
        batch["suggested_review"] = bool(batch["failed"] > 0 or self._fail_streak >= 3)
        status_text = "未判定" if passed is None else ("通过" if passed else "不合格")
        review_text = "人工复检" if batch.get("manual_review") else ("建议复检" if batch.get("suggested_review") else "")
        if should_log_realtime:
            self._log_rows.append([
                ts,
                self._total,
                batch_no,
                valid_defects,
                f"{avg:.2f}",
                status_text,
                review_text,
            ])

        if n:
            self._defect_frames += 1
            self._total_defects += valid_defects
            if self.log_edit and should_log_realtime:
                if passed is not None:
                    status_str = "通过" if passed else "不合格"
                    color = "#d13438" if not passed else "#107c10"
                    detect_obj = self.current_standard.detect_object if self.current_standard else "scratch"
                    obj_name = "划痕" if detect_obj == "scratch" else "铝合金零件"

                    # 聚集检测
                    clusters = check_defect_clustering(defect_details, self.pixel_per_cm) if defect_details else []

                    # 增强置信度显示样式 - 使用高亮背景气泡框，醒目黑字
                    conf_bg = "#FFEB3B" if avg >= 0.7 else "#FF9800" if avg >= 0.5 else "#F44336"
                    conf_html = f"<span style='background:{conf_bg}; color:black; padding:2px 8px; border-radius:6px; font-weight:bold; font-size:18px;'>{avg:.2f}</span>"

                    self.log_edit.append(
                        f"<span style='color:#ffb900;'>[{ts}]</span> "
                        f"样本 #{self._total} 原始缺陷 {n} 处，有效缺陷 {valid_defects} 处 "
                        f"| <span style='color:{color}; font-weight:bold;'>{status_str}</span> "
                        f"<span style='font-weight:bold;'>置信度 {conf_html}</span> "
                        f"<span style='color:#5a5a5a;'>(超标: {oversized}/{self.current_standard.max_defects if self.current_standard else '?'}, "
                        f"致命({self.current_standard.lethal_mm}mm以上): {lethal_count})</span>"
                    )
                    if defect_details:
                        for i, d in enumerate(defect_details):
                            size_color = "#ff8c00" if d['diameter'] > self.current_standard.size_threshold_mm else "#5a5a5a"
                            lethal_color = "#d13438" if d['diameter'] > self.current_standard.lethal_mm else size_color
                            # 每个缺陷的置信度单独高亮显示 - 醒目的气泡样式
                            d_conf = results[i][4] if i < len(results) else 0.0
                            d_conf_bg = "#FFEB3B" if d_conf >= 0.7 else "#FF9800" if d_conf >= 0.5 else "#F44336"
                            d_conf_html = f"<span style='background:{d_conf_bg}; color:black; padding:1px 6px; border-radius:4px; font-weight:bold; font-size:16px;'>{d_conf:.2f}</span>"
                            self.log_edit.append(
                                f"  <span style='color:{lethal_color};'>→ 缺陷{i+1}: 等效直径 {d['diameter']:.2f}mm "
                                f"(宽{d['w_mm']:.1f}×高{d['h_mm']:.1f}) [像素: {d['w_px']:.0f}×{d['h_px']:.0f}] "
                                f"<span style='font-weight:bold;'>置信度 {d_conf_html}</span></span>"
                            )
                    if clusters:
                        cluster_info = "; ".join([f"簇{idx+1}({len(c)}个)" for idx, c in enumerate(clusters)])
                        self.log_edit.append(
                            f"  <span style='color:#8764b8;'>⚠ 聚集缺陷: {cluster_info}</span>"
                        )
                else:
                    self.log_edit.append(
                        f"<span style='color:#ffb900;'>[{ts}]</span> "
                        f"样本 #{self._total} 发现 <b style='color:#d13438;'>{valid_defects}</b> 处有效缺陷 (原始{ n })"
                    )
        else:
            if passed is not None and self.log_edit and should_log_realtime:
                color = "#d13438" if not passed else "#107c10"
                self.log_edit.append(
                    f"<span style='color:#ffb900;'>[{ts}]</span> "
                    f"样本 #{self._total} 无明显缺陷 | <span style='color:{color}; font-weight:bold;'>{'通过' if passed else '不合格'}</span>"
                )

        if should_refresh_stats_ui:
            self.lv_total.setText(str(self._total))
            self.lv_defect.setText(str(self._defect_frames))
            self.lv_count.setText(str(self._total_defects))
            self.lv_fps.setText(f"{fps:.1f}")
            fit_label_font(self.lv_total)
            fit_label_font(self.lv_defect)
            fit_label_font(self.lv_count)
            fit_label_font(self.lv_fps)
            self.refresh_analysis()
            self._refresh_batch_mark_button()

    def reset_stats(self):
        self._total = self._defect_frames = self._total_defects = 0
        self.qualified = 0
        self.unqualified = 0
        self._fps_times = []
        self._fps_start_time = None
        self._fps_frame_count = 0
        self._fps_last_time = None
        self._log_rows = []
        self._event_rows = []
        self._batch_stats = {}
        self._last_realtime_log_time = 0.0
        self._last_realtime_stats_ui_time = 0.0
        self._recent_passed = []
        self._fail_streak = 0
        self._defect_diameters = []
        self.reset_filter_state()
        if self.log_edit:
            self.log_edit.clear()
        for lb in (self.lv_total, self.lv_defect, self.lv_count):
            lb.setText("0")
        self.lv_fps.setText("—")
        self.refresh_analysis()
        self._refresh_batch_mark_button()

    def append_summary(self, total_samples, total_time=0.0, avg_per_image_io=0.0, avg_per_image_infer=0.0):
        """批量/视频结束时输出总结"""
        summary = f"[总结] 检测样本数量: {total_samples} | 合格: {self.qualified} | 不合格: {self.unqualified}"
        self._append_event_row("任务总结", self._current_batch_no(), summary)
        if self.log_edit:
            self.log_edit.append(f"<span style='color:#0078d4; font-weight:bold;'>{summary}</span>")
            if total_time > 0:
                total_str = f"{total_time * 1000:.0f}ms" if total_time < 1 else f"{total_time:.1f}s"
                throughput = total_samples / total_time if total_time > 0 else 0
                time_summary = (
                    f"[吞吐量测试] "
                    f"总耗时(全流程): {total_str} | "
                    f"吞吐量: {throughput:.2f} 张/s | "
                    f"平均每张(全流程): {avg_per_image_io:.1f}ms | "
                    f"平均每张(纯推理): {avg_per_image_infer:.1f}ms"
                )
                self._append_event_row("吞吐量测试", self._current_batch_no(), time_summary)
                self.log_edit.append(f"<span style='color:#0078d4; font-weight:bold;'>{time_summary}</span>")

    def export_csv(self):
        if not self._log_rows and not self._event_rows and not self._batch_stats:
            QMessageBox.information(None, "系统提示", "当前无有效检测记录")
            return
        path, _ = QFileDialog.getSaveFileName(
            None, "导出检测日志", f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "CSV 报表 (*.csv)")
        if path:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["检测明细"])
                writer.writerow(["时间", "样本序号", "批次", "缺陷总数", "均值置信度", "判定", "复检状态"])
                writer.writerows(self._log_rows)
                writer.writerow([])
                writer.writerow(["产线事件"])
                writer.writerow(["时间", "事件类型", "批次", "内容"])
                writer.writerows(self._event_rows)
                writer.writerow([])
                writer.writerow(["批次汇总"])
                writer.writerow(["批次", "批大小", "样本数", "不合格数", "缺陷累计", "是否建议复检", "是否人工标记复检"])
                for batch_no in sorted(self._batch_stats):
                    batch = self._batch_stats[batch_no]
                    writer.writerow([
                        batch_no,
                        self._batch_size,
                        batch.get("samples", 0),
                        batch.get("failed", 0),
                        batch.get("defects", 0),
                        "是" if batch.get("suggested_review") else "否",
                        "是" if batch.get("manual_review") else "否",
                    ])
            QMessageBox.information(None, "操作成功", f"日志报表已保存至:\n{path}")


# ══════════════════════════════════════════════════════════════════
#  DirectShow 原生采集
# ══════════════════════════════════════════════════════════════════
class DirectShowNativeCapture:
    """Use DirectShow IAMStreamConfig + SampleGrabber when OpenCV ignores FOURCC."""

    def __init__(self, index, width, height, fps, fourcc="MJPG"):
        self.index = int(index)
        self.width = int(width)
        self.height = int(height)
        self.target_fps = float(fps or 0)
        self.fourcc = (fourcc or "MJPG").strip().upper()
        self.graph = None
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._frame = None
        self._opened = False
        self._format = None
        self._last_error = ""
        self._open()

    @property
    def last_error(self):
        return self._last_error

    def _on_frame(self, img):
        with self._lock:
            self._frame = img.copy()
        self._event.set()

    def _format_score(self, fmt):
        fmt_name = str(fmt.get("media_type_str", "")).upper()
        w = int(fmt.get("width", 0) or 0)
        h = int(fmt.get("height", 0) or 0)
        max_fps = float(fmt.get("max_framerate", 0) or 0)
        fmt_alias = {"MJPEG"} if self.fourcc == "MJPG" else set()
        fmt_penalty = 0 if fmt_name in ({self.fourcc} | fmt_alias) else 10000
        size_penalty = 0 if (w == self.width and h == self.height) else abs(w - self.width) + abs(h - self.height) + 1000
        fps_penalty = abs(max_fps - self.target_fps) if self.target_fps > 0 else 0
        if self.target_fps > 0 and max_fps < self.target_fps:
            fps_penalty += 100
        return fmt_penalty + size_penalty + fps_penalty - max_fps * 0.01

    def _open(self):
        try:
            from pygrabber.dshow_graph import FilterGraph, FilterType
        except Exception as exc:
            self._last_error = f"pygrabber/comtypes 不可用: {exc}"
            return
        try:
            graph = FilterGraph()
            graph.add_video_input_device(self.index)
            video = graph.filters[FilterType.video_input]
            formats = video.get_formats()
            aliases = {self.fourcc}
            if self.fourcc == "MJPG":
                aliases.add("MJPEG")
            candidates = [fmt for fmt in formats if str(fmt.get("media_type_str", "")).upper() in aliases]
            if not candidates:
                self._last_error = f"DirectShow 未枚举到 {self.fourcc} 格式"
                return
            selected = min(candidates, key=self._format_score)
            video.set_format(selected["index"])
            graph.add_sample_grabber(self._on_frame)
            graph.add_null_render()
            graph.prepare_preview_graph()
            graph.run()
            self.graph = graph
            self._format = selected
            self._opened = True
        except Exception as exc:
            self._last_error = f"DirectShow 原生采集初始化失败: {exc}"
            logger.exception("DirectShow native capture initialization failed")
            self.release()

    def isOpened(self):
        return self._opened

    def read(self):
        if not self._opened or self.graph is None:
            return False, None
        self._event.clear()
        try:
            self.graph.grab_frame()
        except Exception as exc:
            self._last_error = f"DirectShow 取帧失败: {exc}"
            return False, None
        if not self._event.wait(0.8):
            self._last_error = "DirectShow 取帧超时"
            return False, None
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
        return frame is not None, frame

    def release(self):
        self._opened = False
        graph = self.graph
        self.graph = None
        if graph is not None:
            try:
                graph.stop()
            except Exception:
                pass
            try:
                graph.remove_filters()
            except Exception:
                pass

    def set(self, prop, value):
        return False

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float((self._format or {}).get("width", self.width))
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float((self._format or {}).get("height", self.height))
        if prop == cv2.CAP_PROP_FPS:
            return float((self._format or {}).get("max_framerate", self.target_fps))
        if prop == cv2.CAP_PROP_FOURCC:
            return float(cv2.VideoWriter_fourcc(*self.fourcc[:4]))
        if prop == cv2.CAP_PROP_BUFFERSIZE:
            return 1.0
        return 0.0

    def getBackendName(self):
        return "DSHOW_NATIVE"


# ══════════════════════════════════════════════════════════════════
#  状态栏辅助 Mixin
# ══════════════════════════════════════════════════════════════════
class StatusBarMixin:
    def open_camera_driver_settings(self):
        cap = getattr(self, "cap", None)
        if cap is None or not hasattr(self, "_configure_capture"):
            QMessageBox.information(self, "相机设置", "当前页面没有可配置的摄像头。")
            return
        if hasattr(self, "_stop_pipeline"):
            self._stop_pipeline()
        timer = getattr(self, "timer", None)
        if timer is not None:
            timer.stop()
        if cap.isOpened():
            cap.release()
        camera_index = int(getattr(self, "camera_index", 0))
        settings_cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not settings_cap.isOpened():
            settings_cap.release()
            QMessageBox.warning(self, "相机设置", "无法通过 DirectShow 打开摄像头设置页。")
            return
        self._configure_capture(settings_cap, getattr(self, "camera_fourcc", "MJPG") or "MJPG")
        opened = bool(settings_cap.set(cv2.CAP_PROP_SETTINGS, 1))
        settings_cap.release()
        self.cap = cv2.VideoCapture()
        if opened:
            self.lbl_status.setText("相机驱动设置已关闭，请重新连接摄像头。建议在 AMCAP 的 Capture Format 中选择 MJPG。")
        else:
            QMessageBox.information(
                self,
                "相机设置",
                "当前驱动没有向 OpenCV 暴露设置页。可继续用 AMCAP 切到 MJPG，或使用厂商 SDK 方式接入。",
            )

    def _apply_status_style(self, text, bg, border, fg):
        style_key = (text, bg, border, fg)
        if getattr(self, "_last_status_style_key", None) == style_key:
            return
        self._last_status_style_key = style_key
        self.status_text_label.setText(text)
        self.status_text_label.setStyleSheet(
            f"font-size:18px; font-weight:bold; background:{bg}; "
            f"border:2px solid {border}; border-radius:12px; padding:8px 12px; color:{fg}; letter-spacing:1px;"
        )

    def _set_status_neutral(self, text="系统待机状态"):
        self._apply_status_style(text, "#ffffff", "#c0c0c0", "#5a5a5a")

    def _set_status_pass(self):
        self._apply_status_style("当前状态: 检测通过", "#e8f5e9", "#4caf50", "#107c10")

    def _set_status_fail(self):
        self._apply_status_style("当前状态: 检测不合格", "#ffebee", "#d13438", "#d13438")

    def _make_status_bar(self):
        wrap = QWidget()
        layout = QHBoxLayout(wrap)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.status_text_label = QLabel("系统待机状态")
        self.status_text_label.setAlignment(Qt.AlignCenter)
        self.status_text_label.setFixedHeight(42)
        self.status_text_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.backend_badge = QLabel("后端: 未加载")
        self.backend_badge.setAlignment(Qt.AlignCenter)
        self.backend_badge.setFixedHeight(42)
        self.backend_badge.setMinimumWidth(270)
        self.backend_badge.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.backend_badge.setStyleSheet(
            "font-size:15px; font-weight:bold; background:#f5f7fa; "
            "border:2px solid #d7dde5; border-radius:12px; padding:6px 10px; color:#5a5a5a;"
        )

        self.backend_select = QComboBox()
        self.backend_select.setFixedHeight(42)
        self.backend_select.setMinimumWidth(270)
        self.backend_select.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        for label, value in BACKEND_CHOICES:
            self.backend_select.addItem(f"后端: {label}", value)
        self.backend_select.setToolTip("选择当前模型加载使用的推理后端")
        self.backend_select.setStyleSheet(
            "QComboBox { font-size:15px; font-weight:bold; background:#f5f7fa; "
            "border:2px solid #d7dde5; border-radius:12px; padding:6px 10px; color:#5a5a5a; }"
        )
        self.backend_select.currentIndexChanged.connect(self._on_backend_choice_changed)

        layout.addWidget(self.status_text_label, 1)
        layout.addWidget(self.backend_select, 0)
        self._set_status_neutral()
        return wrap

    def _on_backend_choice_changed(self, _index):
        selected = self.backend_select.currentData()
        if hasattr(self, "_backend_enabled_values") and selected not in self._backend_enabled_values:
            self._set_backend_choice(getattr(self, "_backend_current_preference", "auto"))
            return
        if hasattr(self, "ctrl") and hasattr(self.ctrl, "backend_changed"):
            self.ctrl.backend_changed.emit(selected)

    def _set_backend_choice(self, preference: str):
        if not hasattr(self, "backend_select"):
            return
        self._backend_current_preference = preference
        self._refresh_backend_combo_labels()
        idx = self.backend_select.findData(preference)
        if idx < 0:
            idx = self.backend_select.findData("auto")
        self.backend_select.blockSignals(True)
        self.backend_select.setCurrentIndex(idx)
        self.backend_select.blockSignals(False)

    def _refresh_backend_combo_labels(self):
        if not hasattr(self, "backend_select"):
            return
        runtime = getattr(self, "_backend_runtime_text", "")
        for label, value in BACKEND_CHOICES:
            text = f"后端: {label}"
            if value == "auto" and runtime:
                text = f"后端: {label}: {runtime}"
            idx = self.backend_select.findData(value)
            if idx >= 0:
                self.backend_select.setItemText(idx, text)

    def _set_backend_availability(self, availability: dict, reasons: dict = None):
        if not hasattr(self, "backend_select"):
            return
        reasons = reasons or {}
        self._backend_enabled_values = {
            value for value, enabled in availability.items() if enabled
        } | {"auto"}
        model = self.backend_select.model()
        for label, value in BACKEND_CHOICES:
            idx = self.backend_select.findData(value)
            if idx < 0:
                continue
            item = model.item(idx)
            enabled = value in self._backend_enabled_values
            item.setEnabled(enabled)
            item.setForeground(QColor("#107c10" if enabled and value != "auto" else "#2b2b2b" if enabled else "#9a9a9a"))
            item.setBackground(QColor("#e8f5e9" if enabled and value != "auto" else "#ffffff" if enabled else "#eeeeee"))
            note = reasons.get(value, "可选择" if enabled else "当前不可用")
            item.setToolTip(f"{label}: {note}")

    def _set_backend_runtime_text(self, text: str):
        if not hasattr(self, "backend_select"):
            return
        runtime = str(text).replace("后端:", "", 1).replace("鍚庣:", "", 1).strip()
        self._backend_runtime_text = runtime
        self._refresh_backend_combo_labels()

    def _set_backend_badge(self, text="后端: 未加载", color="#5a5a5a"):
        if hasattr(self, "backend_select"):
            self.backend_select.setToolTip(f"{text}\n点击下拉选择推理后端")
            self.backend_select.setStyleSheet(
                f"QComboBox {{ font-size:15px; font-weight:bold; background:#f5f7fa; "
                f"border:2px solid #d7dde5; border-radius:12px; padding:6px 10px; color:{color}; }}"
            )
        self.backend_badge.setText(text)
        self.backend_badge.setStyleSheet(
            f"font-size:15px; font-weight:bold; background:#f5f7fa; "
            f"border:2px solid #d7dde5; border-radius:12px; padding:6px 10px; color:{color};"
        )

    def _make_display_label(self, text):
        return AspectRatioDisplay(text)

    def _make_separator(self):
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background:#c8c8c8; margin:2px 0;")
        return sep

    def _make_section_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("color:#5a5a5a; font-size:14px; font-weight:bold; letter-spacing:1px;")
        return lbl


# ══════════════════════════════════════════════════════════════════
#  Tab 类
# ══════════════════════════════════════════════════════════════════
class ImageTab(StatusBarMixin, QWidget):
    def __init__(self, det_ref: list, ctrl: ControlPanel, parent=None):
        super().__init__(parent)
        self.det_ref = det_ref
        self.ctrl = ctrl
        self.im0_orig = None
        self._result_img = None
        self._batch_worker = None
        self._folder_files = []
        self._batch_saved_files = []
        self._avg_detect_time = 0
        self._total_detect_time = 0
        self._detect_count = 0
        self._batch_saved_files = []
        self._batch_start_cancel_requested = False
        self._suppress_batch_abort_dialog = False
        self._build_ui()
        self.ctrl.standard_changed.connect(self.on_standard_changed)
        self.current_standard = self.ctrl.current_standard
        self.pixel_per_cm = self.ctrl.pixel_per_cm

    @property
    def det(self):
        return self.det_ref[0]

    def on_standard_changed(self, std, px_per_cm):
        self.current_standard = std
        self.pixel_per_cm = px_per_cm

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(2)

        root.addWidget(self._make_status_bar())

        img_row = QHBoxLayout()
        img_row.setContentsMargins(0, 0, 0, 0)
        img_row.setSpacing(6)
        self.lbl_src = self._make_display_label("输入原始图像")
        self.lbl_dst = self._make_display_label("系统检测视图")
        img_row.addWidget(self.lbl_src)
        img_row.addWidget(self.lbl_dst)
        root.addLayout(img_row)

        root.addWidget(self._make_separator())
        root.addWidget(self._make_section_label("基础操作"))

        r1 = QHBoxLayout()
        r1.setSpacing(4)
        self.btn_open = QPushButton("载入本地图像")
        self.btn_open.setObjectName("btn_enlarged")
        self.btn_detect = QPushButton("执行单次检测")
        self.btn_detect.setObjectName("btn_enlarged")
        self.btn_save = QPushButton("保存当前结果")
        self.btn_save.setObjectName("btn_enlarged")
        self.btn_reset = QPushButton("重置运行状态")
        self.btn_reset.setObjectName("btn_enlarged")
        self.btn_detect.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.btn_open.clicked.connect(self.open_image)
        self.btn_detect.clicked.connect(self.detect_image)
        self.btn_save.clicked.connect(self.save_result)
        self.btn_reset.clicked.connect(self.reset)
        for b in (self.btn_open, self.btn_detect, self.btn_save, self.btn_reset):
            r1.addWidget(b)
        root.addLayout(r1)

        self.lbl_status = QLabel("")
        self.lbl_status.setMinimumHeight(24)
        self.lbl_status.setStyleSheet("color:#ff8c00; font-size:16px; font-weight:bold; padding:2px 0;")
        root.addWidget(self.lbl_status)

    def open_image(self):
        p, _ = QFileDialog.getOpenFileName(self, "载入图像", "",
                                           "图片 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;All Files (*)")
        if not p:
            return
        self.im0_orig, read_error = read_image_unicode(p)
        if self.im0_orig is None:
            QMessageBox.warning(self, "读取错误", "图像解码失败，请确认文件完整性")
            return
        self.lbl_src.setPixmap(
            cv2_to_qpixmap(self.ctrl.preprocess(self.im0_orig), self.lbl_src.width(), self.lbl_src.height()))
        self.lbl_dst.setText("系统检测视图")
        self.lbl_dst.setPixmap(QPixmap())
        self.lbl_status.setText(f"图像已载入: {os.path.basename(p)}")
        self.btn_detect.setEnabled(True)
        self.btn_save.setEnabled(False)
        if hasattr(self, "btn_quick_snap"):
            self.btn_quick_snap.setEnabled(False)
        self._result_img = None
        self._set_status_neutral()

    def detect_image(self):
        if self.im0_orig is None:
            self.lbl_status.setText(f"图像读取失败: {read_error}")
            return
        if not ensure_detector_available(self, self.det):
            return
        self.btn_detect.setEnabled(False)
        self.btn_detect.setText("检测中...")
        QApplication.processEvents()

        try:
            t0 = time.time()
            img = self.ctrl.preprocess(self.im0_orig)
            results, im_out = self.det.run(img, conf_thres=self.ctrl.get_conf(), iou_thres=self.ctrl.get_iou())
            raw_results = results
            results, filter_info = self.ctrl.apply_detection_filters(results, img.shape, source="image", temporal=False)
            ms = (time.time() - t0) * 1000
            # 更新检测耗时统计
            self._detect_count += 1
            self._total_detect_time += ms
            self._avg_detect_time = self._total_detect_time / self._detect_count
            self.ctrl.lbl_last_time.setText(f"{ms:.0f}ms")
            self.ctrl.lbl_avg_time.setText(f"{self._avg_detect_time:.1f}ms")
            fit_label_font(self.ctrl.lbl_last_time)
            fit_label_font(self.ctrl.lbl_avg_time)
            # 获取实际图像宽度用于动态像素当量计算
            img_h, img_w = img.shape[:2]
            oversized, passed, details, lethal, calc_log = evaluate_defects_detailed(
                results, self.pixel_per_cm, self.current_standard, self.current_standard.min_defect_mm,
                image_width=img_w, ref_width=1920, detect_height=self.ctrl.detect_height)
            logger.debug("Single image detect: pixel_per_cm=%.2f%s", self.pixel_per_cm, calc_log)
            im_out = draw_defect_labels(im_out, details, self.current_standard, results)
            im_out = draw_filter_overlay(im_out, filter_info)
            self._result_img = im_out
            self.lbl_dst.setPixmap(cv2_to_qpixmap(im_out, self.lbl_dst.width(), self.lbl_dst.height()))
            n = len(results)
            filter_status = self.ctrl.filter_status_text(filter_info)
            filter_suffix = f" | {filter_status}" if filter_status else ""
            self.lbl_status.setText(
                f"计算耗时 {ms:.0f}ms | 原始缺陷 {len(raw_results)} 处 | 过滤后 {n} 处 | 有效缺陷 {len(details)} 处 | 超标数 {oversized} (容忍度≤{self.current_standard.max_defects}) | 致命缺陷 {lethal} | {'通过' if passed else '不合格'}{filter_suffix}"
            )
            self.ctrl.update_stats(results, passed, oversized, details, lethal)
            self.btn_save.setEnabled(True)
            if hasattr(self, "btn_quick_snap"):
                self.btn_quick_snap.setEnabled(True)
            if passed:
                self._set_status_pass()
            else:
                self._set_status_fail()
            if self.ctrl.is_auto_save_enabled(passed):
                result_str = "qualified" if passed else "unqualified"
                auto_save_defect(im_out, self.ctrl.auto_save_dir, prefix=f"single_{result_str}", extra_info=f"_oversized{oversized}")
        except Exception as e:
            logger.exception("Image detection failed")
            QMessageBox.critical(self, "检测错误", f"检测过程中发生错误：\n{str(e)}")
        finally:
            self.btn_detect.setEnabled(True)
            self.btn_detect.setText("执行单次检测")

    def save_result(self):
        if self._result_img is None:
            return
        p, _ = QFileDialog.getSaveFileName(self, "输出检测画面",
                                           f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                                           "JPEG (*.jpg);;PNG (*.png)")
        if p:
            write_image_unicode(p, self._result_img)
            QMessageBox.information(self, "操作成功", f"结果已保存至:\n{p}")

    def quick_screenshot(self):
        """快速截图保存当前检测结果"""
        if self._result_img is None:
            QMessageBox.warning(self, "提示", "当前无检测结果可截图")
            return
        save_dir = os.path.join(os.getcwd(), "screenshots")
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(save_dir, f"screenshot_{timestamp}.jpg")
        write_image_unicode(filepath, self._result_img)
        QMessageBox.information(self, "操作成功", f"截图已保存至:\n{filepath}")

    def export_detection_report(self):
        """一键导出检测报告"""
        if self._result_img is None and not self._folder_files:
            QMessageBox.warning(self, "提示", "当前无检测结果可导出报告")
            return
        report_dir = os.path.join(os.getcwd(), "reports")
        os.makedirs(report_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(report_dir, f"report_{timestamp}.txt")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("          划痕缺陷检测系统 - 检测报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"检测标准: {self.current_standard.name if self.current_standard else '未设置'}\n")
            f.write(f"检测对象: {'划痕' if self.current_standard.detect_object == 'scratch' else '铝合金零件'}\n")
            f.write(f"置信度阈值: {self.ctrl.get_conf():.2f}\n")
            f.write(f"IoU阈值: {self.ctrl.get_iou():.2f}\n\n")

            # 统计数据
            total = self.ctrl._total if hasattr(self.ctrl, '_total') else 0
            qualified = self.ctrl.qualified if hasattr(self.ctrl, 'qualified') else 0
            unqualified = self.ctrl.unqualified if hasattr(self.ctrl, 'unqualified') else 0
            avg_time = self._avg_detect_time if hasattr(self, '_avg_detect_time') and self._avg_detect_time else 0

            f.write("-" * 60 + "\n")
            f.write("检测统计:\n")
            f.write(f"  总检测数: {total}\n")
            f.write(f"  合格数: {qualified}\n")
            f.write(f"  不合格数: {unqualified}\n")
            f.write(f"  平均检测耗时: {avg_time:.1f}ms\n")
            f.write("-" * 60 + "\n\n")

            if self._result_img is not None:
                f.write("最新检测结果: 已保存为截图\n")

            f.write("=" * 60 + "\n")
            f.write("报告结束\n")

        QMessageBox.information(self, "操作成功", f"检测报告已保存至:\n{report_path}")

    def reset(self):
        self.im0_orig = self._result_img = None
        self.lbl_src.setText("输入原始图像")
        self.lbl_src.setPixmap(QPixmap())
        self.lbl_dst.setText("系统检测视图")
        self.lbl_dst.setPixmap(QPixmap())
        self.lbl_status.setText("")
        self.btn_detect.setEnabled(False)
        self.btn_save.setEnabled(False)
        if hasattr(self, "btn_quick_snap"):
            self.btn_quick_snap.setEnabled(False)
        self._avg_detect_time = 0
        self._total_detect_time = 0
        self._detect_count = 0
        self._batch_saved_files = []
        self.ctrl.lbl_avg_time.setText("—")
        self.ctrl.lbl_last_time.setText("—")
        self.ctrl.reset_stats()
        self._set_status_neutral()

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "指定工作目录")
        if not folder:
            return
        files = sorted(str(p) for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXTS)
        if not files:
            QMessageBox.warning(self, "扫描提示", "指定工作目录下未发现支持的图像格式数据")
            return
        self._folder_path = folder
        self._folder_files = files
        # 重置所有状态
        self._avg_detect_time = 0
        self._total_detect_time = 0
        self._detect_count = 0
        self._batch_saved_files = []
        self.batch_bar.setValue(0)
        self.lbl_folder.setText(f"{os.path.basename(folder)}  (共解析到 {len(files)} 份样本)")
        self.btn_batch.setEnabled(True)
        self.btn_abort.setEnabled(False)
        self.btn_pause.setEnabled(False)
        if hasattr(self, "btn_quick_snap"):
            self.btn_quick_snap.setEnabled(False)
        self.lbl_status.setText("")
        self._set_status_neutral()
        self.ctrl.lbl_avg_time.setText("—")
        self.ctrl.lbl_last_time.setText("—")
        self.ctrl.reset_stats()

    def start_batch(self):
        if not self._folder_files:
            return
        if not ensure_detector_available(self, self.det):
            return
        self._batch_saved_files = []
        self._batch_start_cancel_requested = False
        # 使用 dect_result 代替旧的 detect_results
        save_dir = self.ctrl.get_auto_save_dir(self._folder_path)
        self._batch_button_idle_text = self.btn_batch.text()
        self.batch_bar.setValue(0)
        self.btn_batch.setEnabled(False)
        self.btn_batch.setText("启动中...")
        self.btn_abort.setEnabled(True)
        self.btn_folder.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.lbl_status.setText(f"批量检测正在启动... 共 {len(self._folder_files)} 张")
        QApplication.processEvents()
        # 清空当前批次已保存文件记录
        self._batch_saved_files = []
        QTimer.singleShot(0, lambda: self._launch_batch_worker(save_dir))

    def _launch_batch_worker(self, save_dir):
        if self._batch_start_cancel_requested:
            return
        self._batch_worker = BatchWorker(
            self.det, self._folder_files,
            self.ctrl.get_conf(), self.ctrl.get_iou(),
            self.ctrl.get_grayscale(), self.ctrl.get_exposure(), save_dir,
            self.pixel_per_cm, self.current_standard,
            auto_save=self.ctrl.auto_save_cb.isChecked(),
            auto_save_dir=save_dir,
            auto_save_type=self.ctrl.auto_save_type,
            detect_height=self.ctrl.detect_height,
            filter_settings=self.ctrl.get_filter_settings()
        )
        self._batch_worker.sig_status.connect(self._on_batch_status)
        self._batch_worker.sig_prepare_progress.connect(self._on_prepare_prog)
        self._batch_worker.sig_progress.connect(self._on_prog)
        self._batch_worker.sig_result.connect(self._on_result)
        self._batch_worker.sig_done.connect(self._on_done)
        self._batch_worker.sig_aborted.connect(self._on_aborted)
        self._batch_worker.start()

    def abort_batch(self):
        if self._batch_worker:
            self._batch_worker.abort()
            self.lbl_status.setText("正在中止批量检测任务...")
            self.btn_abort.setEnabled(False)
        elif not self.btn_batch.isEnabled():
            self._batch_start_cancel_requested = True
            self._on_aborted(self.ctrl.get_auto_save_dir(self._folder_path))
        self.btn_pause.setEnabled(False)

    def toggle_pause(self):
        if self._batch_worker:
            paused = self._batch_worker.toggle_pause()
            self.lbl_status.setText("批量检测已暂停" if paused else "批量检测继续运行")

    def _on_batch_status(self, text):
        self.lbl_status.setText(text)

    def _on_prepare_prog(self, cur, total):
        self.batch_bar.setRange(0, 100)
        self.batch_bar.setValue(int(cur / total * 100) if total else 0)
        self.lbl_status.setText(f"正在准备批量检测数据... {cur}/{total}")

    def _on_prog(self, cur, total):
        self.batch_bar.setRange(0, 100)
        self.batch_bar.setValue(int(cur / total * 100))
        if hasattr(self, "btn_batch"):
            self.btn_batch.setText("检测中...")
        self.lbl_status.setText(f"批处理工作流运行中: 进度 {cur}/{total}")

    def _on_result(self, _path, dets, im_out, passed, oversized, details, lethal, saved_path="", image_time_ms=0.0):
        del _path
        self._result_img = im_out
        self.lbl_dst.setPixmap(cv2_to_qpixmap(im_out, self.lbl_dst.width(), self.lbl_dst.height()))
        if hasattr(self, "btn_quick_snap"):
            self.btn_quick_snap.setEnabled(True)
        self.ctrl.update_stats(dets, passed, oversized, details, lethal)
        # 记录保存的文件路径（用于后续分批整理）
        if saved_path:
            self._batch_saved_files.append(saved_path)
            batch_no = self.ctrl._current_batch_no()
            batch = self.ctrl._ensure_batch_stats(batch_no)
            batch.setdefault("saved_files", []).append(saved_path)
        # 批量时跟踪耗时（使用 worker 报告的实际每张耗时，而非 UI 回调间隔）
        if image_time_ms > 0:
            self._detect_count += 1
            self._total_detect_time += image_time_ms
            self._avg_detect_time = self._total_detect_time / self._detect_count
            self.ctrl.lbl_last_time.setText(f"{image_time_ms:.0f}ms")
            self.ctrl.lbl_avg_time.setText(f"{self._avg_detect_time:.1f}ms")
            fit_label_font(self.ctrl.lbl_last_time)
            fit_label_font(self.ctrl.lbl_avg_time)

    def _on_done(self, out_dir, total_time=0.0, avg_per_image_io=0.0, avg_per_image_infer=0.0):
        self.btn_batch.setEnabled(True)
        self.btn_batch.setText(getattr(self, "_batch_button_idle_text", self.btn_batch.text()))
        self.btn_abort.setEnabled(False)
        self.btn_folder.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self._batch_worker = None
        self.batch_bar.setValue(100)
        self.lbl_status.setText(f"批处理任务全部完成。结果归档路径: {out_dir}")
        self.ctrl.append_summary(len(self._folder_files), total_time, avg_per_image_io, avg_per_image_infer)

        # 如果当前批次已标记为需复检，自动整理完整文件集
        last_batch_no = self.ctrl._current_batch_no()
        last_batch = self.ctrl._ensure_batch_stats(last_batch_no)
        if last_batch.get("manual_review") and last_batch.get("saved_files"):
            saved_files = last_batch["saved_files"]
            dect_dir = os.path.dirname(saved_files[0])
            batch_dir = os.path.join(dect_dir, f"Batch_{last_batch_no:03d}_\xe9\x9c\x80\xe4\xba\xba\xe5\xb7\xa5\xe5\xa4\x8d\xe6\xa3\x80")
            os.makedirs(batch_dir, exist_ok=True)
            for f in saved_files:
                if os.path.exists(f) and os.path.dirname(f) != batch_dir:
                    shutil.copy2(f, os.path.join(batch_dir, os.path.basename(f)))
            last_batch["batch_folder"] = batch_dir

        # 仅当勾选了自动留存或自动生成报告时才弹出完成对话框
        show_popup = self.ctrl.auto_save_cb.isChecked() or self.ctrl.auto_report_cb.isChecked()
        if show_popup:
            dlg = QMessageBox(self)
            dlg.setWindowTitle("工作流结束")
            dlg.setText(f"渲染视图及报表 batch_report.csv 已存储至:\n{out_dir}")
            dlg.setIcon(QMessageBox.Information)

            open_folder_btn = dlg.addButton("打开结果文件夹", QMessageBox.ActionRole)
            ok_btn = dlg.addButton("确定", QMessageBox.AcceptRole)
            dlg.setDefaultButton(ok_btn)

            dlg.exec_()

            if dlg.clickedButton() == open_folder_btn:
                if not open_local_path(out_dir):
                    QMessageBox.information(self, "提示", f"结果目录已生成，但当前系统未能直接打开：\n{out_dir}")

    def _ensure_abort_report(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "batch_report.csv")
        if os.path.exists(csv_path):
            return csv_path
        rows = [
            ["文件名", "缺陷数", "超标缺陷数", "是否通过", "均值置信度", "时间", "高级过滤"],
            [
                "任务状态", "", "", "已中止", "",
                datetime.now().strftime("%H:%M:%S"),
                f"任务启动前已中止，共 {len(self._folder_files)} 张"
            ]
        ]
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerows(rows)
        return csv_path

    def _on_aborted(self, out_dir):
        self._ensure_abort_report(out_dir)
        self.btn_batch.setEnabled(True)
        self.btn_batch.setText(getattr(self, "_batch_button_idle_text", self.btn_batch.text()))
        self.btn_abort.setEnabled(False)
        self.btn_folder.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self._batch_worker = None
        self.lbl_status.setText(f"批量检测已中止。报告路径: {os.path.join(out_dir, 'batch_report.csv')}")
        if self._suppress_batch_abort_dialog:
            return

        dlg = QMessageBox(self)
        dlg.setWindowTitle("批量检测已中止")
        dlg.setText(f"任务已中止，已生成当前进度的 batch_report.csv：\n{out_dir}")
        dlg.setIcon(QMessageBox.Information)
        open_folder_btn = dlg.addButton("打开结果文件夹", QMessageBox.ActionRole)
        ok_btn = dlg.addButton("确定", QMessageBox.AcceptRole)
        dlg.setDefaultButton(ok_btn)
        dlg.exec_()
        if dlg.clickedButton() == open_folder_btn:
            if not open_local_path(out_dir):
                QMessageBox.information(self, "提示", f"结果目录已生成，但当前系统未能直接打开：\n{out_dir}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.im0_orig is not None:
            self.lbl_src.setPixmap(
                cv2_to_qpixmap(self.ctrl.preprocess(self.im0_orig), self.lbl_src.width(), self.lbl_src.height()))
        if self._result_img is not None:
            self.lbl_dst.setPixmap(cv2_to_qpixmap(self._result_img, self.lbl_dst.width(), self.lbl_dst.height()))


class CameraTab(StatusBarMixin, QWidget):
    def __init__(self, det_ref: list, ctrl: ControlPanel, parent=None):
        super().__init__(parent)
        self.det_ref = det_ref
        self.ctrl = ctrl
        self.cap = cv2.VideoCapture()
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self._frame_queue = Queue(maxsize=1)
        self._result_queue = Queue(maxsize=1)
        self._reader_thread = None
        self._detect_thread = None
        self._pipeline_active = False
        self._detect_mode = False
        self._last_raw = None
        self._last_save_time = 0
        self.camera_index = int(os.getenv("CAMERA_INDEX", "0"))
        self.camera_width = int(os.getenv("CAMERA_WIDTH", "1280"))
        self.camera_height = int(os.getenv("CAMERA_HEIGHT", "720"))
        self.camera_fps = float(os.getenv("CAMERA_FPS", "30"))
        self.camera_fourcc = os.getenv("CAMERA_FOURCC", "MJPG").strip().upper() or "MJPG"
        self.camera_backend = os.getenv("CAMERA_BACKEND", "DSHOW").strip().upper()
        self.camera_native_dshow = os.getenv("CAMERA_NATIVE_DSHOW", "1").strip().lower() in {"1", "true", "yes", "on"}
        self.camera_backend_candidates = [
            item.strip().upper()
            for item in os.getenv("CAMERA_BACKENDS", "DSHOW").split(",")
            if item.strip()
        ]
        self.camera_fourcc_candidates = [
            item.strip().upper()
            for item in os.getenv("CAMERA_FOURCCS", "MJPG,YUY2").split(",")
            if item.strip()
        ]
        self.camera_auto_select = os.getenv("CAMERA_AUTO_SELECT", "1").strip().lower() in {"1", "true", "yes", "on"}
        self._last_camera_info = {}
        self.realtime_imgsz = int(os.getenv("YOLO_REALTIME_IMGSZ", "416"))
        self._display_every = max(1, int(os.getenv("UI_REALTIME_DISPLAY_EVERY", "3")))
        self._display_frame_idx = 0
        self._build_ui()
        self.ctrl.standard_changed.connect(self.on_standard_changed)
        self.current_standard = self.ctrl.current_standard
        self.pixel_per_cm = self.ctrl.pixel_per_cm

    @property
    def det(self):
        return self.det_ref[0]

    def on_standard_changed(self, std, px_per_cm):
        self.current_standard = std
        self.pixel_per_cm = px_per_cm

    def _camera_backend_api(self, name):
        name = (name or "ANY").strip().upper()
        if name == "DSHOW":
            return cv2.CAP_DSHOW
        if name == "MSMF":
            return cv2.CAP_MSMF
        return cv2.CAP_ANY

    def _capture_backend_name(self, cap, fallback):
        try:
            return cap.getBackendName()
        except Exception:
            return fallback

    def _decode_fourcc(self, value):
        try:
            value = int(value)
            chars = [chr((value >> 8 * i) & 0xFF) for i in range(4)]
            text = "".join(chars)
            return text if text.strip("\x00 ") else "未知"
        except Exception:
            return "未知"

    def _fourcc_matches(self, requested, actual):
        requested = (requested or "").strip().upper()
        actual = (actual or "").strip().upper()
        if not requested:
            return True
        aliases = {
            "MJPG": {"MJPG", "MJPEG"},
            "YUY2": {"YUY2", "YUYV"},
        }
        return actual in aliases.get(requested, {requested})

    def _capture_open_params(self, fourcc):
        params = []
        fourcc = (fourcc or "").strip().upper()
        if fourcc:
            params.extend([cv2.CAP_PROP_FOURCC, int(cv2.VideoWriter_fourcc(*fourcc[:4]))])
        params.extend([
            cv2.CAP_PROP_FRAME_WIDTH, int(self.camera_width),
            cv2.CAP_PROP_FRAME_HEIGHT, int(self.camera_height),
            cv2.CAP_PROP_FPS, int(round(self.camera_fps)),
        ])
        return params

    def _open_capture_handle(self, backend, fourcc):
        api = self._camera_backend_api(backend)
        if self.camera_native_dshow and api == cv2.CAP_DSHOW and (fourcc or "").strip():
            cap = DirectShowNativeCapture(
                self.camera_index,
                self.camera_width,
                self.camera_height,
                self.camera_fps,
                fourcc=fourcc,
            )
            if cap.isOpened():
                return cap, "native_dshow"
            logger.warning("Native DirectShow capture failed: %s", cap.last_error)
            cap.release()
        params = self._capture_open_params(fourcc)
        if api != cv2.CAP_ANY and params:
            try:
                cap = cv2.VideoCapture(self.camera_index, api, params)
                if cap.isOpened():
                    return cap, "open_params"
                cap.release()
            except Exception as exc:
                logger.debug("Open camera with params failed: backend=%s fourcc=%s error=%s", backend, fourcc, exc)
        cap = cv2.VideoCapture(self.camera_index, api) if api != cv2.CAP_ANY else cv2.VideoCapture(self.camera_index)
        return cap, "post_open"

    def _configure_capture(self, cap, fourcc):
        fourcc = (fourcc or "").strip().upper()
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc[:4]))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc[:4]))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        try:
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        except Exception:
            pass

    def _measure_capture_fps(self, cap, frames=18, max_seconds=1.2):
        start = time.perf_counter()
        count = 0
        last_frame = None
        while count < frames and (time.perf_counter() - start) < max_seconds:
            ok, frame = cap.read()
            if not ok:
                break
            last_frame = frame
            count += 1
        elapsed = max(0.001, time.perf_counter() - start)
        return count / elapsed if count else 0.0, last_frame

    def _camera_attempts(self):
        backends = self.camera_backend_candidates if self.camera_backend == "AUTO" else [self.camera_backend]
        fourccs = [self.camera_fourcc or "MJPG"]
        seen = set()
        for backend in backends:
            for fourcc in fourccs:
                key = (backend, fourcc)
                if key in seen:
                    continue
                seen.add(key)
                yield backend, fourcc

    def _open_camera_best(self):
        best = None
        attempt_logs = []
        attempts = list(self._camera_attempts()) if self.camera_auto_select else [
            (self.camera_backend if self.camera_backend != "AUTO" else "DSHOW", self.camera_fourcc)
        ]
        for backend, fourcc in attempts:
            cap, open_mode = self._open_capture_handle(backend, fourcc)
            opened = cap.isOpened()
            if not opened:
                cap.release()
                attempt_logs.append(f"{backend}/{fourcc or '默认'} 打开失败")
                continue
            self._configure_capture(cap, fourcc)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            actual_fourcc = self._decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
            measured_fps, last_frame = self._measure_capture_fps(cap)
            post_read_fourcc = self._decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
            if post_read_fourcc != "未知":
                actual_fourcc = post_read_fourcc
            attempt_logs.append(
                f"{backend}/{fourcc or '默认'}({open_mode}) -> {actual_w}x{actual_h} 属性{actual_fps:.1f} 实测{measured_fps:.1f} 格式{actual_fourcc}"
            )
            info = {
                "backend": backend,
                "opencv_backend": self._capture_backend_name(cap, backend),
                "open_mode": open_mode,
                "requested_fourcc": fourcc,
                "fourcc": fourcc or actual_fourcc,
                "actual_w": actual_w,
                "actual_h": actual_h,
                "actual_fps": actual_fps,
                "actual_fourcc": actual_fourcc,
                "format_matched": self._fourcc_matches(fourcc, actual_fourcc),
                "measured_fps": measured_fps,
                "attempt_logs": attempt_logs[:],
                "last_frame": last_frame,
            }
            if actual_w <= 0 or actual_h <= 0:
                cap.release()
                continue
            if best is None or measured_fps > best[0]:
                if best is not None:
                    best[1].release()
                best = (measured_fps, cap, info)
            else:
                cap.release()
        if best is None:
            self._last_camera_info = {"attempt_logs": attempt_logs}
            return None
        self._last_camera_info = best[2]
        return best[1]

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(2)
        root.addWidget(self._make_status_bar())

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.lbl_raw = self._make_display_label("输入视频源流")
        self.lbl_det = self._make_display_label("实时视觉分析")
        row.addWidget(self.lbl_raw)
        row.addWidget(self.lbl_det)
        root.addLayout(row)

        root.addWidget(self._make_separator())
        root.addWidget(self._make_section_label("硬件控制"))

        r = QHBoxLayout()
        r.setSpacing(4)
        self.btn_open = QPushButton("连接摄像头硬件")
        self.btn_open.setObjectName("btn_enlarged")
        self.btn_detect = QPushButton("接入视觉分析流")
        self.btn_detect.setObjectName("btn_enlarged")
        self.btn_snap = QPushButton("截取当前数据帧")
        self.btn_snap.setObjectName("btn_enlarged")
        self.btn_cam_settings = QPushButton("相机驱动设置")
        self.btn_cam_settings.setObjectName("btn_enlarged")
        self.btn_detect.setEnabled(False)
        self.btn_snap.setEnabled(False)
        self.btn_open.clicked.connect(self.toggle_camera)
        self.btn_detect.clicked.connect(self.toggle_detect)
        self.btn_snap.clicked.connect(self.snapshot)
        self.btn_cam_settings.clicked.connect(self.open_camera_driver_settings)
        for b in (self.btn_open, self.btn_detect, self.btn_snap, self.btn_cam_settings):
            r.addWidget(b)
        root.addLayout(r)

        fmt_row = QHBoxLayout()
        fmt_row.setSpacing(6)
        fmt_label = QLabel("当前采集格式")
        fmt_label.setStyleSheet("color:#5a5a5a; font-size:15px; font-weight:bold;")
        self.lbl_camera_format = QLabel("未连接")
        self.lbl_camera_format.setMinimumHeight(34)
        self.lbl_camera_format.setStyleSheet(
            "QLabel { color:#0078d4; font-size:15px; font-weight:bold; padding:2px 8px; "
            "border:1px solid #d0d7de; border-radius:6px; background:#f6f8fa; }"
        )
        fmt_row.addWidget(fmt_label, 0)
        fmt_row.addWidget(self.lbl_camera_format, 1)
        fmt_row.addStretch(1)
        root.addLayout(fmt_row)

        self.lbl_status = QLabel("硬件设备当前处于离线状态")
        self.lbl_status.setMinimumHeight(24)
        self.lbl_status.setStyleSheet("color:#5a5a5a; font-size:16px; font-weight:bold; padding:2px 0;")
        root.addWidget(self.lbl_status)
        self.lbl_perf = QLabel("")
        self.lbl_perf.setMinimumHeight(22)
        self.lbl_perf.setStyleSheet("color:#0078d4; font-size:15px; font-weight:bold; padding:1px 0;")
        root.addWidget(self.lbl_perf)

    def toggle_camera(self):
        if self.timer.isActive():
            self.stop()
        else:
            self.lbl_status.setText("正在测试摄像头后端/格式...")
            QApplication.processEvents()
            self.cap.release()
            selected = self._open_camera_best()
            if selected is None:
                logs = "\n".join(self._last_camera_info.get("attempt_logs", []))
                QMessageBox.warning(self, "摄像头连接失败", f"没有找到可用的摄像头采集模式。\n\n{logs}")
                return
            self.cap = selected
            info = self._last_camera_info
            actual_w = int(info.get("actual_w") or self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(info.get("actual_h") or self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = float(info.get("actual_fps") or self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
            measured_fps = float(info.get("measured_fps") or 0.0)
            actual_fourcc = info.get("actual_fourcc") or self._decode_fourcc(self.cap.get(cv2.CAP_PROP_FOURCC))
            requested_fourcc = info.get("requested_fourcc") or self.camera_fourcc
            if actual_w <= 0 or actual_h <= 0:
                self.cap.release()
                QMessageBox.warning(self, "摄像头连接失败", "摄像头返回了无效分辨率。")
                return
            self.timer.start(24)
            self.btn_open.setText("断开摄像头连接")
            self.btn_detect.setEnabled(True)
            self.btn_snap.setEnabled(True)
            self.lbl_status.setText(
                f"摄像头接入: {actual_w}x{actual_h} 属性{actual_fps:.1f}FPS / 实测{measured_fps:.1f}FPS "
                f"| 后端{info.get('opencv_backend') or info.get('backend', '?')} | {info.get('open_mode', 'post_open')} | 格式{actual_fourcc}"
            )
            self.lbl_camera_format.setText(
                f"{info.get('opencv_backend') or info.get('backend', '?')} / {info.get('open_mode', 'post_open')} / "
                f"{actual_fourcc} / {actual_w}x{actual_h} / {actual_fps:.1f}FPS"
            )
            if measured_fps and measured_fps < max(15.0, self.camera_fps * 0.7):
                if requested_fourcc and not info.get("format_matched", True):
                    self.lbl_perf.setText(
                        f"已请求 {requested_fourcc}，但驱动返回 {actual_fourcc}。OpenCV 未能应用 AMCAP 的 Capture Format，需要厂商 SDK 或 DirectShow IAMStreamConfig。"
                    )
                else:
                    self.lbl_perf.setText("采集低于目标帧率：请点“相机驱动设置”或在 AMCAP 的 Capture Format 中选择 MJPG。")
            else:
                if requested_fourcc and not info.get("format_matched", True):
                    self.lbl_perf.setText(f"格式未匹配：请求 {requested_fourcc}，实际 {actual_fourcc}。")
                else:
                    self.lbl_perf.setText("")

    def toggle_detect(self):
        if not self._detect_mode:
            if not ensure_detector_available(self, self.det, "引擎不可用"):
                return
            self._detect_mode = True
            self.ctrl.reset_filter_state("camera")
            self._start_pipeline()
        else:
            self._detect_mode = False
            self._stop_pipeline()
            if self.cap.isOpened():
                self.timer.start(24)
            self.lbl_det.setText("实时视觉分析")
            self.lbl_det.setPixmap(QPixmap())
            self._set_status_neutral()
        self.btn_detect.setText("中止视觉分析流" if self._detect_mode else "接入视觉分析流")
        self.lbl_status.setText("引擎实时计算中..." if self._detect_mode else "硬件视频流读取中...")

    def _clear_camera_queues(self):
        for q in (self._frame_queue, self._result_queue):
            while True:
                try:
                    q.get_nowait()
                except Empty:
                    break

    def _start_pipeline(self):
        if not self.cap.isOpened() or self._pipeline_active:
            return
        self.timer.stop()
        self._clear_camera_queues()
        self._pipeline_active = True
        self._reader_thread = FrameReaderThread(self.cap, self._frame_queue)
        self._detect_thread = DetectionWorkerThread(
            self._frame_queue,
            self._result_queue,
            self.det,
            self.ctrl,
            self.realtime_imgsz,
            source="camera",
            display_every=self._display_every,
        )
        self._detect_thread.finished.connect(self._finish_pipeline)
        self._detect_thread.error.connect(self._on_pipeline_error)
        self._reader_thread.start()
        self._detect_thread.start()
        self.timer.start(16)

    def _stop_pipeline(self):
        self._pipeline_active = False
        for worker in (self._reader_thread, self._detect_thread):
            if worker:
                worker.stop()
        for worker in (self._reader_thread, self._detect_thread):
            if worker and worker.isRunning() and not worker.wait(3000):
                logger.warning("Camera worker did not stop within timeout: %s", worker)
        self._reader_thread = None
        self._detect_thread = None
        self._clear_camera_queues()

    def _finish_pipeline(self):
        self._pipeline_active = False

    def _on_pipeline_error(self, backend, message):
        self._stop_pipeline()
        self._detect_mode = False
        self.btn_detect.setText("接入视觉分析流")
        self.lbl_status.setText(f"视觉引擎异常({backend}): {message}")
        self.lbl_det.setText("实时视觉分析")
        self.lbl_det.setPixmap(QPixmap())
        self._set_status_neutral()
        QMessageBox.critical(self, "检测错误", f"实时检测过程中发生错误：\n后端: {backend}\n{message}")

    def _apply_camera_result(self, payload):
        processed = payload["processed"]
        im_out = payload.get("im_out")
        if payload.get("rendered", False):
            self.lbl_raw.setPixmap(cv2_to_qpixmap(processed, self.lbl_raw.width(), self.lbl_raw.height()))
            if im_out is not None:
                self.lbl_det.setPixmap(cv2_to_qpixmap(im_out, self.lbl_det.width(), self.lbl_det.height()))
        results = payload["results"]
        raw_results = payload["raw_results"]
        filter_info = payload["filter_info"]
        passed = payload["passed"]
        oversized = payload["oversized"]
        details = payload["details"]
        lethal = payload["lethal"]
        timing = payload.get("timing") or {}
        self.ctrl.update_stats(results, passed, oversized, details, lethal, realtime=True, fps_override=timing.get("worker_fps"))
        filter_status = self.ctrl.filter_status_text(filter_info)
        filter_suffix = f" | {filter_status}" if filter_status else ""
        timing_suffix = realtime_timing_text(payload)
        self.lbl_status.setText(
            f"引擎状态正常 | 原始缺陷 {len(raw_results)} | 过滤后 {len(results)} | 有效缺陷 {len(details)} | "
            f"超标 {oversized} (容忍度≤{self.current_standard.max_defects}) | 致命缺陷 {lethal} | "
            f"{'通过' if passed else '不合格'}{filter_suffix}{timing_suffix}"
        )
        self.lbl_perf.setText(realtime_perf_line(payload))
        self._set_status_pass() if passed else self._set_status_fail()
        if im_out is not None and self.ctrl.is_auto_save_enabled(passed):
            now = time.time()
            if now - self._last_save_time >= 1.0:
                self._last_save_time = now
                result_str = "qualified" if passed else "unqualified"
                auto_save_defect(im_out, self.ctrl.auto_save_dir, prefix=f"cam_{result_str}", extra_info=f"_oversized{oversized}")

    def _tick(self):
        if self._pipeline_active:
            latest = None
            while True:
                try:
                    latest = self._result_queue.get_nowait()
                except Empty:
                    break
            if latest is not None:
                self._apply_camera_result(latest)
            return
        ok, frame = self.cap.read()
        if not ok:
            return
        self._last_raw = frame.copy()
        processed = self.ctrl.preprocess(frame)
        self._display_frame_idx += 1
        should_refresh_display = (not self._detect_mode) or (self._display_frame_idx % self._display_every == 0)
        
        # 性能预测建议：如果是在低配电脑上，可以降低左侧原始视频更新频率，此处仅更新分析后的结果
        if should_refresh_display:
            self.lbl_raw.setPixmap(cv2_to_qpixmap(processed, self.lbl_raw.width(), self.lbl_raw.height()))
        if self._detect_mode:
            try:
                results, im_out = self.det.run(processed, imgsz=self.realtime_imgsz,
                                               conf_thres=self.ctrl.get_conf(), iou_thres=self.ctrl.get_iou())
                raw_results = results
                results, filter_info = self.ctrl.apply_detection_filters(results, processed.shape, source="camera", temporal=True)
                # 获取实际图像宽度用于动态像素当量计算
                img_h, img_w = processed.shape[:2]
                oversized, passed, details, lethal, calc_log = evaluate_defects_detailed(
                    results, self.pixel_per_cm, self.current_standard, self.current_standard.min_defect_mm,
                    image_width=img_w, ref_width=1920, detect_height=self.ctrl.detect_height)
                im_out = draw_defect_labels(im_out, details, self.current_standard, results)
                im_out = draw_filter_overlay(im_out, filter_info)
                if should_refresh_display:
                    self.lbl_det.setPixmap(cv2_to_qpixmap(im_out, self.lbl_det.width(), self.lbl_det.height()))
                self.ctrl.update_stats(results, passed, oversized, details, lethal, realtime=True)
                filter_status = self.ctrl.filter_status_text(filter_info)
                filter_suffix = f" | {filter_status}" if filter_status else ""
                self.lbl_status.setText(
                    f"引擎状态正常 | 原始缺陷 {len(raw_results)} | 过滤后 {len(results)} | 有效缺陷 {len(details)} | 超标 {oversized} (容忍度≤{self.current_standard.max_defects}) | 致命缺陷 {lethal} | {'通过' if passed else '不合格'}{filter_suffix}"
                )
                if passed:
                    self._set_status_pass()
                else:
                    self._set_status_fail()
                if self.ctrl.is_auto_save_enabled(passed):
                    now = time.time()
                    if now - self._last_save_time >= 1.0:
                        self._last_save_time = now
                        result_str = "qualified" if passed else "unqualified"
                        auto_save_defect(im_out, self.ctrl.auto_save_dir, prefix=f"cam_{result_str}", extra_info=f"_oversized{oversized}")
            except Exception as e:
                logger.exception("Camera detection failed")
                self._detect_mode = False
                self.btn_detect.setText("接入视觉分析流")
                backend = f"{getattr(self.det, 'backend_name', 'Unknown')} / {getattr(self.det, 'device_name', '')}".strip()
                self.lbl_status.setText(f"视觉引擎异常({backend}): {e}")
                self.lbl_det.setText("实时视觉分析")
                self.lbl_det.setPixmap(QPixmap())
                self._set_status_neutral()
                QMessageBox.critical(self, "检测错误", f"实时检测过程中发生错误：\n后端: {backend}\n{e}")

    def snapshot(self):
        if self._last_raw is None:
            return
        p, _ = QFileDialog.getSaveFileName(self, "帧数据固化", f"snap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                                           "JPEG (*.jpg);;PNG (*.png)")
        if p:
            write_image_unicode(p, self._last_raw)
            QMessageBox.information(self, "操作成功", f"快照数据已转储至:\n{p}")

    def stop(self):
        self._stop_pipeline()
        self.timer.stop()
        self.cap.release()
        self.lbl_raw.setText("输入视频源流")
        self.lbl_raw.setPixmap(QPixmap())
        self.lbl_det.setText("实时视觉分析")
        self.lbl_det.setPixmap(QPixmap())
        self.btn_open.setText("连接摄像头硬件")
        self.btn_detect.setEnabled(False)
        self.btn_snap.setEnabled(False)
        self.lbl_status.setText("硬件设备当前处于离线状态")
        self.lbl_perf.setText("")
        self.lbl_camera_format.setText("未连接")
        self._detect_mode = False
        self.btn_detect.setText("接入视觉分析流")
        self._set_status_neutral()


class FrameReaderThread(QThread):
    finished = pyqtSignal()

    def __init__(self, cap, frame_queue, parent=None, start_frame_no=0, use_capture_pos=False):
        super().__init__(parent)
        self.cap = cap
        self.frame_queue = frame_queue
        self._stop = False
        self._frame_no = int(start_frame_no)
        self.use_capture_pos = bool(use_capture_pos)
        self._last_read_time = None
        self._read_intervals = []

    def stop(self):
        self._stop = True

    def run(self):
        try:
            while not self._stop:
                ok, frame = self.cap.read()
                if not ok:
                    break
                now = time.perf_counter()
                if self._last_read_time is None:
                    capture_fps = 0.0
                else:
                    interval = max(0.001, now - self._last_read_time)
                    self._read_intervals.append(interval)
                    self._read_intervals = self._read_intervals[-30:]
                    capture_fps = len(self._read_intervals) / max(0.001, sum(self._read_intervals))
                self._last_read_time = now
                self._frame_no += 1
                cap_frame_no = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) if self.use_capture_pos else 0
                frame_no = cap_frame_no if cap_frame_no > 0 else self._frame_no
                try:
                    self.frame_queue.put_nowait((frame_no, frame, capture_fps))
                except Full:
                    try:
                        self.frame_queue.get_nowait()
                    except Empty:
                        pass
                    try:
                        self.frame_queue.put_nowait((frame_no, frame, capture_fps))
                    except Full:
                        pass
        except Exception:
            logger.exception("Video frame reader failed")
        finally:
            try:
                self.frame_queue.put_nowait(None)
            except Full:
                try:
                    self.frame_queue.get_nowait()
                except Empty:
                    pass
                try:
                    self.frame_queue.put_nowait(None)
                except Full:
                    pass
            self.finished.emit()


class DetectionWorkerThread(QThread):
    error = pyqtSignal(str, str)
    finished = pyqtSignal()

    def __init__(self, frame_queue, result_queue, det, ctrl, realtime_imgsz,
                 source="video", display_every=3, parent=None):
        super().__init__(parent)
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.det = det
        self.ctrl = ctrl
        self.realtime_imgsz = realtime_imgsz
        self.source = source
        self.display_every = max(1, int(display_every))
        self._stop = False
        self._last_done_time = None
        self._done_intervals = []

    def stop(self):
        self._stop = True

    def run(self):
        while not self._stop:
            try:
                item = self.frame_queue.get(timeout=0.2)
            except Empty:
                continue
            if item is None:
                break
            if len(item) >= 3:
                frame_no, frame, capture_fps = item[:3]
            else:
                frame_no, frame = item
                capture_fps = 0.0
            try:
                t0 = time.perf_counter()
                processed = self.ctrl.preprocess(frame)
                t_pre = time.perf_counter()
                with QMutexLocker(detector_mutex):
                    results, im_out = self.det.run(
                        processed,
                        imgsz=self.realtime_imgsz,
                        conf_thres=self.ctrl.get_conf(),
                        iou_thres=self.ctrl.get_iou(),
                    )
                t_infer = time.perf_counter()
                raw_results = results
                results, filter_info = self.ctrl.apply_detection_filters(
                    results, processed.shape, source=self.source, temporal=True
                )
                t_filter = time.perf_counter()
                _, img_w = processed.shape[:2]
                standard = self.ctrl.current_standard
                oversized, passed, details, lethal, calc_log = evaluate_defects_detailed(
                    results,
                    self.ctrl.pixel_per_cm,
                    standard,
                    standard.min_defect_mm,
                    image_width=img_w,
                    ref_width=1920,
                    detect_height=self.ctrl.detect_height,
                )
                t_eval = time.perf_counter()
                should_render = int(frame_no) % self.display_every == 0
                if should_render:
                    im_out = draw_defect_labels(im_out, details, standard, results)
                    im_out = draw_filter_overlay(im_out, filter_info)
                else:
                    im_out = None
                t_render = time.perf_counter()
                total_ms = max(0.001, (t_render - t0) * 1000.0)
                calc_fps = 1000.0 / total_ms
                if self._last_done_time is None:
                    worker_fps = 0.0
                else:
                    done_interval = max(0.001, t_render - self._last_done_time)
                    self._done_intervals.append(done_interval)
                    self._done_intervals = self._done_intervals[-30:]
                    worker_fps = len(self._done_intervals) / max(0.001, sum(self._done_intervals))
                self._last_done_time = t_render
                payload = {
                    "frame_no": frame_no,
                    "processed": processed,
                    "im_out": im_out,
                    "rendered": should_render,
                    "raw_results": raw_results,
                    "results": results,
                    "filter_info": filter_info,
                    "passed": passed,
                    "oversized": oversized,
                    "details": details,
                    "lethal": lethal,
                    "timing": {
                        "pre": (t_pre - t0) * 1000.0,
                        "infer": (t_infer - t_pre) * 1000.0,
                        "filter": (t_filter - t_infer) * 1000.0,
                        "eval": (t_eval - t_filter) * 1000.0,
                        "render": (t_render - t_eval) * 1000.0,
                        "total": total_ms,
                        "capture_fps": float(capture_fps or 0.0),
                        "calc_fps": calc_fps,
                        "worker_fps": worker_fps,
                    },
                    "backend_profile": dict(getattr(self.det, "last_profile", {}) or {}),
                }
                try:
                    self.result_queue.put_nowait(payload)
                except Full:
                    try:
                        self.result_queue.get_nowait()
                    except Empty:
                        pass
                    self.result_queue.put_nowait(payload)
            except Exception as exc:
                backend = f"{getattr(self.det, 'backend_name', 'Unknown')} / {getattr(self.det, 'device_name', '')}".strip()
                logger.exception("Video detection worker failed")
                self.error.emit(backend, str(exc))
                break
        self.finished.emit()


class VideoTab(StatusBarMixin, QWidget):
    def __init__(self, det_ref: list, ctrl: ControlPanel, parent=None):
        super().__init__(parent)
        self.det_ref = det_ref
        self.ctrl = ctrl
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self._frame_queue = Queue(maxsize=2)
        self._result_queue = Queue(maxsize=2)
        self._reader_thread = None
        self._detect_thread = None
        self._pipeline_active = False
        self._detect_mode = False
        self._total_frames = 0
        self._cur_frame = 0
        self._max_seen_frame = 0
        self._video_completed = False
        self._summary_written = False
        self._marked_frames = set()
        self._last_save_time = 0
        self.realtime_imgsz = int(os.getenv("YOLO_REALTIME_IMGSZ", "416"))
        self._display_every = max(1, int(os.getenv("UI_REALTIME_DISPLAY_EVERY", "3")))
        self._build_ui()
        self.ctrl.standard_changed.connect(self.on_standard_changed)
        self.current_standard = self.ctrl.current_standard
        self.pixel_per_cm = self.ctrl.pixel_per_cm

    @property
    def det(self):
        return self.det_ref[0]

    def on_standard_changed(self, std, px_per_cm):
        self.current_standard = std
        self.pixel_per_cm = px_per_cm

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(2)
        root.addWidget(self._make_status_bar())

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.lbl_raw = self._make_display_label("输入视频源流")
        self.lbl_det = self._make_display_label("视频计算视图")
        row.addWidget(self.lbl_raw)
        row.addWidget(self.lbl_det)
        root.addLayout(row)

        timeline_row = QHBoxLayout()
        timeline_row.setContentsMargins(0, 0, 0, 0)
        timeline_row.setSpacing(8)
        self.progress_label = QLabel("媒体流解码进度: 0% | 00:00 / 00:00")
        self.progress_label.setMinimumWidth(260)
        self.progress_label.setStyleSheet("color:#5a5a5a; font-size:15px; font-weight:bold;")
        self.progress = MarkerSlider()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setEnabled(False)
        self.progress.seek_requested.connect(self.seek_to_frame)
        self.progress.marker_requested.connect(self.seek_to_frame)
        timeline_row.addWidget(self.progress_label, 0)
        timeline_row.addWidget(self.progress, 1)
        root.addLayout(timeline_row)

        root.addWidget(self._make_separator())
        root.addWidget(self._make_section_label("媒体控制台"))

        r = QHBoxLayout()
        r.setSpacing(4)
        self.btn_open = QPushButton("指定媒体文件")
        self.btn_open.setObjectName("btn_enlarged")
        self.btn_play = QPushButton("执行读取流")
        self.btn_play.setObjectName("btn_enlarged")
        self.btn_detect = QPushButton("接入视觉分析引擎")
        self.btn_detect.setObjectName("btn_enlarged")
        self.btn_stop = QPushButton("中断当前任务")
        self.btn_stop.setObjectName("btn_enlarged")
        self.btn_play.setEnabled(False)
        self.btn_detect.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_open.clicked.connect(self.open_video)
        self.btn_play.clicked.connect(self.play_video)
        self.btn_detect.clicked.connect(self.toggle_detect)
        self.btn_stop.clicked.connect(lambda: self.stop_video(release=False, summarize=True))
        for b in (self.btn_open, self.btn_play, self.btn_detect, self.btn_stop):
            r.addWidget(b)
        root.addLayout(r)

        self.lbl_status = QLabel("当前未挂载媒体文件")
        self.lbl_status.setMinimumHeight(24)
        self.lbl_status.setStyleSheet("color:#5a5a5a; font-size:16px; font-weight:bold; padding:2px 0;")
        root.addWidget(self.lbl_status)
        self.lbl_perf = QLabel("")
        self.lbl_perf.setMinimumHeight(22)
        self.lbl_perf.setStyleSheet("color:#0078d4; font-size:15px; font-weight:bold; padding:1px 0;")
        root.addWidget(self.lbl_perf)

    def open_video(self):
        p, _ = QFileDialog.getOpenFileName(self, "挂载媒体", "",
                                           "视频 (*.mp4 *.avi *.mov *.rmvb *.mkv *.ts);;All Files (*)")
        if not p:
            return
        if self.cap:
            self.stop_video(release=True, summarize=False)
        self.cap = cv2.VideoCapture(p)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "解析错误", "解析器未能建立媒体连接")
            self.cap = None
            return
        self._total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._cur_frame = 0
        self._max_seen_frame = 0
        self._video_completed = False
        self._summary_written = False
        self._marked_frames = set()
        self.progress.setRange(0, max(1, self._total_frames))
        self.progress.setValue(0)
        self.progress.set_markers([])
        self.progress.setEnabled(self._total_frames > 0)
        self._update_progress_label()
        self.lbl_status.setText(f"媒体挂载完毕: {os.path.basename(p)}  (共解析 {self._total_frames} 帧)")
        self.btn_play.setEnabled(True)
        self.btn_detect.setEnabled(True)

    def _clear_video_queues(self):
        for q in (self._frame_queue, self._result_queue):
            while True:
                try:
                    q.get_nowait()
                except Empty:
                    break

    def _start_pipeline(self):
        if not self.cap or self._pipeline_active:
            return
        self.timer.stop()
        self._clear_video_queues()
        self._pipeline_active = True
        start_frame_no = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) if self.cap else self._cur_frame
        self._reader_thread = FrameReaderThread(
            self.cap,
            self._frame_queue,
            start_frame_no=start_frame_no,
        )
        self._detect_thread = DetectionWorkerThread(
            self._frame_queue,
            self._result_queue,
            self.det,
            self.ctrl,
            self.realtime_imgsz,
            source="video",
            display_every=self._display_every,
        )
        self._detect_thread.finished.connect(self._finish_video_playback)
        self._detect_thread.error.connect(self._on_pipeline_error)
        self._reader_thread.start()
        self._detect_thread.start()
        self.timer.start(15)

    def _stop_pipeline(self):
        self._pipeline_active = False
        for worker in (self._reader_thread, self._detect_thread):
            if worker:
                worker.stop()
        for worker in (self._reader_thread, self._detect_thread):
            if worker and worker.isRunning() and not worker.wait(3000):
                logger.warning("Video worker did not stop within timeout: %s", worker)
        self._reader_thread = None
        self._detect_thread = None
        self._clear_video_queues()

    def _on_pipeline_error(self, backend, message):
        self._stop_pipeline()
        self._detect_mode = False
        self.btn_detect.setText("Enable video detection")
        self.lbl_status.setText(f"Video detector error ({backend}): {message}")
        self.lbl_det.setText("Video detection view")
        self.lbl_det.setPixmap(QPixmap())
        self._set_status_neutral()
        QMessageBox.critical(self, "Detection error", f"Video detection failed:\nBackend: {backend}\n{message}")

    def _apply_video_result(self, payload):
        self._cur_frame = int(payload["frame_no"])
        self._max_seen_frame = max(self._max_seen_frame, self._cur_frame)
        if self._total_frames > 0:
            self.progress.setValue(min(self._cur_frame, self._total_frames))
            if self._cur_frame >= self._total_frames:
                self._video_completed = True
            self._update_progress_label()

        should_refresh_display = self._cur_frame % self._display_every == 0
        if should_refresh_display and payload.get("rendered", True):
            self.lbl_raw.setPixmap(cv2_to_qpixmap(payload["processed"], self.lbl_raw.width(), self.lbl_raw.height()))
            if payload.get("im_out") is not None:
                self.lbl_det.setPixmap(cv2_to_qpixmap(payload["im_out"], self.lbl_det.width(), self.lbl_det.height()))

        results = payload["results"]
        raw_results = payload["raw_results"]
        filter_info = payload["filter_info"]
        passed = payload["passed"]
        oversized = payload["oversized"]
        details = payload["details"]
        lethal = payload["lethal"]
        timing = payload.get("timing") or {}
        self.ctrl.update_stats(results, passed, oversized, details, lethal, realtime=True, fps_override=timing.get("worker_fps"))
        filter_status = self.ctrl.filter_status_text(filter_info)
        filter_suffix = f" | {filter_status}" if filter_status else ""
        timing_suffix = realtime_timing_text(payload)
        self.lbl_status.setText(
            f"Video detecting | raw {len(raw_results)} | filtered {len(results)} | valid {len(details)} | "
            f"oversized {oversized} (limit {self.current_standard.max_defects}) | lethal {lethal} | "
            f"{'PASS' if passed else 'FAIL'}{filter_suffix}{timing_suffix}"
        )
        self.lbl_perf.setText(realtime_perf_line(payload))
        self._set_status_pass() if passed else self._set_status_fail()
        if payload.get("im_out") is not None and self.ctrl.is_auto_save_enabled(passed):
            now = time.time()
            if now - self._last_save_time >= 1.0:
                self._last_save_time = now
                result_str = "qualified" if passed else "unqualified"
                auto_save_defect(payload["im_out"], self.ctrl.auto_save_dir, prefix=f"video_{result_str}",
                                 extra_info=f"_frame{self._cur_frame}")

    def play_video(self):
        if not self.cap:
            return
        if self._video_completed and self._cur_frame >= self._total_frames > 0:
            self.seek_to_frame(0, force=True)
        if self._detect_mode:
            self._start_pipeline()
        else:
            self.timer.start(30)
        self.btn_play.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText("流数据读取中...")

    def toggle_detect(self):
        if not self._detect_mode:
            if not ensure_detector_available(self, self.det, "引擎不可用"):
                return
            self._detect_mode = True
            self.ctrl.reset_filter_state("video")
            if self.cap and self.timer.isActive():
                self._start_pipeline()
        else:
            self._detect_mode = False
            self._stop_pipeline()
            if self.cap and self.btn_stop.isEnabled():
                self.timer.start(30)
            self._set_status_neutral()
        self.btn_detect.setText("切断视觉分析" if self._detect_mode else "接入视觉分析引擎")

    def _tick(self):
        if self._pipeline_active:
            latest = None
            while True:
                try:
                    latest = self._result_queue.get_nowait()
                except Empty:
                    break
            if latest is not None:
                self._apply_video_result(latest)
            return
        if not self.cap:
            return
        ok, frame = self.cap.read()
        if not ok:
            self._finish_video_playback()
            self.lbl_status.setText("媒体流推送结束")
            return
        self._cur_frame += 1
        self._max_seen_frame = max(self._max_seen_frame, self._cur_frame)
        if self._total_frames > 0:
            self.progress.setValue(min(self._cur_frame, self._total_frames))
            if self._cur_frame >= self._total_frames:
                self._video_completed = True
            self._update_progress_label()
        processed = self.ctrl.preprocess(frame)
        should_refresh_display = (not self._detect_mode) or (self._cur_frame % self._display_every == 0)
        if should_refresh_display:
            self.lbl_raw.setPixmap(cv2_to_qpixmap(processed, self.lbl_raw.width(), self.lbl_raw.height()))
        if self._detect_mode:
            try:
                results, im_out = self.det.run(processed, imgsz=self.realtime_imgsz,
                                               conf_thres=self.ctrl.get_conf(), iou_thres=self.ctrl.get_iou())
                raw_results = results
                results, filter_info = self.ctrl.apply_detection_filters(results, processed.shape, source="video", temporal=True)
                # 获取实际图像宽度用于动态像素当量计算
                img_h, img_w = processed.shape[:2]
                oversized, passed, details, lethal, calc_log = evaluate_defects_detailed(
                    results, self.pixel_per_cm, self.current_standard, self.current_standard.min_defect_mm,
                    image_width=img_w, ref_width=1920, detect_height=self.ctrl.detect_height)
                im_out = draw_defect_labels(im_out, details, self.current_standard, results)
                im_out = draw_filter_overlay(im_out, filter_info)
                if should_refresh_display:
                    self.lbl_det.setPixmap(cv2_to_qpixmap(im_out, self.lbl_det.width(), self.lbl_det.height()))
                self.ctrl.update_stats(results, passed, oversized, details, lethal, realtime=True)
                filter_status = self.ctrl.filter_status_text(filter_info)
                filter_suffix = f" | {filter_status}" if filter_status else ""
                self.lbl_status.setText(
                    f"视觉引擎运算中 | 原始缺陷 {len(raw_results)} | 过滤后 {len(results)} | 有效缺陷 {len(details)} | 超标 {oversized} (容忍度≤{self.current_standard.max_defects}) | 致命缺陷 {lethal} | {'通过' if passed else '不合格'}{filter_suffix}"
                )
                if passed:
                    self._set_status_pass()
                else:
                    self._set_status_fail()
                if self.ctrl.is_auto_save_enabled(passed):
                    now = time.time()
                    if now - self._last_save_time >= 1.0:
                        self._last_save_time = now
                        result_str = "qualified" if passed else "unqualified"
                        auto_save_defect(im_out, self.ctrl.auto_save_dir, prefix=f"video_{result_str}",
                                         extra_info=f"_frame{self._cur_frame}")
            except Exception as e:
                logger.exception("Video detection failed")
                self._detect_mode = False
                self.btn_detect.setText("接入视觉分析引擎")
                backend = f"{getattr(self.det, 'backend_name', 'Unknown')} / {getattr(self.det, 'device_name', '')}".strip()
                self.lbl_status.setText(f"视觉引擎异常({backend}): {e}")
                self.lbl_det.setText("视频计算视图")
                self.lbl_det.setPixmap(QPixmap())
                self._set_status_neutral()
                QMessageBox.critical(self, "检测错误", f"视频检测过程中发生错误：\n后端: {backend}\n{e}")

    def _format_time(self, frame_no):
        fps = float(self.cap.get(cv2.CAP_PROP_FPS)) if self.cap else 0.0
        if fps <= 0:
            fps = 30.0
        seconds = max(0, int(frame_no / fps))
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    def _update_progress_label(self):
        if self._total_frames > 0:
            percent = min(100, max(0, int(self._cur_frame / self._total_frames * 100)))
            current = self._format_time(self._cur_frame)
            total = self._format_time(self._total_frames)
            unlocked = "已解锁全片跳转" if self._video_completed else f"可回看至 {self._format_time(self._max_seen_frame)}"
            self.progress_label.setText(f"媒体流解码进度: {percent}% | {current} / {total} | {unlocked}")
        else:
            self.progress_label.setText("媒体流解码进度: 0% | 00:00 / 00:00")

    def _display_frame(self, frame):
        processed = self.ctrl.preprocess(frame)
        self.lbl_raw.setPixmap(cv2_to_qpixmap(processed, self.lbl_raw.width(), self.lbl_raw.height()))
        return processed

    def seek_to_frame(self, frame_no, force=False):
        if not self.cap or self._total_frames <= 0:
            return
        target = int(max(0, min(frame_no, self._total_frames)))
        if not force and not self._video_completed and target > self._max_seen_frame:
            QMessageBox.information(
                self,
                "进度未解锁",
                "当前视频还没有完整播放完。播放完成前，只能回看已经解码过的片段。"
            )
            self.progress.setValue(self._cur_frame)
            return
        was_pipeline = self._pipeline_active
        if was_pipeline:
            self._stop_pipeline()
        was_running = self.timer.isActive()
        if was_running:
            self.timer.stop()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target - 1))
        ok, frame = self.cap.read()
        if ok:
            self._cur_frame = max(1, target)
            self.progress.setValue(self._cur_frame)
            self._display_frame(frame)
            self.lbl_det.setText("视频计算视图")
            self.lbl_det.setPixmap(QPixmap())
            self._update_progress_label()
            self.lbl_status.setText(f"已跳转到第 {self._cur_frame} 帧")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self._cur_frame)
        if was_pipeline and self._detect_mode:
            self._start_pipeline()
        elif was_running:
            self.timer.start(30)

    def mark_current_frame(self):
        if not self.cap or self._cur_frame <= 0:
            QMessageBox.information(self, "提示", "当前没有可标记的视频帧")
            return
        self._marked_frames.add(int(self._cur_frame))
        self.progress.set_markers(self._marked_frames)
        text = f"视频第 {self._cur_frame} 帧已标记，可点击时间轴红色标记点回跳复核"
        self.lbl_status.setText(text)
        if self.ctrl.log_edit:
            self.ctrl.log_edit.append(
                f"<span style='color:#d13438; font-weight:bold;'>[{datetime.now().strftime('%H:%M:%S')}] {text}</span>"
            )

    def _finish_video_playback(self):
        if self._pipeline_active:
            latest = None
            while True:
                try:
                    latest = self._result_queue.get_nowait()
                except Empty:
                    break
            if latest is not None:
                self._apply_video_result(latest)
            self._pipeline_active = False
            if self._reader_thread and self._reader_thread.isRunning():
                self._reader_thread.stop()
                self._reader_thread.wait(3000)
            if self._detect_thread and self._detect_thread.isRunning():
                self._detect_thread.stop()
                self._detect_thread.wait(3000)
            self._reader_thread = None
            self._detect_thread = None
        self.timer.stop()
        self._video_completed = True
        self._cur_frame = self._total_frames if self._total_frames > 0 else self._cur_frame
        self._max_seen_frame = max(self._max_seen_frame, self._cur_frame)
        self.progress.setValue(self._cur_frame)
        self._update_progress_label()
        self.btn_play.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_play.setText("重新读取流")
        if not self._summary_written:
            self.ctrl.append_summary(self._cur_frame)
            self._summary_written = True

    def stop_video(self, release=False, summarize=True):
        self._stop_pipeline()
        self.timer.stop()
        if release and self.cap:
            self.cap.release()
            self.cap = None
        self._detect_mode = False
        self.btn_play.setEnabled(bool(self.cap))
        self.btn_detect.setEnabled(bool(self.cap))
        self.btn_stop.setEnabled(False)
        self.btn_detect.setText("接入视觉分析引擎")
        self.btn_play.setText("执行读取流")
        self.lbl_det.setText("视频计算视图")
        self.lbl_det.setPixmap(QPixmap())
        self.lbl_perf.setText("")
        if release:
            self._cur_frame = 0
            self._max_seen_frame = 0
            self._video_completed = False
            self.progress.setValue(0)
            self.progress.set_markers([])
            self.progress.setEnabled(False)
        self._update_progress_label()
        self._set_status_neutral()
        if summarize and self._cur_frame > 0 and not self._summary_written:
            self.ctrl.append_summary(self._cur_frame)
            self._summary_written = True


# ══════════════════════════════════════════════════════════════════
#  产线分析摘要对话框（matplotlib 图表 + 统计数据）
# ══════════════════════════════════════════════════════════════════
class AnalysisSummaryDialog(QDialog):
    """分析摘要弹出窗口：显示合格率饼图、批次柱状图、缺陷分布直方图等"""

    def __init__(self, parent, ctrl, batch_source_dir=None):
        super().__init__(parent)
        self.ctrl = ctrl
        self.batch_source_dir = batch_source_dir
        self.setWindowTitle("产线分析摘要")
        self.resize(1053, 1053)
        self.setMinimumSize(800, 700)
        self._chart_buf = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # 图表显示区域（滚动）
        self.chart_label = QLabel("正在生成图表...")
        self.chart_label.setAlignment(Qt.AlignCenter)
        self.chart_label.setMinimumSize(600, 420)
        self.chart_label.setStyleSheet("background:#ffffff; border:1px solid #d0d0d0; border-radius:8px;")

        scroll = QScrollArea()
        scroll.setWidget(self.chart_label)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(440)
        layout.addWidget(scroll, 1)

        # 统计文本
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(160)
        self.stats_text.setStyleSheet("font-size:14px; font-family:Microsoft YaHei UI;")
        layout.addWidget(self.stats_text)

        # 按钮行
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        self.btn_save = QPushButton("保存图表")
        self.btn_save.setObjectName("btn_side_compact")
        self.btn_save.setFixedHeight(36)
        self.btn_save.clicked.connect(self._save_chart)
        self.btn_close = QPushButton("关闭")
        self.btn_close.setObjectName("btn_side_compact")
        self.btn_close.setFixedHeight(36)
        self.btn_close.clicked.connect(self.accept)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_close)
        layout.addLayout(btn_layout)

        # 首次生成
        QTimer.singleShot(100, self._generate_chart)
        self._update_stats_text()

    def _generate_chart(self):
        """使用 matplotlib 生成四象限统计图表"""
        ctrl = self.ctrl
        qualified = max(0, ctrl.qualified)
        unqualified = max(0, ctrl.unqualified)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # 设置中文字体
            plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei", "Noto Sans CJK"]
            plt.rcParams["axes.unicode_minus"] = False

            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            fig.suptitle("产线分析摘要", fontsize=14, fontweight="bold", y=0.98)
            plt.subplots_adjust(hspace=0.35, wspace=0.30, top=0.92, bottom=0.06, left=0.08, right=0.95)

            # ── 左上：合格/不合格 饼图 ──
            ax1 = axes[0, 0]
            if unqualified > 0 or qualified > 0:
                labels = [f"合格\n({qualified})", f"不合格\n({unqualified})"]
                sizes = [qualified, unqualified]
                colors = ["#4caf50", "#d13438"]
                ax1.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
                        startangle=90, textprops={"fontsize": 10})
            ax1.set_title("合格/不合格比例", fontsize=12, fontweight="bold")

            # ── 右上：每批次缺陷数柱状图 ──
            ax2 = axes[0, 1]
            if ctrl._batch_stats:
                batch_nos = sorted(ctrl._batch_stats.keys())
                batch_defects = [ctrl._batch_stats[b].get("defects", 0) for b in batch_nos]
                batch_fails = [ctrl._batch_stats[b].get("failed", 0) for b in batch_nos]
                x = range(len(batch_nos))
                w = 0.35
                ax2.bar([i - w / 2 for i in x], batch_defects, w, label="缺陷数", color="#0078d4")
                ax2.bar([i + w / 2 for i in x], batch_fails, w, label="不合格数", color="#d13438")
                ax2.set_xticks(list(x))
                ax2.set_xticklabels([f"批次{b}" for b in batch_nos], fontsize=8)
                ax2.legend(fontsize=8)
            else:
                ax2.text(0.5, 0.5, "暂无批次数据", ha="center", va="center", transform=ax2.transAxes)
            ax2.set_title("每批次缺陷统计", fontsize=12, fontweight="bold")

            # ── 左下：样本合格趋势 ──
            ax3 = axes[1, 0]
            recent = ctrl._recent_passed[-50:]
            if len(recent) >= 2:
                ax3.plot(range(len(recent)), recent, color="#0078d4", linewidth=1.5, marker=".", markersize=4)
                ax3.axhline(y=0.5, color="#d13438", linestyle="--", linewidth=0.8, alpha=0.6)
                ax3.set_ylim(-0.1, 1.1)
                ax3.set_ylabel("合格(1) / 不合格(0)", fontsize=9)
            else:
                ax3.text(0.5, 0.5, "样本数不足", ha="center", va="center", transform=ax3.transAxes)
            ax3.set_title("近期合格趋势", fontsize=12, fontweight="bold")

            # ── 右下：缺陷尺寸直方图 ──
            ax4 = axes[1, 1]
            diameters = ctrl._defect_diameters
            if diameters:
                ax4.hist(diameters, bins=20, color="#107c10", edgecolor="white", alpha=0.8)
                ax4.set_xlabel("等效直径 (mm)", fontsize=9)
                ax4.set_ylabel("频次", fontsize=9)
            else:
                ax4.text(0.5, 0.5, "暂无缺陷尺寸数据", ha="center", va="center", transform=ax4.transAxes)
            ax4.set_title("缺陷尺寸分布", fontsize=12, fontweight="bold")

            # 渲染到内存缓冲区
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=110, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            self._chart_buf = buf

            # 显示
            pixmap = QPixmap()
            if pixmap.loadFromData(buf.getvalue()):
                self.chart_label.setPixmap(pixmap)
            else:
                self.chart_label.setText("图表渲染失败")
        except ImportError:
            self.chart_label.setText(
                "需要安装 matplotlib 才能生成图表。\n"
                "请在终端执行: pip install matplotlib"
            )
            self._chart_buf = None
        except Exception as e:
            self.chart_label.setText(f"图表生成错误:\n{str(e)}")
            self._chart_buf = None

    def _update_stats_text(self):
        ctrl = self.ctrl
        total = max(1, ctrl._total)
        qualified = ctrl.qualified
        unqualified = ctrl.unqualified
        pass_rate = qualified / total * 100

        # 批次摘要
        batch_lines = []
        for b in sorted(ctrl._batch_stats):
            bs = ctrl._batch_stats[b]
            review = "需复检" if bs.get("manual_review") else ("建议复检" if bs.get("suggested_review") else "正常")
            batch_lines.append(
                f"批次 {b:03d}: 样本={bs.get('samples',0)} 不合格={bs.get('failed',0)} "
                f"缺陷={bs.get('defects',0)} [{review}]"
            )

        html = (
            f"<h3>总体统计</h3>"
            f"<p>总样本数: {total} | 合格: {qualified} | "
            f"不合格: {unqualified} | 合格率: <b>{pass_rate:.1f}%</b></p>"
            f"<p>累计缺陷: {ctrl._total_defects} | "
            f"连续异常次数: {ctrl._fail_streak}</p>"
            f"<hr><h4>批次汇总 ({len(ctrl._batch_stats)} 批次)</h4>"
            f"<pre>{chr(10).join(batch_lines)}</pre>"
        )
        self.stats_text.setHtml(html)

    def _save_chart(self):
        """保存摘要图片到 dect_result 目录"""
        if self._chart_buf is None:
            QMessageBox.warning(self, "保存失败", "没有可保存的图表数据")
            return
        save_dir = self.ctrl.get_auto_save_dir(self.batch_source_dir)
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(save_dir, f"analysis_summary_{timestamp}.png")
        with open(path, "wb") as f:
            f.write(self._chart_buf.getvalue())
        QMessageBox.information(self, "保存完成", f"摘要图片已保存至:\n{path}")


# ══════════════════════════════════════════════════════════════════
#  主窗口框架
# ══════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("划痕缺陷检测系统 v6.1")
        self.resize(1920, 1080)
        self.setMinimumSize(1200, 800)
        self.setWindowFlags(Qt.FramelessWindowHint)

        # 限制窗口最大尺寸为屏幕可用区域，防止遮挡
        screen = QApplication.primaryScreen()
        if screen is not None:
            geometry = screen.availableGeometry()
            self.setMaximumSize(geometry.width(), geometry.height())

        icon_path = find_existing_path(
            str(BUNDLE_ROOT / "data" / "source_image" / "logo.ico"),
            str(BUNDLE_ROOT / "logo.ico"),
            str(APP_ROOT / "logo.ico"),
        )
        if icon_path:
            self.setWindowIcon(QIcon(icon_path))

        self._det_ref = [UnavailableDetector("默认 TensorRT 引擎尚未加载")]
        default_trt = BUNDLE_ROOT / "weights" / "best.trt"
        default_onnx = BUNDLE_ROOT / "weights" / "best.onnx"
        self._default_model_path = str(default_trt if default_trt.exists() else default_onnx)
        self._active_model_path = ""
        self._model_worker = None
        self._backend_preference = os.getenv("YOLO_BACKEND", "auto").strip().lower() or "auto"
        self._backend_failures = set()
        self._backend_successes = {"auto"}
        self._backend_reasons = {}

        root_w = QWidget()
        self.setCentralWidget(root_w)
        root_v = QVBoxLayout(root_w)
        root_v.setContentsMargins(0, 0, 0, 0)
        root_v.setSpacing(0)

        self.titlebar = TitleBar(self)
        root_v.addWidget(self.titlebar)

        content = QWidget()
        content_h = QHBoxLayout(content)
        content_h.setContentsMargins(0, 0, 0, 0)
        content_h.setSpacing(0)

        self.ctrl = ControlPanel()
        if os.path.exists(self._default_model_path):
            self.ctrl.model_panel.set_initial(self._default_model_path)
        self.ctrl.model_changed.connect(self._reload_model)
        self.ctrl.backend_changed.connect(self._on_backend_preference_changed)

        scroll_area = QScrollArea()
        self.left_scroll_area = scroll_area
        scroll_area.setWidget(self.ctrl)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setMinimumWidth(240)
        scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        QTimer.singleShot(0, self._sync_left_density)

        content_h.addWidget(scroll_area, 1)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("background:#c0c0c0; max-width:1px;")
        content_h.addWidget(sep, 0)

        self.right_splitter = QSplitter(Qt.Vertical)
        self.right_splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.right_splitter.setChildrenCollapsible(False)

        self.tabs = QTabWidget()
        self.tab_img = ImageTab(self._det_ref, self.ctrl)
        self.tab_cam = CameraTab(self._det_ref, self.ctrl)
        self.tab_vid = VideoTab(self._det_ref, self.ctrl)
        self.tabs.addTab(self.tab_img, "静态图像分析")
        self.tabs.addTab(self.tab_cam, "实时视频流监测")
        self.tabs.addTab(self.tab_vid, "媒体离线测算")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self._sync_backend_choice_widgets()
        self._sync_backend_availability_widgets()
        self.right_splitter.addWidget(self.tabs)

        log_widget = QWidget()
        log_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        log_vbox = QVBoxLayout(log_widget)
        log_vbox.setContentsMargins(2, 1, 2, 2)
        log_vbox.setSpacing(2)

        log_group = QGroupBox("日志")
        self.log_group = log_group
        log_group.setStyleSheet("QGroupBox { border: 2px solid #c0c0c0; border-radius: 8px; margin-top: 8px; padding-top: 14px; background-color: #ffffff; } QGroupBox::title { subcontrol-origin: padding; subcontrol-position: top left; padding: 0 8px; margin-top: -7px; color: #0078d4; font-weight: bold; font-size: 20px; background-color: #f0f2f5; }")
        lg_lay = QVBoxLayout(log_group)
        lg_lay.setSpacing(3)
        lg_lay.setStretch(0, 1)  # log_edit gets stretch

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # 启用垂直滚动条
        lg_lay.addWidget(self.log_edit)

        btn_log_layout = QHBoxLayout()
        btn_log_layout.setSpacing(4)
        self.btn_export_log = QPushButton("导出日志")
        self.btn_export_log.setObjectName("btn_batch_neutral")
        self.btn_export_log.setFixedHeight(48)
        self.btn_clear_log = QPushButton("清除日志")
        self.btn_clear_log.setObjectName("btn_batch_neutral")
        self.btn_clear_log.setFixedHeight(48)
        btn_log_layout.addWidget(self.btn_export_log)
        btn_log_layout.addWidget(self.btn_clear_log)
        lg_lay.addLayout(btn_log_layout)

        log_vbox.addWidget(log_group)

        # 日志区域和创新功能区域水平排列
        self.bottom_widget = QWidget()
        self.bottom_widget.setMinimumHeight(280)
        self.bottom_widget.setMaximumHeight(460)
        bottom_h = QHBoxLayout(self.bottom_widget)
        bottom_h.setContentsMargins(2, 1, 2, 1)
        bottom_h.setSpacing(3)

        # 日志区域 - 50%宽度
        log_widget.setMinimumWidth(180)
        bottom_h.addWidget(log_widget, 1)

        # 分隔符
        sep_v = QFrame()
        sep_v.setFrameShape(QFrame.VLine)
        sep_v.setStyleSheet("background:#c0c0c0;")
        bottom_h.addWidget(sep_v, 0)

        # 批处理面板 - 50%宽度，垂直按钮布局
        batch_panel = QGroupBox("自动化批处理")
        self.batch_panel = batch_panel
        batch_panel.setStyleSheet("QGroupBox { border: 2px solid #c0c0c0; border-radius: 8px; margin-top: 8px; padding-top: 14px; background-color: #ffffff; } QGroupBox::title { subcontrol-origin: padding; subcontrol-position: top left; padding: 0 8px; margin-top: -7px; color: #0078d4; font-weight: bold; font-size: 20px; background-color: #f0f2f5; }")
        batch_vbox = QVBoxLayout(batch_panel)
        self.batch_layout = batch_vbox
        batch_vbox.setContentsMargins(8, 8, 8, 8)
        batch_vbox.setSpacing(7)

        # 进度条（横贯宽度）
        self.batch_bar = QProgressBar()
        self.batch_bar.setRange(0, 100)
        self.batch_bar.setValue(0)
        self.batch_bar.setFormat("%p%")
        self.batch_bar.setFixedHeight(20)
        batch_vbox.addWidget(self.batch_bar)

        # 垂直排列的按钮（大小一致）
        btn_width = 180
        btn_height = 56

        self.btn_folder = QPushButton("指定数据目录")
        self.btn_folder.setObjectName("btn_batch_compact")
        self.btn_folder.setFixedHeight(btn_height)
        self.btn_folder.setMinimumWidth(btn_width)
        self.btn_batch = QPushButton("执行批量检测")
        self.btn_batch.setObjectName("btn_batch_compact")
        self.btn_batch.setFixedHeight(btn_height)
        self.btn_batch.setMinimumWidth(btn_width)
        self.btn_batch.setEnabled(False)
        self.btn_abort = QPushButton("强行中止任务")
        self.btn_abort.setObjectName("btn_batch_compact")
        self.btn_abort.setFixedHeight(btn_height)
        self.btn_abort.setMinimumWidth(btn_width)
        self.btn_abort.setEnabled(False)
        self.btn_pause = QPushButton("暂停/继续任务")
        self.btn_pause.setObjectName("btn_batch_compact")
        self.btn_pause.setFixedHeight(btn_height)
        self.btn_pause.setMinimumWidth(btn_width)
        self.btn_pause.setEnabled(False)
        self.btn_quick_snap = QPushButton("截图保存")
        self.btn_quick_snap.setObjectName("btn_batch_compact")
        self.btn_quick_snap.setFixedHeight(btn_height)
        self.btn_quick_snap.setMinimumWidth(btn_width)
        self.btn_quick_snap.setEnabled(False)
        self.btn_clear_batch = QPushButton("清空检测日志")
        self.btn_clear_batch.setObjectName("btn_batch_compact")
        self.btn_clear_batch.setFixedHeight(btn_height)
        self.btn_clear_batch.setMinimumWidth(btn_width)
        self.btn_clear_batch.clicked.connect(self.ctrl.clear_log)

        self.btn_folder.clicked.connect(self.tab_img.select_folder)
        self.btn_batch.clicked.connect(self.tab_img.start_batch)
        self.btn_abort.clicked.connect(self.tab_img.abort_batch)
        self.btn_pause.clicked.connect(self.tab_img.toggle_pause)
        self.btn_quick_snap.clicked.connect(self.tab_img.quick_screenshot)

        self.batch_buttons = [
            self.btn_folder, self.btn_batch, self.btn_abort,
            self.btn_pause, self.btn_quick_snap, self.btn_clear_batch
        ]
        self.batch_button_grid = QGridLayout()
        self.batch_button_grid.setContentsMargins(0, 0, 0, 0)
        self.batch_button_grid.setHorizontalSpacing(6)
        self.batch_button_grid.setVerticalSpacing(6)
        for i, btn in enumerate(self.batch_buttons):
            self.batch_button_grid.addWidget(btn, i // 2, i % 2)
        batch_vbox.addLayout(self.batch_button_grid)

        # 目录显示
        self.lbl_folder = QLabel("未指定")
        self.lbl_folder.setMinimumHeight(28)
        self.lbl_folder.setStyleSheet("color:#5a5a5a; font-size:20px; font-weight:bold; padding:2px 4px; background:#f5f7fa; border-radius:4px;")
        self.lbl_folder.setWordWrap(True)
        batch_vbox.addWidget(self.lbl_folder)


        # 将批处理控件关联到ImageTab
        self.tab_img.btn_folder = self.btn_folder
        self.tab_img.btn_batch = self.btn_batch
        self.tab_img.btn_abort = self.btn_abort
        self.tab_img.btn_pause = self.btn_pause
        self.tab_img.btn_quick_snap = self.btn_quick_snap
        self.tab_img.lbl_folder = self.lbl_folder
        self.tab_img.batch_bar = self.batch_bar

        batch_panel.setMinimumWidth(160)
        batch_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.tools_stack = QStackedWidget()
        self.context_tool_panels = []
        self.tools_stack.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.tools_stack.addWidget(batch_panel)
        self.tools_stack.addWidget(self._make_context_tools_panel(
            "实时质控辅助",
            [
                "节拍观察: 关注连续不合格、漏检波动和相机掉帧",
                "实时加速: 摄像头采集 1280x720, 推理尺寸默认 512",
                "工位建议: 光源、焦距、治具偏移异常时及时标记当前帧",
            ],
            [
                ("标记当前帧", lambda: self._append_panel_note("摄像头画面已人工标记，建议复核光源/焦距/治具状态")),
                ("保存当前帧", self.tab_cam.snapshot),
                ("清空检测日志", self.ctrl.clear_log),
            ],
        ))
        self.tools_stack.addWidget(self._make_context_tools_panel(
            "媒体回放分析",
            [
                "异常片段: 结合日志定位缺陷高发时间段",
                "抽帧复核: 对不合格帧做人工确认和追溯",
                "速度策略: 离线视频同样使用实时推理尺寸, 优先保证流畅回放",
            ],
            [
                ("标记当前帧", self.tab_vid.mark_current_frame),
                ("导出检测日志", self.ctrl.export_csv),
                ("清空检测日志", self.ctrl.clear_log),
            ],
        ))
        bottom_h.addWidget(self.tools_stack, 1)

        self.right_splitter.addWidget(self.bottom_widget)
        self.right_splitter.setStretchFactor(0, 5)
        self.right_splitter.setStretchFactor(1, 2)

        # 图像区域占更多空间，日志+批处理平分下半部分
        self.right_splitter.setSizes([780, 300])

        content_h.addWidget(self.right_splitter, 4)

        self.ctrl.attach_log_widget(self.log_edit, self.btn_export_log, self.btn_clear_log)

        root_v.addWidget(content, 1)

        # 确保标题栏在最上层
        self.titlebar.raise_()

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        QTimer.singleShot(0, self._sync_window_density)
        self.status.showMessage(f"系统运行就绪  |  {datetime.now().strftime('%Y-%m-%d')}")
        if os.getenv("UI_SKIP_INITIAL_MODEL_LOAD", "0").strip().lower() not in {"1", "true", "yes", "on"}:
            QTimer.singleShot(int(os.getenv("UI_INITIAL_MODEL_LOAD_DELAY_MS", "350")), self._load_initial_model)

    def _make_context_tools_panel(self, title, notes, actions):
        panel = QGroupBox(title)
        panel.setStyleSheet("QGroupBox { border: 2px solid #c0c0c0; border-radius: 8px; margin-top: 8px; padding-top: 14px; background-color: #ffffff; } QGroupBox::title { subcontrol-origin: padding; subcontrol-position: top left; padding: 0 8px; margin-top: -7px; color: #0078d4; font-weight: bold; font-size: 20px; background-color: #f0f2f5; }")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        note_labels = []
        action_buttons = []

        for note in notes:
            lbl = QLabel(note)
            lbl.setWordWrap(True)
            lbl.setStyleSheet("color:#1a1a1a; font-size:20px; font-weight:bold; padding:8px 10px; background:#f5f7fa; border:1px solid #d7dde5; border-radius:6px;")
            layout.addWidget(lbl)
            note_labels.append(lbl)

        layout.addStretch()
        for text, callback in actions:
            btn = QPushButton(text)
            btn.setObjectName("btn_batch_compact")
            btn.setFixedHeight(56)
            if callback:
                btn.clicked.connect(callback)
            layout.addWidget(btn)
            action_buttons.append(btn)
        if hasattr(self, "context_tool_panels"):
            self.context_tool_panels.append((panel, layout, note_labels, action_buttons))
        return panel

    def _append_panel_note(self, text):
        if self.log_edit:
            self.log_edit.append(
                f"<span style='color:#0078d4; font-weight:bold;'>[{datetime.now().strftime('%H:%M:%S')}] {text}</span>"
            )

    def _on_tab_changed(self, index):
        if index != 1:
            self.tab_cam.stop()
        if index != 2:
            self.tab_vid.stop_video(release=False, summarize=False)
        self.ctrl.reset_stats()
        if hasattr(self, "tools_stack"):
            self.tools_stack.setCurrentIndex(index)

    def _sync_left_density(self):
        if hasattr(self, "left_scroll_area"):
            self.ctrl.apply_density(self.left_scroll_area.viewport().height())

    def _sync_window_density(self):
        if not hasattr(self, "bottom_widget"):
            return
        screen = QApplication.primaryScreen()
        available_h = screen.availableGeometry().height() if screen else self.height()
        effective_h = min(self.height() or available_h, available_h)
        profile = "compact" if effective_h <= 1120 else "roomy"
        if getattr(self, "_window_density_profile", None) == profile:
            return
        self._window_density_profile = profile

        if profile == "compact":
            cfg = {
                "bottom_min": 268, "bottom_max": 390, "splitter": [760, 300],
                "group_title": 16, "group_mt": 7, "group_pt": 11,
                "batch_h": 34, "batch_font": 16, "batch_gap": 5, "batch_margin": (6, 7, 6, 6),
                "progress_h": 18, "folder_h": 30, "folder_font": 14,
                "log_btn_h": 34, "log_font": 16, "tab_font": 15, "tab_h": 30,
                "note_font": 14, "note_pad": "4px 6px", "context_btn_h": 34,
            }
        else:
            cfg = {
                "bottom_min": 300, "bottom_max": 620, "splitter": [780, 420],
                "group_title": 20, "group_mt": 8, "group_pt": 14,
                "batch_h": 56, "batch_font": 28, "batch_gap": 7, "batch_margin": (8, 8, 8, 8),
                "progress_h": 20, "folder_h": 28, "folder_font": 20,
                "log_btn_h": 48, "log_font": 20, "tab_font": 18, "tab_h": 38,
                "note_font": 20, "note_pad": "8px 10px", "context_btn_h": 56,
            }

        def group_style(title_size):
            return (
                f"QGroupBox {{ border: 2px solid #c0c0c0; border-radius: 8px; margin-top: {cfg['group_mt']}px; "
                f"padding-top: {cfg['group_pt']}px; background-color: #ffffff; }} "
                f"QGroupBox::title {{ subcontrol-origin: padding; subcontrol-position: top left; padding: 0 8px; "
                f"margin-top: 0px; color: #0078d4; font-weight: bold; font-size: {title_size}px; background-color: #f0f2f5; }}"
            )

        self.bottom_widget.setMinimumHeight(cfg["bottom_min"])
        self.bottom_widget.setMaximumHeight(cfg["bottom_max"])
        self.right_splitter.setSizes(cfg["splitter"])
        self.tabs.setStyleSheet(
            f"QTabBar::tab {{ min-height: {cfg['tab_h']}px; padding: 4px 14px; font-size: {cfg['tab_font']}px; font-weight: bold; }}"
        )

        for group in (getattr(self, "log_group", None), getattr(self, "batch_panel", None)):
            if group:
                group.setStyleSheet(group_style(cfg["group_title"]))
        if hasattr(self, "batch_layout"):
            self.batch_layout.setContentsMargins(*cfg["batch_margin"])
            self.batch_layout.setSpacing(cfg["batch_gap"])
        if hasattr(self, "batch_button_grid"):
            self.batch_button_grid.setHorizontalSpacing(cfg["batch_gap"])
            self.batch_button_grid.setVerticalSpacing(cfg["batch_gap"])
        if hasattr(self, "batch_bar"):
            self.batch_bar.setFixedHeight(cfg["progress_h"])
        for btn in getattr(self, "batch_buttons", []):
            btn.setFixedHeight(cfg["batch_h"])
            btn.setMinimumWidth(0)
            btn.setStyleSheet(
                f"font-size:{cfg['batch_font']}px; min-height:0px; max-height:{cfg['batch_h']}px; padding:2px 6px;"
            )
        for btn in (getattr(self, "btn_export_log", None), getattr(self, "btn_clear_log", None)):
            if btn:
                btn.setFixedHeight(cfg["log_btn_h"])
                btn.setStyleSheet(
                    f"font-size:{cfg['log_font']}px; min-height:0px; max-height:{cfg['log_btn_h']}px; padding:2px 8px;"
                )
        if hasattr(self, "lbl_folder"):
            self.lbl_folder.setMinimumHeight(cfg["folder_h"])
            self.lbl_folder.setStyleSheet(
                f"color:#5a5a5a; font-size:{cfg['folder_font']}px; font-weight:bold; padding:2px 4px; background:#f5f7fa; border-radius:4px;"
            )

        for panel, layout, labels, buttons in getattr(self, "context_tool_panels", []):
            panel.setStyleSheet(group_style(cfg["group_title"]))
            layout.setContentsMargins(*cfg["batch_margin"])
            layout.setSpacing(cfg["batch_gap"])
            for lbl in labels:
                lbl.setStyleSheet(
                    f"color:#1a1a1a; font-size:{cfg['note_font']}px; font-weight:bold; padding:{cfg['note_pad']}; background:#f5f7fa; border:1px solid #d7dde5; border-radius:6px;"
                )
            for btn in buttons:
                btn.setFixedHeight(cfg["context_btn_h"])
                btn.setStyleSheet(
                    f"font-size:{cfg['batch_font']}px; min-height:0px; max-height:{cfg['context_btn_h']}px; padding:2px 6px;"
                )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._sync_left_density()
        self._sync_window_density()

    def open_settings_from_titlebar(self):
        """从标题栏设置按钮打开检测设置对话框"""
        dlg = DetectionSettingsDialog(
            self,
            conf=self.ctrl._conf,
            iou=self.ctrl._iou,
            grayscale=self.ctrl._gray,
            exposure=self.ctrl._exposure,
            filter_settings=self.ctrl.get_filter_settings()
        )
        dlg.settings_changed.connect(self.ctrl.on_settings_changed)
        dlg.exec_()

    def _append_runtime_log(self, message: str, color: str = "#0078d4"):
        if self.log_edit:
            self.log_edit.append(
                f"<span style='color:{color}; font-weight:bold;'>[{datetime.now().strftime('%H:%M:%S')}] {message}</span>"
            )

    def _set_runtime_indicator(self, backend: str, device: str = "", color: str = "#5a5a5a"):
        text = f"后端: {backend}" if not device else f"后端: {backend} / {device}"
        for tab_name in ("tab_img", "tab_cam", "tab_vid"):
            tab = getattr(self, tab_name, None)
            if tab and hasattr(tab, "_set_backend_badge"):
                tab._set_backend_badge(text, color)
            if tab and hasattr(tab, "_set_backend_runtime_text"):
                tab._set_backend_runtime_text(text)
        self.ctrl.model_panel.set_runtime_status(text, color)

    def _sync_backend_choice_widgets(self):
        for tab_name in ("tab_img", "tab_cam", "tab_vid"):
            tab = getattr(self, tab_name, None)
            if tab and hasattr(tab, "_set_backend_choice"):
                tab._set_backend_choice(self._backend_preference)

    def _on_backend_preference_changed(self, preference: str):
        preference = (preference or "auto").strip().lower()
        if preference == self._backend_preference:
            self._sync_backend_choice_widgets()
            return
        if self._model_worker is not None and self._model_worker.isRunning():
            self._append_runtime_log("模型仍在后台加载，请稍候再切换后端。", "#ff8c00")
            self._sync_backend_choice_widgets()
            return
        self._backend_preference = preference
        self._sync_backend_choice_widgets()
        active_path = self._active_model_path or self._default_model_path
        current = self._det_ref[0] if self._det_ref else None
        current_key = self._backend_key_for_runtime(
            getattr(current, "backend_name", ""),
            getattr(current, "device_name", ""),
        ) if current is not None else ""
        if preference != "auto" and current_key == preference:
            self._append_runtime_log(f"当前已经是 {self._backend_label(preference)} 后端，无需重新加载。", "#107c10")
            return
        if active_path and os.path.exists(active_path):
            self._append_runtime_log(f"切换推理后端偏好: {self._backend_label(preference)}", "#0078d4")
            self.tab_cam.stop()
            self.tab_vid.stop_video(release=False, summarize=False)
            self._start_model_load_worker(active_path, allow_cpu_fallback=False, show_dialog=False)

    def _backend_label(self, preference: str) -> str:
        for label, value in BACKEND_CHOICES:
            if value == preference:
                return label
        return "自动优先"

    def _backend_load_action(self, preference: str, allow_cpu_fallback: bool) -> str:
        if preference == "auto":
            return "CPU 回退加载" if allow_cpu_fallback else "GPU/TensorRT 后台加载"
        return f"{self._backend_label(preference)} 后台加载"

    def _backend_min_visible_ms(self, preference: str, allow_cpu_fallback: bool) -> int:
        if allow_cpu_fallback or preference in {"onnx_cpu", "opencv_cpu"}:
            return MODEL_LOAD_MIN_VISIBLE_MS_CPU
        return MODEL_LOAD_MIN_VISIBLE_MS_GPU

    def _backend_key_for_runtime(self, backend: str, device: str = "") -> str:
        name = f"{backend} {device}".upper()
        if "TENSORRT" in name:
            return "tensorrt"
        if "ONNX RUNTIME CUDA" in name:
            return "onnx_cuda"
        if "ONNX RUNTIME CPU" in name:
            return "onnx_cpu"
        if "OPENCV" in name:
            return "opencv_cpu"
        return ""

    def _backend_key_from_attempt(self, attempt: str) -> str:
        text = str(attempt).upper()
        if text.startswith("TENSORRT"):
            return "tensorrt"
        if text.startswith("ONNX RUNTIME CUDA"):
            return "onnx_cuda"
        if text.startswith("ONNX RUNTIME CPU"):
            return "onnx_cpu"
        if text.startswith("OPENCV DNN CPU"):
            return "opencv_cpu"
        return ""

    def _current_model_family_paths(self):
        model_path = Path(self._active_model_path or self._default_model_path)
        if model_path.suffix.lower() == ".onnx":
            return model_path.with_suffix(".trt"), model_path
        return model_path, model_path.with_suffix(".onnx")

    def _backend_availability(self):
        trt_path, onnx_path = self._current_model_family_paths()
        availability = {"auto": True}
        reasons = {"auto": "自动尝试 GPU，失败后回退 CPU"}
        for value in ("tensorrt", "onnx_cuda", "onnx_cpu", "opencv_cpu"):
            availability[value] = value not in self._backend_failures
            reasons[value] = "可选择"
        if not trt_path.exists() and "tensorrt" not in self._backend_successes:
            availability["tensorrt"] = False
            reasons["tensorrt"] = "尚未生成可用 TensorRT 引擎"
        if not onnx_path.exists():
            for value in ("onnx_cuda", "onnx_cpu", "opencv_cpu"):
                availability[value] = False
                reasons[value] = "缺少 companion ONNX 模型"
        for value in self._backend_failures:
            if value in availability and value not in self._backend_successes:
                availability[value] = False
                reasons[value] = "上次加载失败，修复环境或自动加载成功后可选"
        for value in self._backend_successes:
            if value in availability:
                availability[value] = True
                reasons[value] = "已验证可用"
        return availability, reasons

    def _sync_backend_availability_widgets(self):
        availability, reasons = self._backend_availability()
        for tab_name in ("tab_img", "tab_cam", "tab_vid"):
            tab = getattr(self, tab_name, None)
            if tab and hasattr(tab, "_set_backend_availability"):
                tab._set_backend_availability(availability, reasons)

    def _record_backend_result(self, detector=None, error=None):
        if detector is not None:
            key = self._backend_key_for_runtime(
                getattr(detector, "backend_name", ""),
                getattr(detector, "device_name", ""),
            )
            if key:
                self._backend_successes.add(key)
                self._backend_failures.discard(key)
            attempts = list(getattr(detector, "gpu_attempts", [])) + list(getattr(detector, "cpu_attempts", []))
            for attempt in attempts:
                failed_key = self._backend_key_from_attempt(attempt)
                if failed_key and failed_key not in self._backend_successes:
                    self._backend_failures.add(failed_key)
        if error is not None:
            for attempt in getattr(error, "gpu_attempts", []):
                failed_key = self._backend_key_from_attempt(attempt)
                if failed_key and failed_key not in self._backend_successes:
                    self._backend_failures.add(failed_key)
            pref = getattr(self, "_pending_model_backend_preference", "auto")
            if pref != "auto":
                self._backend_failures.add(pref)
                self._backend_successes.discard(pref)
        self._sync_backend_availability_widgets()

    def _prompt_cpu_fallback(self, error: CpuFallbackRequiredError) -> bool:
        detail_lines = error.gpu_attempts[-2:] if error.gpu_attempts else []
        detail = "\n".join(detail_lines)
        message = (
            "当前机器未能建立 CUDA 推理链路。\n\n"
            "可以改用 CPU 推理继续，但速度会明显下降。\n"
            "选择“是”继续，选择“否”则本次不能继续检测。"
        )
        if detail:
            message += f"\n\n最近错误:\n{detail}"
        reply = QMessageBox.question(
            self,
            "CUDA 不可用",
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return reply == QMessageBox.Yes

    def _load_initial_model(self):
        fallback_path = self.ctrl.model_panel._history[0] if self.ctrl.model_panel._history else ""
        initial_path = self._default_model_path if os.path.exists(self._default_model_path) else fallback_path
        if initial_path and os.path.exists(initial_path):
            self._reload_model(initial_path, show_dialog=False)
        else:
            msg = f"未找到默认模型文件: {self._default_model_path}"
            self._det_ref[0] = UnavailableDetector(msg)
            self.status.showMessage(msg)
            self._append_runtime_log(msg, "#d13438")
            self._set_runtime_indicator("未加载", "", "#d13438")

    def _reload_model(self, path: str, show_dialog: bool = True):
        path = os.path.normpath(path)
        if not os.path.exists(path):
            message = f"模型文件不存在:\n{path}"
            if show_dialog:
                QMessageBox.warning(self, "链路异常", message)
            if isinstance(self._det_ref[0], UnavailableDetector):
                self._det_ref[0] = UnavailableDetector(message)
            self.status.showMessage("默认引擎未加载，等待选择有效模型")
            self._append_runtime_log(message, "#d13438")
            self._set_runtime_indicator("未加载", "", "#d13438")
            return
        self._start_model_load_worker(path, allow_cpu_fallback=False, show_dialog=show_dialog)
        return
        try:
            self.status.showMessage(f"内存装载中: {os.path.basename(path)} ...")
            QApplication.processEvents()
            new_detector = v5detect(model_path=path, allow_cpu_fallback=False)
            old_detector = self._det_ref[0]
            self._det_ref[0] = new_detector
            self._active_model_path = path
            if old_detector and hasattr(old_detector, "close"):
                try:
                    old_detector.close()
                except Exception:
                    pass
            for note in getattr(new_detector, "load_notes", []):
                self._append_runtime_log(note, "#5c2d91")
            loaded_model_name = os.path.basename(path)
            if getattr(new_detector, "engine_generated_from_onnx", False):
                loaded_model_name = f"{os.path.basename(path)} → {new_detector.model_path.name}"
            self.status.showMessage(
                f"视觉分析引擎成功切换模型 → {loaded_model_name}  |  {datetime.now().strftime('%Y-%m-%d')}")
            self._append_runtime_log(f"视觉分析引擎已加载: {path}", "#107c10")
            self._append_runtime_log(f"当前推理后端: {new_detector.backend_name} / {new_detector.device_name}", "#107c10")
            backend_color = "#107c10" if "CPU" not in new_detector.device_name.upper() else "#d13438"
            self._set_runtime_indicator(new_detector.backend_name, new_detector.device_name, backend_color)
            if getattr(new_detector, "engine_generated_from_onnx", False):
                self._append_runtime_log(f"本机 TensorRT 引擎路径: {new_detector.model_path}", "#107c10")
        except CpuFallbackRequiredError as e:
            self._append_runtime_log("CUDA 推理链路不可用，准备切换 CPU 回退。", "#d13438")
            for attempt in getattr(e, "gpu_attempts", []):
                self._append_runtime_log(attempt, "#d13438")
            self._set_runtime_indicator("CUDA 不可用", "", "#d13438")
            if self._prompt_cpu_fallback(e):
                try:
                    self.status.showMessage(f"切换 CPU 推理中: {os.path.basename(path)} ...")
                    QApplication.processEvents()
                    new_detector = v5detect(model_path=path, allow_cpu_fallback=True)
                    old_detector = self._det_ref[0]
                    self._det_ref[0] = new_detector
                    self._active_model_path = path
                    if old_detector and hasattr(old_detector, "close"):
                        try:
                            old_detector.close()
                        except Exception:
                            pass
                    for note in getattr(new_detector, "load_notes", []):
                        self._append_runtime_log(note, "#5c2d91")
                    self.status.showMessage(
                        f"视觉分析引擎已切换 CPU 模式 → {os.path.basename(path)}  |  {datetime.now().strftime('%Y-%m-%d')}"
                    )
                    self._append_runtime_log(f"当前推理后端: {new_detector.backend_name} / {new_detector.device_name}", "#d13438")
                    self._set_runtime_indicator(new_detector.backend_name, new_detector.device_name, "#d13438")
                except Exception as cpu_error:
                    if show_dialog:
                        QMessageBox.critical(self, "CPU 回退失败", str(cpu_error))
                    self._det_ref[0] = UnavailableDetector(str(cpu_error))
                    self.status.showMessage("CPU 回退失败，当前无法继续检测")
                    self._append_runtime_log(f"CPU 回退失败: {cpu_error}", "#d13438")
                    self._set_runtime_indicator("CPU 回退失败", "", "#d13438")
            else:
                message = "用户拒绝 CPU 回退，当前不允许继续检测。"
                if show_dialog:
                    QMessageBox.warning(self, "无法继续", message)
                self._det_ref[0] = UnavailableDetector(message)
                self.status.showMessage(message)
                self._append_runtime_log(message, "#d13438")
                self._set_runtime_indicator("未加载", "", "#d13438")
        except Exception as e:
            if show_dialog:
                QMessageBox.critical(self, "异常中断", str(e))
            self._det_ref[0] = UnavailableDetector(str(e))
            self.status.showMessage("主分析引擎加载失败，请检查 TensorRT/CUDA/模型兼容性")
            self._append_runtime_log(f"模型加载失败: {e}", "#d13438")
            self._set_runtime_indicator("加载失败", "", "#d13438")

    def _start_model_load_worker(self, path: str, allow_cpu_fallback: bool = False, show_dialog: bool = True):
        if self._model_worker is not None and self._model_worker.isRunning():
            self._append_runtime_log("模型仍在后台加载，请稍候。", "#ff8c00")
            return

        self._pending_model_show_dialog = show_dialog
        self._pending_model_allow_cpu = allow_cpu_fallback
        self._det_ref[0] = UnavailableDetector("模型正在后台加载，请稍候。")
        self.status.showMessage(f"后台加载模型中: {os.path.basename(path)} ...")
        self._set_runtime_indicator("加载中", "", "#ff8c00")
        self._append_runtime_log(f"后台加载模型: {path}", "#0078d4")

        worker = ModelLoadWorker(
            path,
            allow_cpu_fallback=allow_cpu_fallback,
            backend_preference="auto" if allow_cpu_fallback else self._backend_preference,
            min_visible_ms=self._backend_min_visible_ms(
                "auto" if allow_cpu_fallback else self._backend_preference,
                allow_cpu_fallback,
            ),
            parent=self,
        )
        worker.sig_loaded.connect(self._on_model_worker_loaded)
        worker.sig_failed.connect(self._on_model_worker_failed)
        worker.finished.connect(worker.deleteLater)
        self._model_worker = worker
        worker.start()

    def _on_model_worker_loaded(self, path: str, new_detector):
        old_detector = self._det_ref[0]
        self._det_ref[0] = new_detector
        self._active_model_path = path
        self._model_worker = None
        self._record_backend_result(detector=new_detector)
        if old_detector and hasattr(old_detector, "close"):
            try:
                old_detector.close()
            except Exception:
                pass
        for note in getattr(new_detector, "load_notes", []):
            self._append_runtime_log(note, "#5c2d91")
        self.status.showMessage(f"模型已加载: {os.path.basename(path)}  |  {datetime.now().strftime('%Y-%m-%d')}")
        self._append_runtime_log(f"模型已加载: {path}", "#107c10")
        self._append_runtime_log(f"当前推理后端: {new_detector.backend_name} / {new_detector.device_name}", "#107c10")
        if getattr(new_detector, "engine_generated_from_onnx", False):
            self._append_runtime_log(f"本机 TensorRT 引擎路径: {new_detector.model_path}", "#107c10")
        backend_color = "#107c10" if "CPU" not in new_detector.device_name.upper() else "#d13438"
        self._set_runtime_indicator(new_detector.backend_name, new_detector.device_name, backend_color)

    def _on_model_worker_failed(self, path: str, error):
        self._model_worker = None
        self._record_backend_result(error=error)
        show_dialog = getattr(self, "_pending_model_show_dialog", True)
        if isinstance(error, CpuFallbackRequiredError):
            self._append_runtime_log("CUDA 推理链路不可用，准备切换 CPU 回退。", "#d13438")
            for attempt in getattr(error, "gpu_attempts", []):
                self._append_runtime_log(attempt, "#d13438")
            self._set_runtime_indicator("CUDA 不可用", "", "#d13438")
            if self._prompt_cpu_fallback(error):
                self._start_model_load_worker(path, allow_cpu_fallback=True, show_dialog=show_dialog)
            else:
                message = "用户拒绝 CPU 回退，当前不能继续检测。"
                self._det_ref[0] = UnavailableDetector(message)
                self.status.showMessage(message)
                self._append_runtime_log(message, "#d13438")
                self._set_runtime_indicator("未加载", "", "#d13438")
            return

        message = f"模型加载失败: {error}"
        if show_dialog:
            QMessageBox.critical(self, "模型加载失败", str(error))
        self._det_ref[0] = UnavailableDetector(str(error))
        self.status.showMessage("模型加载失败")
        self._append_runtime_log(message, "#d13438")
        self._set_runtime_indicator("加载失败", "", "#d13438")

    def _start_model_load_worker(self, path: str, allow_cpu_fallback: bool = False, show_dialog: bool = True):
        if self._model_worker is not None and self._model_worker.isRunning():
            self._append_runtime_log("模型仍在后台加载，请稍候。", "#ff8c00")
            return

        self._pending_model_show_dialog = show_dialog
        self._pending_model_allow_cpu = allow_cpu_fallback
        preference = "auto" if allow_cpu_fallback else self._backend_preference
        self._pending_model_backend_preference = self._backend_preference
        self._model_worker_kept_detector = not isinstance(self._det_ref[0], UnavailableDetector)
        if not self._model_worker_kept_detector:
            self._det_ref[0] = UnavailableDetector("模型正在后台加载，请稍候。")

        action = "CPU 回退加载" if allow_cpu_fallback else "GPU/TensorRT 后台加载"
        preference = "auto" if allow_cpu_fallback else self._backend_preference
        action = self._backend_load_action(preference, allow_cpu_fallback)
        self.status.showMessage(f"{action}: {os.path.basename(path)} ...")
        self._set_runtime_indicator("后台加载中", "", "#ff8c00")
        self._append_runtime_log(f"{action}: {path}", "#0078d4")
        model_path = Path(path)
        trt_path = model_path.with_suffix(".trt") if model_path.suffix.lower() == ".onnx" else model_path
        onnx_path = model_path.with_suffix(".onnx") if model_path.suffix.lower() == ".trt" else model_path
        if not allow_cpu_fallback and onnx_path.exists() and (
            model_path.suffix.lower() == ".onnx" or not trt_path.exists()
        ):
            self._append_runtime_log("首次 TensorRT 引擎生成可能较久；加载在后台进行，界面可继续操作。", "#ff8c00")

        worker = ModelLoadWorker(
            path,
            allow_cpu_fallback=allow_cpu_fallback,
            backend_preference=preference,
            min_visible_ms=self._backend_min_visible_ms(preference, allow_cpu_fallback),
            parent=self,
        )
        worker.sig_loaded.connect(self._on_model_worker_loaded)
        worker.sig_failed.connect(self._on_model_worker_failed)
        worker.finished.connect(worker.deleteLater)
        self._model_worker = worker
        worker.start()

    def _on_model_worker_loaded(self, path: str, new_detector):
        old_detector = self._det_ref[0]
        old_device = getattr(old_detector, "device_name", "")
        self._det_ref[0] = new_detector
        self._active_model_path = path
        self._model_worker = None
        self._record_backend_result(detector=new_detector)
        if old_detector and hasattr(old_detector, "close"):
            try:
                old_detector.close()
            except Exception:
                pass
        for note in getattr(new_detector, "load_notes", []):
            self._append_runtime_log(note, "#5c2d91")
        self.status.showMessage(f"模型已加载: {os.path.basename(path)}  |  {datetime.now().strftime('%Y-%m-%d')}")
        self._append_runtime_log(f"模型已加载: {path}", "#107c10")
        self._append_runtime_log(f"当前推理后端: {new_detector.backend_name} / {new_detector.device_name}", "#107c10")
        if getattr(new_detector, "engine_generated_from_onnx", False):
            self._append_runtime_log(f"本机 TensorRT 引擎路径: {new_detector.model_path}", "#107c10")
        backend_color = "#107c10" if "CPU" not in new_detector.device_name.upper() else "#d13438"
        self._set_runtime_indicator(new_detector.backend_name, new_detector.device_name, backend_color)
        if "CPU" in str(new_detector.device_name).upper():
            self._append_runtime_log("当前使用 CPU 回退；CUDA/TensorRT 环境修复后重新加载模型，会自动优先切回 GPU。", "#ff8c00")
        if "CPU" in str(old_device).upper() and "CPU" not in str(new_detector.device_name).upper():
            QMessageBox.information(
                self,
                "GPU 后端已就绪",
                f"后台 GPU/TensorRT 后端已加载完成，当前已切换到:\n{new_detector.backend_name} / {new_detector.device_name}"
            )

    def _on_model_worker_failed(self, path: str, error):
        self._model_worker = None
        self._record_backend_result(error=error)
        show_dialog = getattr(self, "_pending_model_show_dialog", True)
        allow_cpu = getattr(self, "_pending_model_allow_cpu", False)
        kept_detector = getattr(self, "_model_worker_kept_detector", False)
        if isinstance(error, CpuFallbackRequiredError):
            fallback_note = (
                "CUDA 推理链路不可用，将自动切换 CPU 回退。"
                if not show_dialog and not allow_cpu
                else "CUDA 推理链路不可用，等待用户决定是否切换 CPU。"
            )
            self._append_runtime_log(fallback_note, "#d13438")
            for attempt in getattr(error, "gpu_attempts", []):
                self._append_runtime_log(attempt, "#d13438")
            if kept_detector:
                self.status.showMessage("GPU 后台加载失败，继续使用当前后端。")
                self._append_runtime_log("GPU 后台加载失败，继续使用当前可用后端。", "#ff8c00")
                current = self._det_ref[0]
                if not isinstance(current, UnavailableDetector):
                    color = "#107c10" if "CPU" not in str(getattr(current, "device_name", "")).upper() else "#d13438"
                    self._set_runtime_indicator(getattr(current, "backend_name", "当前后端"), getattr(current, "device_name", ""), color)
                return
            self._set_runtime_indicator("CUDA 不可用", "", "#d13438")
            if not show_dialog and not allow_cpu:
                message = "GPU/TensorRT 后台加载失败，自动切换到 CPU 回退后台加载。"
                self.status.showMessage(message)
                self._append_runtime_log(message, "#ff8c00")
                self._start_model_load_worker(path, allow_cpu_fallback=True, show_dialog=False)
                return
            if not show_dialog:
                message = "CPU 回退加载失败，当前没有可用推理后端。"
                self._det_ref[0] = UnavailableDetector(message)
                self.status.showMessage(message)
                self._append_runtime_log(message, "#d13438")
                return
            if self._prompt_cpu_fallback(error):
                self._start_model_load_worker(path, allow_cpu_fallback=True, show_dialog=show_dialog)
            else:
                message = "用户拒绝 CPU 回退，当前不能继续检测。"
                self._det_ref[0] = UnavailableDetector(message)
                self.status.showMessage(message)
                self._append_runtime_log(message, "#d13438")
                self._set_runtime_indicator("未加载", "", "#d13438")
            return

        message = f"模型加载失败: {error}"
        if show_dialog:
            QMessageBox.critical(self, "模型加载失败", str(error))
        if not kept_detector:
            self._det_ref[0] = UnavailableDetector(str(error))
            self._set_runtime_indicator("加载失败", "", "#d13438")
        else:
            current = self._det_ref[0]
            if not isinstance(current, UnavailableDetector):
                color = "#107c10" if "CPU" not in str(getattr(current, "device_name", "")).upper() else "#d13438"
                self._set_runtime_indicator(getattr(current, "backend_name", "当前后端"), getattr(current, "device_name", ""), color)
        self.status.showMessage("模型后台加载失败" if kept_detector else "模型加载失败")
        self._append_runtime_log(message, "#d13438")

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_F11:
            if self.isFullScreen():
                self.showNormal()
                self.titlebar.btn_restore.setText("❐")
            else:
                self.showFullScreen()
                self.titlebar.btn_restore.setText("⊡")
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.tab_cam.stop()
        self.tab_vid.stop_video(release=True, summarize=False)
        w = self.tab_img._batch_worker
        if w and w.isRunning():
            self.tab_img._suppress_batch_abort_dialog = True
            w.abort()
            w.wait(2000)
        detector = self._det_ref[0] if self._det_ref else None
        if detector and hasattr(detector, "close"):
            try:
                detector.close()
            except Exception:
                pass
        super().closeEvent(event)


class AnimatedSplash(QWidget):
    """带动画进度条的启动画面"""

    def __init__(self, logo_path=None, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.SplashScreen)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(480, 320)

        # 主容器
        container = QFrame(self)
        container.setObjectName("splashContainer")
        container.setGeometry(0, 0, 480, 320)
        container.setStyleSheet("""
            QFrame#splashContainer {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ffffff, stop:1 #f0f2f5);
                border: 1px solid #d7dde5;
                border-radius: 16px;
            }
        """)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(12)

        # Logo
        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignCenter)
        if logo_path and os.path.exists(logo_path):
            pix = QPixmap(logo_path)
            scaled = pix.scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.logo_label.setPixmap(scaled)
        else:
            self.logo_label.setText("LOGO")
            self.logo_label.setStyleSheet("font-size: 36px; font-weight: bold; color: #0078d4;")
        layout.addWidget(self.logo_label)

        # 应用名称
        self.title_label = QLabel("划痕缺陷检测系统")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #1a1a1a; font-family: 'Microsoft YaHei UI';"
        )
        layout.addWidget(self.title_label)

        layout.addSpacing(8)

        # 状态文字
        self.status_label = QLabel("正在初始化...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(
            "font-size: 13px; color: #5a5a5a; font-family: 'Microsoft YaHei UI';"
        )
        layout.addWidget(self.status_label)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background: #e8e8e8;
                border: none;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0078d4, stop:1 #60cdff);
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)

        layout.addSpacing(4)

        # 版本/版权
        self.version_label = QLabel("v6.1")
        self.version_label.setAlignment(Qt.AlignCenter)
        self.version_label.setStyleSheet(
            "font-size: 11px; color: #999999; font-family: 'Microsoft YaHei UI';"
        )
        layout.addWidget(self.version_label)

        # 动画
        self._target_value = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._animate_progress)
        self._timer.start(16)  # ~60fps

    def _animate_progress(self):
        current = self.progress_bar.value()
        if current < self._target_value:
            step = max(1, (self._target_value - current) // 8)
            self.progress_bar.setValue(min(current + step, self._target_value))

    def set_progress(self, value, text=""):
        self._target_value = value
        if text:
            self.status_label.setText(text)

    def center_on_screen(self):
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)


def main():
    multiprocessing.freeze_support()
    logging.basicConfig(
        level=getattr(logging, os.getenv("QT5YOLO_LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setStyleSheet(LIGHT_STYLE)

    logo = find_existing_path(
        str(BUNDLE_ROOT / "data" / "source_image" / "logo.png"),
        str(BUNDLE_ROOT / "data" / "source_image" / "logo.ico"),
        str(BUNDLE_ROOT / "logo.ico"),
        str(APP_ROOT / "logo.ico"),
    )
    splash = AnimatedSplash(logo)
    splash.center_on_screen()
    splash.show()
    QApplication.processEvents()

    splash.set_progress(20, "系统引导初始化...")
    QApplication.processEvents()
    time.sleep(0.15)

    splash.set_progress(45, "视觉模型装载...")
    QApplication.processEvents()
    time.sleep(0.15)

    splash.set_progress(70, "UI视图构建...")
    QApplication.processEvents()
    time.sleep(0.15)

    splash.set_progress(90, "系统就绪...")
    QApplication.processEvents()

    win = MainWindow()
    splash.set_progress(100, "启动完成")
    start_fullscreen = os.getenv("QT_START_FULLSCREEN", "0").strip().lower() not in {"0", "false", "no"}
    if start_fullscreen:
        win.showFullScreen()
    else:
        if os.getenv("QT_START_NORMAL", "0").strip().lower() in {"1", "true", "yes"}:
            win.show()
        else:
            win.showMaximized()
    splash.close()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
