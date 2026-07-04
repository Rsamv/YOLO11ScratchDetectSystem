# 辅助工具函数
import os
import cv2
import time
import numpy as np
from pathlib import Path
from PyQt5.QtGui import QPixmap, QImage, QFont, QFontMetrics
from PyQt5.QtWidgets import QLabel

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

def fit_label_font(lbl: QLabel, min_size=10):
    base_style = lbl.property("_fit_base_style")
    if base_style is None:
        base_style = lbl.styleSheet()
        lbl.setProperty("_fit_base_style", base_style)
    for size in range(36, min_size - 1, -1):
        f = QFont(lbl.font())
        f.setPixelSize(size)
        metrics = QFontMetrics(f)
        if metrics.horizontalAdvance(lbl.text()) <= max(1, lbl.width() - 8):
            lbl.setStyleSheet(f"{base_style}; font-size:{size}px;")
            return
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
