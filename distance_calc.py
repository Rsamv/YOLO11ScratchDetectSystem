#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
距离计算脚本 - 用于计算缺陷实际大小和缺陷间距离
支持划痕(2mm阈值)和螺栓(3mm阈值)的判断

该模块不直接参与模型推理，而是在 YOLO 输出检测框后，
为 ui.py 中的缺陷量化、标准判定和聚集分析提供基础计算函数。
"""

import math


def pixels_to_mm(pixels, pixel_per_cm=100.0):
    """像素长度转换为毫米长度。

    pixel_per_cm 表示每厘米对应的像素数量，因此 1 像素对应
    10 / pixel_per_cm 毫米。
    """
    return pixels * 10.0 / pixel_per_cm


def calculate_distance(point1, point2, pixel_per_cm=100.0):
    """
    计算两点之间的距离

    Args:
        point1: (x1, y1) 第一个点坐标(像素)
        point2: (x2, y2) 第二个点坐标(像素)
        pixel_per_cm: 像素当量(像素/厘米)

    Returns:
        距离(毫米)
    """
    x1, y1 = point1
    x2, y2 = point2

    # 先在图像坐标系中计算两点欧氏距离，再统一换算成毫米。
    dist_px = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    dist_mm = pixels_to_mm(dist_px, pixel_per_cm)
    return dist_mm


def calculate_defect_size(bbox, pixel_per_cm=100.0):
    """
    计算缺陷等效直径

    Args:
        bbox: (x1, y1, x2, y2) 边界框坐标(像素)
        pixel_per_cm: 像素当量(像素/厘米)

    Returns:
        dict: 包含宽度、高度、等效直径(毫米)的字典
    """
    x1, y1, x2, y2 = bbox

    # YOLO 输出的是矩形检测框，这里分别计算框的像素宽度和高度。
    w_px = abs(x2 - x1)
    h_px = abs(y2 - y1)
    w_mm = pixels_to_mm(w_px, pixel_per_cm)
    h_mm = pixels_to_mm(h_px, pixel_per_cm)

    # 本项目将较长边作为缺陷等效直径，便于和毫米级阈值进行比较。
    equiv_diameter = max(w_mm, h_mm)
    return {
        'width_mm': w_mm,
        'height_mm': h_mm,
        'equiv_diameter_mm': equiv_diameter,
        'center_x': (x1 + x2) / 2,
        'center_y': (y1 + y2) / 2
    }


def is_lethal_scratch(diameter_mm, lethal_threshold=2.0):
    """判断划痕是否为致命缺陷(>2mm)"""
    return diameter_mm > lethal_threshold


def is_lethal_aluminum_alloy(diameter_mm, lethal_threshold=0.8):
    """判断铝合金零件缺陷是否为致命缺陷(>0.8mm)"""
    return diameter_mm > lethal_threshold


def evaluate_defect_type(diameter_mm, detect_object="scratch"):
    """
    根据检测对象判断缺陷是否超标

    Args:
        diameter_mm: 等效直径(毫米)
        detect_object: 检测对象类型 "scratch" 或 "aluminum_alloy"

    Returns:
        tuple: (是否超标, 是否致命, 判定说明)
    """
    # 不同检测对象采用不同的尺寸阈值，返回值供界面层生成 OK/NG 结果。
    if detect_object == "scratch":
        # 划痕: >5mm为致命, >2mm为超标
        if diameter_mm > 5.0:
            return diameter_mm > 2.0, True, f"致命划痕({diameter_mm:.2f}mm)"
        return diameter_mm > 2.0, False, f"普通划痕({diameter_mm:.2f}mm)"
    else:
        # 铝合金零件: >2.0mm为致命, >1.0mm为超标
        if diameter_mm > 2.0:
            return diameter_mm > 1.0, True, f"致命铝合金缺陷({diameter_mm:.2f}mm)"
        return diameter_mm > 1.0, False, f"普通铝合金缺陷({diameter_mm:.2f}mm)"


def check_clustering(defect_details, pixel_per_cm=100.0, cluster_threshold_mm=10.0):
    """
    检测缺陷聚集情况

    Args:
        defect_details: 缺陷详情列表,每个包含center坐标
        pixel_per_cm: 像素当量
        cluster_threshold_mm: 聚集判定阈值(mm)

    Returns:
        list: 聚集簇列表
    """
    clusters = []
    used = set()

    # 使用简单的贪心分组：以未访问缺陷为起点，查找阈值距离内的相邻缺陷。
    for i, d1 in enumerate(defect_details):
        if i in used:
            continue
        cluster = [i]
        used.add(i)

        for j, d2 in enumerate(defect_details):
            if j in used:
                continue
            dist = calculate_distance(d1['center'], d2['center'], pixel_per_cm)

            # 两个缺陷中心距离小于阈值时，认为它们属于同一聚集区域。
            if dist < cluster_threshold_mm:
                cluster.append(j)
                used.add(j)

        # 只有包含两个及以上缺陷的分组才记录为聚集簇。
        if len(cluster) > 1:
            clusters.append(cluster)

    return clusters


def main():
    print("=" * 50)
    print("缺陷距离计算工具")
    print("=" * 50)

    # 示例用法
    # 假设像素当量为100像素/厘米
    pixel_per_cm = 100.0

    # 示例: 两个缺陷中心点坐标(像素)
    defect1_center = (100, 100)
    defect2_center = (150, 120)

    dist_mm = calculate_distance(defect1_center, defect2_center, pixel_per_cm)
    print(f"\n缺陷1中心: {defect1_center}")
    print(f"缺陷2中心: {defect2_center}")
    print(f"缺陷间距离: {dist_mm:.2f} mm")

    # 示例: 缺陷边界框(像素)
    bbox1 = (50, 50, 150, 150)  # x1, y1, x2, y2
    size_info = calculate_defect_size(bbox1, pixel_per_cm)
    print(f"\n缺陷边界框: {bbox1}")
    print(f"等效直径: {size_info['equiv_diameter_mm']:.2f} mm")

    # 划痕判定
    print(f"\n划痕判定: {evaluate_defect_type(size_info['equiv_diameter_mm'], 'scratch')[2]}")
    # 铝合金零件判定
    print(f"铝合金判定: {evaluate_defect_type(size_info['equiv_diameter_mm'], 'aluminum_alloy')[2]}")


if __name__ == "__main__":
    main()
