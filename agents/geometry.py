# -*- coding: utf-8 -*-
"""
几何工具 —— 多个 Agent 共用的边界框计算

之前 detector / judgment / stabilizer 各自写了一份 IOU，实现几乎一样但签名不同。
合并到这里，避免改一处忘一处（比如 IOS 只在 detector 里有，另外两处没有）。
"""
from __future__ import annotations

def bbox_iou(a: list[float], b: list[float]) -> float:
    """两个 [x1,y1,x2,y2] 的 IOU（交集 / 并集）。

    两个框大小接近时有效；小框被大框包住时 IOU 会很低，
    这种情况应该用 bbox_ios。
    """
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = x2 - x1
    ih = y2 - y1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih

    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def bbox_ios(a: list[float], b: list[float]) -> float:
    """IOS（交集 / 较小框面积）。

    小框被大框完全包住时 IOU 只有 0.05，但 IOS 接近 1。
    判断「同一目标被标了两个类别」必须用这个。
    """
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = x2 - x1
    ih = y2 - y1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih

    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0

def bbox_area(b: list[float]) -> float:
    """框面积。"""
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

def bbox_center(b: list[float]) -> tuple[float, float]:
    """框中心点。"""
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
