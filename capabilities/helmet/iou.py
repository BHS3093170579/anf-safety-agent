"""IoU 工具。

原项目里 two_stage.py 从 behavior.py 借 `_iou`；behavior.py 还带着睡岗/徘徊/打架/
明火烟雾一套逻辑，抽安全帽模块时没必要整包带过来，所以把这个纯几何函数单独落地，
计算方式与原 `behavior._iou` 完全一致（two_stage.py 的 NMS 结果因此不变）。
"""
from __future__ import annotations

from typing import List

def _iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0

iou = _iou
