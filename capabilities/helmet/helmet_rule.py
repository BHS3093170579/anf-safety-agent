"""未戴安全帽兜底规则：先找"人"，再看这个人头上有没有安全帽。

为什么需要
----------
安全帽模型是在 construction-ppe 上训练的，`no_helmet` 正样本只有约 400 个框，
对"裸头"的召回很低：画面里明明站着一个人、头上什么也没戴，模型却一个框都不出。
这个模块用姿态模型（yolov8n-pose）拿到每个人的头部关键点，构造人头框；
只要这个人头框上**没有任何 helmet 检测框覆盖**，就补一条 `no_helmet`。
这样"人出现了但没戴帽子"就不会被漏掉；已戴安全帽的人因为头上压着 helmet 框，
不会被重复报成违规。

工单：CV-安防智能体V1.0-03 / CV专高5-11
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .runtime_env import resolve_device

logger = logging.getLogger("agents.helmet_rule")

#: pose 关键点下标
KP_NOSE = 0
KP_EYES = (1, 2)
KP_EARS = (3, 4)
HEAD_KPS = (KP_NOSE,) + KP_EYES + KP_EARS

#: 已戴安全帽 / 未戴安全帽的类别名（与 judgment 保持一致，避免循环导入这里各写一份）
HELMET_ON = {"helmet", "hardhat", "hard_hat", "safety_helmet", "helmet_on"}
HELMET_OFF = {"no_helmet", "no_hat"}


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)
    area_b = max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)
    return inter / (area_a + area_b - inter)


def _center_inside(box: Sequence[float], region: Sequence[float]) -> bool:
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]


def _center_dist_norm(a: Sequence[float], b: Sequence[float]) -> float:
    """中心距离 / 平均框尺寸，用来识别"同一个头但框大小不同"的重复框。"""
    acx, acy = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bcx, bcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    size = max((a[2] - a[0] + b[2] - b[0]) / 2.0,
               (a[3] - a[1] + b[3] - b[1]) / 2.0, 1.0)
    return float(np.hypot(acx - bcx, acy - bcy)) / size


def head_box_from_keypoints(kpts: np.ndarray, vis_thr: float = 0.30) -> Optional[List[float]]:
    """由头部关键点（鼻/眼/耳）构造人头框；关键点太少时返回 None。"""
    if kpts is None:
        return None
    pts = [(float(kpts[i][0]), float(kpts[i][1])) for i in HEAD_KPS
           if float(kpts[i][2]) >= vis_thr]
    if not pts:
        return None

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    if span < 10.0:          # 人头太小，姿态不可信，别乱报
        return None

    # 以鼻子为中心（鼻子不可见时用可见点均值），按脸宽外扩出头框
    if float(kpts[KP_NOSE][2]) >= vis_thr:
        cx, cy = float(kpts[KP_NOSE][0]), float(kpts[KP_NOSE][1])
    else:
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)

    half_w = span * 1.25
    half_up = span * 1.15
    half_dn = span * 1.25
    return [cx - half_w, cy - half_up, cx + half_w, cy + half_dn]


class HelmetFallbackChecker:
    """用姿态模型找人头，补齐"有人但没戴安全帽"的漏检。

    Parameters
    ----------
    pose_weights : str
        yolov8n-pose 权重路径。
    conf : float
        pose 置信度阈值（人）。
    device : str
        推理设备，"" 表示自动。
    """

    def __init__(self, pose_weights: str, conf: float = 0.25, device: str = ""):
        from pathlib import Path

        from ultralytics import YOLO

        root = Path(__file__).resolve().parents[2]
        path = Path(str(pose_weights))
        if not path.is_absolute():
            path = root / path
        self.pose_weights = str(path)
        self.conf = float(conf)
        self.device = resolve_device(device)
        logger.info("加载兜底规则姿态模型: %s (conf=%.2f, device=%s)",
                    self.pose_weights, self.conf, self.device)
        self._model = YOLO(self.pose_weights)
        try:
            self._model.fuse()
        except Exception:  # pragma: no cover
            pass

    # ------------------------------------------------------------------ #
    def _persons(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """返回 [{bbox, score, kpts}]（同一帧的重复人框只留分数高的）。"""
        r = self._model.predict(frame, conf=self.conf, verbose=False,
                                device=self.device)[0]
        out: List[Dict[str, Any]] = []
        if r.boxes is None:
            return out
        kdata = r.keypoints.data if r.keypoints is not None else None
        for i, b in enumerate(r.boxes):
            kpts = None
            if kdata is not None and i < len(kdata):
                k = kdata[i]
                kpts = k.cpu().numpy() if hasattr(k, "cpu") else np.asarray(k)
            box = [float(v) for v in b.xyxy[0].tolist()]
            score = float(b.conf[0]) if b.conf is not None else 0.0
            # pose 偶尔会把同一个人检成两个框（一大一小），先做一次人去重
            if any(_iou(box, o["bbox"]) > 0.55 for o in out):
                continue
            out.append({"bbox": box, "score": score, "kpts": kpts})
        return out

    def check(self, frame: np.ndarray,
              detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """返回需要补的 `no_helmet` 检测框（不修改入参）。"""
        if frame is None:
            return []

        helmets = [d for d in detections if str(d.get("class", "")) in HELMET_ON]
        reported = [d for d in detections if str(d.get("class", "")) in HELMET_OFF]

        extra: List[Dict[str, Any]] = []
        for p in self._persons(frame):
            head = head_box_from_keypoints(p["kpts"])
            if head is None:
                continue
            hb = p["bbox"]
            if hb[3] - hb[1] < 30:       # 远景小人，跳过，避免噪声
                continue

            # 1) 头上压着安全帽框 -> 已戴，跳过
            if any(_iou(head, h["bbox"]) >= 0.10 or _center_inside(h["bbox"], head)
                   for h in helmets):
                continue
            # 2) 模型已经报过这个头 -> 不重复报（框大小不同但中心重合也算同一个头）
            if any(_iou(head, n["bbox"]) >= 0.20
                   or _center_inside(n["bbox"], head)
                   or _center_dist_norm(head, n["bbox"]) <= 0.55
                   for n in reported):
                continue
            # 3) 本帧规则自己算出的重复头框 -> 只留一个
            if any(_iou(head, e["bbox"]) >= 0.40
                   or _center_dist_norm(head, e["bbox"]) <= 0.55
                   for e in extra):
                continue

            extra.append({
                "class": "no_helmet",
                "bbox": [round(v, 1) for v in head],
                "score": round(min(0.95, p["score"] * 0.9), 3),
                "source": "head_rule",
            })
        return extra
