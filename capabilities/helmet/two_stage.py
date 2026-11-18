"""两步法检测器：先检测人头，再对每个头框做"戴没戴"分类。

为什么要这么拆：
- 端到端直接检出 no_helmet，等价于在 40~80 px 的小目标上同时学定位和分类，
  数据只有 400 个 no_helmet 框时 mAP 上不去；
- 拆开后：
  * 检测器只学"人头"（helmet + no_helmet 的框都是头，合并成 1 类，约 1954 个框）；
  * 分类器只学"这颗头戴没戴"（头框裁剪图二分类，样本量翻倍且目标居中放大）。
  指标也从"小目标 mAP"变成"头检测 mAP × 头分类准确率"，更容易达标。

对外接口与 `DetectionAgent` 完全一致，直接 `__call__(state) -> {"detections": [...]}`，
所以 LangGraph / Web / Streamlit 三套入口都能无痛替换。

工单：CV-安防智能体V1.0-03 / CV专高5-11
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from .headnet import CLASS_NAMES, load_classifier, preprocess
from .runtime_env import cudnn_tune, resolve_device
from .iou import _iou

logger = logging.getLogger("agents.two_stage")

_ROOT = Path(__file__).resolve().parents[2]


class TwoStageDetector:
    """人头检测 + 头框分类，输出与 DetectionAgent 相同结构的 detections。

    Parameters
    ----------
    head_weights : str
        1 类人头检测权重（.pt）。
    cls_weights : str
        头框分类器权重（.pt，由 tools/train_cls_cnn.py 产出）。
    conf : float
        人头检测置信度阈值。
    cls_thr : float
        判为 no_helmet 的概率阈值。
    margin : float
        裁剪时头框外扩比例（与训练时保持一致）。
    """

    def __init__(self, head_weights: str = "runs/detect/head_det/weights/best.pt",
                 cls_weights: str = "models/head_cls.pt",
                 conf: float = 0.35, cls_thr: float = 0.5,
                 device: str = "", imgsz: int = 640,
                 margin: float = 0.18, max_det: int = 50,
                 tile: bool = False, tile_grid: int = 2, tile_overlap: float = 0.2):
        import torch
        from ultralytics import YOLO

        self.conf = conf
        self.cls_thr = cls_thr
        self.margin = margin
        self.imgsz = int(imgsz)
        self.max_det = max_det
        self.device = resolve_device(device)
        self.tile = bool(tile)          # 切片推理：治"远处小人头漏检"
        self.tile_grid = max(1, int(tile_grid))
        self.tile_overlap = float(tile_overlap)
        cudnn_tune()

        hw = Path(head_weights)
        cw = Path(cls_weights)
        self.head_weights = str(hw if hw.is_absolute() else (_ROOT / hw))
        self.cls_weights = str(cw if cw.is_absolute() else (_ROOT / cw))
        for p in (self.head_weights, self.cls_weights):
            if not Path(p).exists():
                raise FileNotFoundError(f"权重不存在：{p}")

        logger.info("两步法：加载人头检测器 %s", self.head_weights)
        self.detector = YOLO(self.head_weights)
        try:
            self.detector.fuse()
        except Exception:  # noqa: BLE001
            pass

        self.torch_device = torch.device(
            f"cuda:{self.device}" if self.device != "cpu" else "cpu")
        logger.info("两步法：加载头框分类器 %s (device=%s)",
                    self.cls_weights, self.torch_device)
        self.classifier = load_classifier(self.cls_weights, self.torch_device)
        self._warmup()

    # ------------------------------------------------------------------ #
    def _warmup(self) -> None:
        try:
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self.detector.predict(dummy, conf=self.conf, imgsz=self.imgsz,
                                  verbose=False, device=self.device,
                                  max_det=self.max_det)
            import torch

            with torch.no_grad():
                self.classifier(preprocess(dummy).unsqueeze(0).to(self.torch_device))
        except Exception as exc:  # pragma: no cover
            logger.warning("两步法 warmup 跳过: %s", exc)

    # ------------------------------------------------------------------ #
    def _crop(self, frame: np.ndarray, box: List[float]) -> np.ndarray | None:
        """按外扩比例裁剪头框。"""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        mx, my = bw * self.margin, bh * self.margin
        cx1 = int(max(0, x1 - mx))
        cy1 = int(max(0, y1 - my))
        cx2 = int(min(w, x2 + mx))
        cy2 = int(min(h, y2 + my))
        if cx2 - cx1 < 8 or cy2 - cy1 < 8:
            return None
        return frame[cy1:cy2, cx1:cx2]

    # ------------------------------------------------------------------ #
    def _nms(self, boxes: List[List[float]], scores: List[float],
             iou_thr: float = 0.5) -> List[int]:
        """按分数贪心做 NMS，返回保留下标。"""
        order = sorted(range(len(boxes)), key=lambda i: -scores[i])
        keep: List[int] = []
        for i in order:
            ok = True
            for j in keep:
                if _iou(boxes[i], boxes[j]) > iou_thr:
                    ok = False
                    break
            if ok:
                keep.append(i)
        return keep

    def _detect_heads(self, frame: np.ndarray):
        """人头检测；开启 tile 时对切片分别推理再合并，提升小人头召回。"""
        h, w = frame.shape[:2]
        boxes: List[List[float]] = []
        scores: List[float] = []

        def run(img, ox=0, oy=0):
            r = self.detector.predict(img, conf=self.conf, imgsz=self.imgsz,
                                      verbose=False, device=self.device,
                                      max_det=self.max_det)
            if not r:
                return
            res = r[0]
            if res.boxes is None:
                return
            for b in res.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                boxes.append([x1 + ox, y1 + oy, x2 + ox, y2 + oy])
                scores.append(float(b.conf[0]))

        if not self.tile:
            run(frame)
        else:
            n = self.tile_grid
            tw, th = w / n, h / n
            ox_ = tw * self.tile_overlap
            oy_ = th * self.tile_overlap
            for gy in range(n):
                for gx in range(n):
                    x1 = int(max(0, gx * tw - ox_))
                    y1 = int(max(0, gy * th - oy_))
                    x2 = int(min(w, (gx + 1) * tw + ox_))
                    y2 = int(min(h, (gy + 1) * th + oy_))
                    if x2 - x1 < 32 or y2 - y1 < 32:
                        continue
                    run(frame[y1:y2, x1:x2], x1, y1)
            # 切片结果可能与全图重复，统一 NMS
            keep = self._nms(boxes, scores, 0.5)
            boxes = [boxes[i] for i in keep]
            scores = [scores[i] for i in keep]
        return boxes, scores

    # ------------------------------------------------------------------ #
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        frame = state.get("frame")
        if frame is None or state.get("eof"):
            return {"detections": []}

        boxes, scores = self._detect_heads(frame)
        dets: List[Dict[str, Any]] = []
        if not boxes:
            return {"detections": dets}

        crops, keep_idx = [], []
        for i, box in enumerate(boxes):
            crop = self._crop(frame, box)
            if crop is None:
                continue
            crops.append(crop)
            keep_idx.append(i)

        if crops:
            import torch

            batch = torch.stack([preprocess(c) for c in crops]).to(self.torch_device)
            with torch.no_grad():
                prob = torch.softmax(self.classifier(batch), dim=1).cpu().numpy()
            for j, i in enumerate(keep_idx):
                p_no = float(prob[j, CLASS_NAMES.index("no_helmet")])
                label = "no_helmet" if p_no >= self.cls_thr else "helmet"
                dets.append({
                    "class": label,
                    "bbox": [round(v, 1) for v in boxes[i]],
                    "score": round(float(max(p_no, 1 - p_no)) * scores[i], 3),
                    "det_conf": round(scores[i], 3),
                    "cls_prob": round(p_no, 3),
                })
        logger.debug("frame_id=%d 两步法：头 %d 个，输出 %d 条",
                     state.get("frame_id"), len(crops), len(dets))
        return {"detections": dets}
