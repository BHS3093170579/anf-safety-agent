"""安全帽三路检测器：把独立项目的三套算法接进 he 框架。

三路对应 `--pipeline`：
    single     端到端 YOLOv8，直接输出 helmet / no_helmet
    two_stage  先检人头，再对每个头框做「戴没戴」分类
    pose_rule  端到端 + 姿态兜底：人出现了但头上没有任何 helmet 框，补一条 no_helmet

对外只暴露 he 的 `DetectionAgent` 接口（detect / reset_stabilizer / stats），
所以前端、规则引擎、告警链路都不用改；唯一的差异是底层 `_infer`。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from agents.detector import DetectionAgent

from .detection import DetectionAgent as QcDetectionAgent
from .helmet_rule import HelmetFallbackChecker
from .two_stage import TwoStageDetector

logger = logging.getLogger("capabilities.helmet")

#: he 框架里的类别 id（与 config/scenes.yaml 的 helmet 段一致）
HE_CLASSES = {"helmet": (0, "hardhat"), "no_helmet": (1, "no-hardhat")}

#: 三路对应的界面名称
PIPELINE_LABELS = {
    "single": "端到端（最快）",
    "two_stage": "两步法（小目标更准）",
    "pose_rule": "端到端 + 姿态兜底",
}

WEIGHT_DIR = Path(__file__).resolve().parents[2] / "weights" / "helmet_qc"


class HelmetCascadeDetector(DetectionAgent):
    """安全帽三路检测器。

    继承 he 的 DetectionAgent 只为了复用后处理：按类别卡阈值、时间稳定器、
    最小显示分数、框尺寸合理性。推理部分由 `_infer` 换成三路之一。
    """

    def __init__(self, scene_cfg: dict, pipeline: str = "single",
                 device: str = "0", warmup: bool = True) -> None:
        super().__init__(scene_cfg, warmup=False, load_model=False)
        if pipeline not in PIPELINE_LABELS:
            raise ValueError(f"未知的安全帽检测方式：{pipeline}")

        self.pipeline = pipeline
        self.device = device
        imgsz = int(scene_cfg["imgsz"])
        conf = float(scene_cfg["conf"])          # 已被 get_detector 压到 DETECT_FLOOR

        weights = {
            "single": WEIGHT_DIR / "best.pt",
            "head": WEIGHT_DIR / "head_det.pt",
            "cls": WEIGHT_DIR / "head_cls.pt",
            "pose": WEIGHT_DIR / "yolov8n-pose.pt",
        }
        needed = ["pose"] if pipeline == "two_stage" else ["single"]
        if pipeline == "two_stage":
            needed = ["head", "cls"]
        elif pipeline == "pose_rule":
            needed = ["single", "pose"]
        for key in needed:
            if not weights[key].exists():
                raise FileNotFoundError(f"安全帽权重缺失：{weights[key]}")

        if pipeline == "two_stage":
            self._base = TwoStageDetector(
                head_weights=str(weights["head"]),
                cls_weights=str(weights["cls"]),
                conf=conf,
                cls_thr=0.50,
                device=device,
                imgsz=imgsz,
                tile=False,
            )
        else:
            self._base = QcDetectionAgent(
                weights=str(weights["single"]),
                conf=conf,
                device=device,
                imgsz=imgsz,
            )

        self._checker = None
        if pipeline == "pose_rule":
            self._checker = HelmetFallbackChecker(
                pose_weights=str(weights["pose"]),
                conf=0.25,
                device=device,
            )

        logger.info("安全帽三路已加载：%s（imgsz=%d, conf=%.2f）",
                    pipeline, imgsz, conf)

    def _infer(self, frame: np.ndarray) -> list[dict[str, Any]]:
        dets = list(self._base({"frame": frame}).get("detections") or [])
        if self._checker is not None:
            # 兜底规则必须先看到 helmet 框，否则会把已戴帽的人误判成违规
            dets += self._checker.check(frame, dets)

        out: list[dict[str, Any]] = []
        for d in dets:
            mapped = HE_CLASSES.get(str(d.get("class", "")))
            if mapped is None:
                continue
            cid, name = mapped
            bbox = d.get("bbox") or []
            if len(bbox) != 4:
                continue
            out.append({
                "class": name,
                "class_id": cid,
                "bbox": [round(float(v), 2) for v in bbox],
                "score": round(float(d.get("score", 0.0)), 4),
            })
        return out
