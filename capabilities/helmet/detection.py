"""检测 Agent：调用 YOLOv8（.pt 或 .onnx）对帧推理。

性能相关的几个开关（V1.0 加速优化）：
- `device`：显式解析为 "0"/"cpu"，避免 `device=None` 在某些版本退化成 CPU 推理；
- `imgsz`：推理输入边长，默认 640；降到 480 可再快一截；
- 精度：GPU 上默认 FP16（8.4 用 quantize=16，8.3 用 half=True，自动适配）；
- `model.fuse()`：Conv+BN 融合；
- `cudnn.benchmark` + TF32：固定输入尺寸下自动选最快卷积；
- 预热两次，首帧开销不计入 FPS。

"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from .runtime_env import cudnn_tune, precision_kwargs, resolve_device

logger = logging.getLogger("agents.detection")

class DetectionAgent:
    """加载 ultralytics YOLOv8 模型，对 BGR ndarray 推理。

    Parameters
    ----------
    weights : str
        .pt 或 .onnx 路径。
    conf : float
        置信度阈值。
    device : str
        "cuda:0" / "0" / "cpu"；"" 表示自动（有 GPU 就用 GPU）。
    imgsz : int
        推理输入边长，默认 640。
    half : bool | None
        FP16 推理。None 表示 GPU 自动开、CPU 自动关。
    max_det : int
        单帧最多保留多少个框。
    """

    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.45,
                 device: str = "", imgsz: int = 640, half: bool | None = None,
                 max_det: int = 100):
        # 延迟 import，避免在没装 ultralytics 的环境里直接崩
        from ultralytics import YOLO

        self.weights = str(weights)
        self.conf = conf
        self.device = resolve_device(device)
        self.imgsz = int(imgsz)
        self.max_det = max_det

        if half is None:
            # 本机实测 FP16 反而更慢（ultralytics 8.4 的 quantize 路径有额外开销：
            # 640 分辨率下 fp32 约 20ms/帧，fp16 约 32ms/帧），因此默认关闭。
            # 换卡后可用 --half 或 DetectionAgent(half=True) 自行验证。
            half = False
        self.half = bool(half) and not self.weights.lower().endswith(".onnx")
        self._precision = precision_kwargs(self.half)

        cudnn_tune()
        logger.info("加载 YOLOv8 模型: %s (conf=%.2f, imgsz=%d, precision=%s, device=%s)",
                    weights, conf, self.imgsz,
                    "fp16" if self.half else "fp32", self.device)
        self.model = YOLO(weights)
        try:
            self.model.fuse()          # Conv+BN 融合
        except Exception as exc:  # pragma: no cover
            logger.debug("fuse 跳过: %s", exc)
        self._warmup()

    # ------------------------------------------------------------------ #
    def _warmup(self) -> None:
        """跑两次空帧，把 kernel 编译/显存分配的开销移出正式计时。"""
        try:
            import numpy as np

            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            for _ in range(2):
                self.model.predict(dummy, conf=self.conf, imgsz=self.imgsz,
                                   verbose=False, device=self.device,
                                   max_det=self.max_det, **self._precision)
        except Exception as exc:  # pragma: no cover
            logger.warning("warmup 跳过: %s", exc)

    # ------------------------------------------------------------------ #
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        frame = state.get("frame")
        if frame is None or state.get("eof"):
            return {"detections": []}

        results = self.model.predict(
            frame, conf=self.conf, imgsz=self.imgsz, verbose=False,
            device=self.device, max_det=self.max_det, **self._precision,
        )
        dets: List[Dict[str, Any]] = []
        if results:
            r = results[0]
            names = r.names or {}
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                dets.append({
                    "class": names.get(int(box.cls[0]), str(int(box.cls[0]))),
                    "bbox": [round(v, 1) for v in (x1, y1, x2, y2)],
                    "score": round(float(box.conf[0]), 3),
                })
        logger.debug("frame_id=%d 检出 %d 个目标", state.get("frame_id"), len(dets))
        return {"detections": dets}
