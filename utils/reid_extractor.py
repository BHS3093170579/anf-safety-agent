# -*- coding: utf-8 -*-
"""OSNet 行人重识别（Re-ID）特征提取封装。

设计要点
--------
1. **懒加载**：进程启动时只有本类的一个空壳，OSNet 的权重不会被读进显存。
   直到 `extract()` 被第一次调用（即规则引擎确认 abnormal 的第一帧），
   才真正 build 模型 + load 权重 + 搬到 GPU。这样主流水线日常运行时
   OSNet 处于"静默休眠"状态，不占显存也不耗算力。

2. **调用计数**：`infer_calls` 记录网络真正被调用的次数，供验收用例
   验证"正常帧从未调用 OSNet、突变违规帧只调用 1 次"。

3. **特征维度**：OSNet 在全连接层前输出 512 维向量，这里直接取
   eval 模式的前向结果（torchreid 的 OSNet.forward 在 eval 下返回 feat）。

依赖
----
torchreid 的 PyPI 包（0.2.5）与 torch 2.x 不兼容，这里改用官方仓库源码
裁剪版，放在 `third_party/torchreid/`，仅保留 OSNet 模型定义。
权重从 HuggingFace 镜像下载的 MSMT17 预训练权重（8.9 MB）。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = PROJECT_ROOT / "third_party"
DEFAULT_WEIGHT = PROJECT_ROOT / "weights" / "reid" / "osnet_x0_25_msmt17.pth"

# OSNet 官方输入尺寸：宽 128 × 高 256（与 MSMT17/Market1501 评测一致）
IN_W, IN_H = 128, 256
# ImageNet 归一化参数（torchreid 的默认预处理）
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class ReidExtractor:
    """OSNet 特征提取器（懒加载，512 维输出）。"""

    def __init__(
        self,
        weight_path: str | Path | None = None,
        model_name: str = "osnet_x0_25",
        device: str | None = None,
        half: bool = False,
    ) -> None:
        self.weight_path = Path(weight_path) if weight_path else DEFAULT_WEIGHT
        self.model_name = model_name
        self._device_pref = device
        self.half = half

        self._model = None          # 懒加载：第一次 extract 才有值
        self._device = None
        self._torch = None

        self.load_calls = 0         # 模型被加载的次数（正常应恒为 1）
        self.infer_calls = 0        # 网络被真正前向的次数
        self.total_crops = 0        # 累计处理的人体裁剪数
        self.last_ms = 0.0          # 最近一次前向耗时

    # ------------------------------------------------------------------ #
    # 懒加载
    # ------------------------------------------------------------------ #
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self) -> None:
        """第一次用到时才加载模型（"静默休眠 → 唤醒")。"""
        if self._model is not None:
            return

        # torchreid 裁剪版放在 third_party，避免污染 site-packages
        if str(THIRD_PARTY) not in sys.path:
            sys.path.insert(0, str(THIRD_PARTY))

        import torch  # noqa: PLC0415  —— 故意延迟导入，保证进程启动不用 torch
        from torchreid import models  # noqa: PLC0415

        if not self.weight_path.exists():
            raise FileNotFoundError(
                f"OSNet 权重不存在：{self.weight_path}\n"
                "请从 https://hf-mirror.com/kaiyangzhou/osnet 下载 "
                "osnet_x0_25_msmt17*.pth 放到 weights/reid/ 下"
            )

        state = torch.load(str(self.weight_path), map_location="cpu", weights_only=False)
        # checkpoint 的 classifier 对应 MSMT17 的 4101 个 ID，必须按原类别数构建
        # 才能 strict 加载；推理走 eval 模式的 feat 分支，不经过 classifier。
        n_cls = int(state["classifier.weight"].shape[0])

        model = models.osnet_x0_25(num_classes=n_cls, pretrained=False)
        model.load_state_dict(state, strict=True)
        model.eval()

        device = self._device_pref or ("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        if self.half and device == "cuda":
            model = model.half()

        self._model = model
        self._device = device
        self._torch = torch
        self.load_calls += 1

    # ------------------------------------------------------------------ #
    # 预处理
    # ------------------------------------------------------------------ #
    @staticmethod
    def _preprocess(crops: Sequence[np.ndarray]) -> np.ndarray:
        """BGR uint8 裁剪图 → NCHW float32 归一化张量（N,3,256,128）。"""
        import cv2  # noqa: PLC0415

        batch = np.empty((len(crops), 3, IN_H, IN_W), dtype=np.float32)
        for i, img in enumerate(crops):
            if img is None or img.size == 0:
                batch[i] = 0.0
                continue
            im = cv2.resize(img, (IN_W, IN_H), interpolation=cv2.INTER_LINEAR)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            im = (im - MEAN) / STD
            batch[i] = im.transpose(2, 0, 1)
        return batch

    # ------------------------------------------------------------------ #
    # 推理
    # ------------------------------------------------------------------ #
    def extract(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """提取特征。crops 为 BGR 裁剪图列表，返回 (N, 512) float32。"""
        if not crops:
            return np.zeros((0, 512), dtype=np.float32)

        self._ensure_loaded()
        torch = self._torch
        blob = self._preprocess(crops)

        x = torch.from_numpy(blob).to(self._device)
        if self.half and self._device == "cuda":
            x = x.half()

        t0 = time.perf_counter()
        with torch.no_grad():
            feat = self._model(x)
        if self._device == "cuda":
            torch.cuda.synchronize()
        self.last_ms = (time.perf_counter() - t0) * 1000.0

        out = feat.detach().float().cpu().numpy().astype(np.float32)
        self.infer_calls += 1
        self.total_crops += len(crops)
        return out

    # ------------------------------------------------------------------ #
    # 便捷接口
    # ------------------------------------------------------------------ #
    def embed_boxes(
        self,
        frame_bgr: np.ndarray,
        bboxes: Sequence[Sequence[float]],
        pad_ratio: float = 0.05,
    ) -> np.ndarray:
        """按 bbox（x1,y1,x2,y2 像素）从原图裁剪后提取特征，返回 (N,512)。"""
        h, w = frame_bgr.shape[:2]
        crops: list[np.ndarray] = []
        for bb in bboxes:
            x1, y1, x2, y2 = [float(v) for v in bb[:4]]
            bw, bh = x2 - x1, y2 - y1
            # 稍微外扩，避免把肩膀和腿裁掉（Re-ID 很吃整体轮廓）
            x1 -= bw * pad_ratio
            x2 += bw * pad_ratio
            y1 -= bh * pad_ratio
            y2 += bh * pad_ratio
            xi1 = max(0, min(w - 1, int(round(x1))))
            yi1 = max(0, min(h - 1, int(round(y1))))
            xi2 = max(xi1 + 1, min(w, int(round(x2))))
            yi2 = max(yi1 + 1, min(h, int(round(y2))))
            crops.append(frame_bgr[yi1:yi2, xi1:xi2])
        return self.extract(crops)

    def stats(self) -> dict:
        return {
            "model": self.model_name,
            "loaded": self.is_loaded,
            "device": self._device,
            "load_calls": self.load_calls,
            "infer_calls": self.infer_calls,
            "total_crops": self.total_crops,
            "last_ms": round(self.last_ms, 2),
        }

    def reset_stats(self) -> None:
        self.infer_calls = 0
        self.total_crops = 0
        self.last_ms = 0.0
