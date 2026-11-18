"""头框分类网络定义与预处理（训练与推理共用，保证 state_dict 能跨进程加载）。

"""
from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn

CLASS_NAMES = ["helmet", "no_helmet"]
IMG_SIZE = 128
MEAN = 0.45
STD = 0.25

class HeadNet(nn.Module):
    """4 层卷积 + 全局池化的小分类网络（约 0.35M 参数）。"""

    def __init__(self, num_classes: int = 2):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32), block(32, 64), block(64, 128), block(128, 128),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))

def preprocess(img_bgr: np.ndarray, size: int = IMG_SIZE) -> torch.Tensor:
    """BGR 裁剪图 -> 归一化张量 (3,size,size)。"""
    img = cv2.resize(img_bgr, (size, size))
    x = img[:, :, ::-1].astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))

def load_classifier(path: str, device: torch.device) -> "HeadNet":
    """加载训练好的分类器。"""
    ckpt: Dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    net = HeadNet(len(ckpt.get("class_names", CLASS_NAMES)))
    net.load_state_dict(ckpt["state_dict"])
    net.to(device).eval()
    return net
