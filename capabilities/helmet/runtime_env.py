"""推理运行时工具：设备解析 + 精度参数适配。

为什么要单独抽出来：ultralytics 8.4 起 `half=` 被 `quantize=` 取代，
而 8.3 还没有 `quantize`；同时 `device=None` 在部分版本里会退化成 CPU 推理
（实测 yolov8n 从 67ms/帧 掉到 ~10ms/帧 的差距就出在这里）。

工单：CV-安防智能体V1.0-03 / CV专高5-11
"""
from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger("utils.runtime_env")


def has_cuda() -> bool:
    """当前环境是否有可用的 CUDA 设备。"""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def resolve_device(device: str | None) -> str:
    """把 ''/None/'auto' 解析成具体设备字符串。

    Returns
    -------
    "0"（有 GPU 时）或 "cpu"
    """
    dev = str(device or "").strip().lower()
    if dev in ("", "none", "auto"):
        return "0" if has_cuda() else "cpu"
    if dev.startswith("cuda"):
        return "0" if dev in ("cuda", "cuda:0") else dev.split(":", 1)[-1]
    if dev.isdigit():
        return dev
    return dev  # "cpu" / "mps" 等


def precision_kwargs(half: bool) -> Dict[str, Any]:
    """返回当前 ultralytics 版本支持的精度参数。

    8.4+ 用 quantize=16；8.3 及以前用 half=True。
    """
    if not half:
        return {}
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT  # type: ignore

        if "quantize" in DEFAULT_CFG_DICT:
            return {"quantize": 16}
    except Exception:  # noqa: BLE001
        pass
    return {"half": True}


def cudnn_tune() -> None:
    """固定输入尺寸下让 cudnn 自动选最快算法，并允许 TF32。"""
    try:
        import torch

        if not torch.cuda.is_available():
            return
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:  # noqa: BLE001
        pass


def gpu_name() -> str:
    """当前 GPU 名称，无 GPU 返回 'CPU'。"""
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    except Exception:  # noqa: BLE001
        return "CPU"
