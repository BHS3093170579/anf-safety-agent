# -*- coding: utf-8 -*-
"""
安防智能体 · Agent 包（工单 02 · CV专高5-10）

模块划分：
    video_capture  VideoCaptureAgent  读视频、按帧率抽帧
    detector       DetectionAgent     目标检测（.pt / .onnx / .engine 通用）
    judgment       JudgmentAgent      规则判断（违规类别 + 连续帧确认）
    alert          AlertAgent         告警推送 + 证据截图
    reporter       ReportAgent        汇总报告

设计原则：
    每个 Agent 职责单一、可独立替换。
    换检测模型不用动判断逻辑，换告警通道不用动检测逻辑。
"""

from .alert import AlertAgent
from .detector import DetectionAgent
from .judgment import JudgmentAgent
from .reporter import ReportAgent
from .video_capture import FramePacket, VideoCaptureAgent

__all__ = [
    "AlertAgent",
    "DetectionAgent",
    "JudgmentAgent",
    "ReportAgent",
    "VideoCaptureAgent",
    "FramePacket",
]
