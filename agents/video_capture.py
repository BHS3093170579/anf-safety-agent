# -*- coding: utf-8 -*-
"""
VideoCaptureAgent ——

职责：
    读取视频源并抽帧，支持两种输入：
      1. 视频文件（MP4 / AVI / MOV）—— 按帧号跳帧采样
      2. 实时摄像头（USB / 内置 / RTSP）—— 按时间间隔采样

设计要点：
    1. 视频文件按帧号跳（cap.set(POS_FRAMES)），比逐帧读再丢弃快得多
    2. 摄像头不能跳帧，只能连续读、按时间间隔决定是否采样
    3. 摄像头是无限流，必须设置 max_frames 或由外部中断
    4. RTSP 网络摄像头按字符串 URL 处理，走摄像头分支

用法：
    # 视频文件
    va = VideoCaptureAgent("datasets/videos/helmet/demo.mp4", sample_fps=1)

    # 本机摄像头
    va = VideoCaptureAgent(0, sample_fps=2, max_frames=100)

    # RTSP 网络摄像头
    va = VideoCaptureAgent("rtsp://user:pass@192.168.1.10:554/stream", sample_fps=2)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

@dataclass
class FramePacket:
    """一帧的完整信息。"""
    index: int              # 采样后的帧序号（第几个采样帧）
    frame: np.ndarray       # BGR 图像
    timestamp: float        # 相对开始的时间（秒）
    source_frame_id: int    # 原始帧号 / 摄像头累计帧数

class VideoCaptureAgent:
    """视频采集 Agent：支持视频文件与实时摄像头。"""

    def __init__(self, source: str | Path | int, sample_fps: float = 1.0,
                 max_frames: int = 0, camera_width: int = 0,
                 camera_height: int = 0) -> None:
        """
        Args:
            source:        视频文件路径 / 摄像头索引（0 表示默认摄像头）/ RTSP 地址
            sample_fps:    每秒抽多少帧
            max_frames:    最多处理多少采样帧，0 表示不限制（摄像头建议必设）
            camera_width:  摄像头期望宽度（0 用默认）
            camera_height: 摄像头期望高度（0 用默认）
        """
        # sample_fps = 0 表示「逐帧处理、不抽帧」，用于需要全帧率分析的场景
        self.full_frame_mode = (sample_fps is not None and float(sample_fps) <= 0)
        self.sample_fps = 0.0 if self.full_frame_mode else (
            float(sample_fps) if sample_fps > 0 else 1.0)
        self.max_frames = int(max_frames)
        self._release_requested = False

        # 判断输入类型：int 或 rtsp/http 开头 → 摄像头
        self.is_camera = isinstance(source, int) or (
            isinstance(source, str) and source.lower().startswith(("rtsp://", "http://"))
        )

        if self.is_camera:
            self.camera_source = source
            self.video_path = f"camera:{source}"
            self._init_camera(int(source) if isinstance(source, int) else 0,
                              camera_width, camera_height,
                              apply_resolution=not isinstance(source, str))
        else:
            self.video_path = str(source)
            if not Path(self.video_path).exists():
                raise FileNotFoundError(f"视频文件不存在：{self.video_path}")
            self._init_file()

    # ---------- 初始化 ----------
    def _init_file(self) -> None:
        """视频文件：探测元信息。"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频（编码不支持或文件损坏）：{self.video_path}")
        self.source_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if self.source_fps <= 0:
            self.source_fps = 25.0
        self.duration = self.total_frames / self.source_fps if self.source_fps else 0.0
        # 逐帧模式下 step=1，每帧都处理
        self.step = 1 if self.full_frame_mode else max(
            1, int(round(self.source_fps / self.sample_fps)))

    def _init_camera(self, index: int, want_w: int, want_h: int,
                     apply_resolution: bool = True) -> None:
        """摄像头：探测可用性并读取实际分辨率。"""
        # Windows 上 DSHOW 后端比默认后端更稳定
        backend = cv2.CAP_DSHOW if isinstance(self.camera_source, int) else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.camera_source, backend)
        if not cap.isOpened():
            # 退一步换默认后端再试
            cap = cv2.VideoCapture(self.camera_source)
        if not cap.isOpened():
            raise IOError(
                f"无法打开摄像头 {self.camera_source}\n"
                f"可能原因：设备被其他程序占用、没有摄像头、或索引不对"
            )

        if apply_resolution and (want_w or want_h):
            if want_w:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, want_w)
            if want_h:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, want_h)

        # 缓冲区设为 1：始终读到最新帧，避免处理积压的旧帧造成画面延迟
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        # 尝试要求 30fps，部分摄像头默认 15fps
        try:
            cap.set(cv2.CAP_PROP_FPS, 30)
        except Exception:
            pass

        self.source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if self.source_fps <= 0 or self.source_fps > 240:
            self.source_fps = 30.0
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.total_frames = 0          # 摄像头无限流
        self.duration = 0.0
        self.step = 1                  # 摄像头不能跳帧，靠时间间隔采样
        cap.release()

    # ---------- 元信息 ----------
    @property
    def expected_samples(self) -> int:
        """预计采样帧数。摄像头是无限流，返回 max_frames（未设则 0）。"""
        if self.is_camera:
            return self.max_frames
        n = int(self.total_frames / self.step) if self.total_frames else 0
        # 逐帧模式下 step=1，n 就是全部帧数，这里无需额外处理
        if self.max_frames > 0:
            n = min(n, self.max_frames)
        return n

    def info(self) -> dict:
        """视频源元信息。"""
        return {
            "path": self.video_path,
            "source_type": "camera" if self.is_camera else "file",
            "source_fps": round(self.source_fps, 2),
            "total_frames": self.total_frames,
            "duration_sec": round(self.duration, 2),
            "resolution": f"{self.width}x{self.height}",
            "sample_fps": self.sample_fps,
            "full_frame_mode": self.full_frame_mode,
            "frame_step": self.step,
            "expected_samples": self.expected_samples,
            "max_frames": self.max_frames,
        }

    # ---------- 迭代 ----------
    def iter_frames(self) -> Iterator[FramePacket]:
        """逐帧产出采样帧。"""
        return self._iter_camera() if self.is_camera else self._iter_file()

    def _iter_file(self) -> Iterator[FramePacket]:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频：{self.video_path}")

        emitted = 0
        src_id = 0
        try:
            while True:
                if src_id > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, src_id)
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                yield FramePacket(
                    index=emitted, frame=frame,
                    timestamp=round(src_id / self.source_fps, 3),
                    source_frame_id=src_id,
                )
                emitted += 1
                if self.max_frames > 0 and emitted >= self.max_frames:
                    break
                src_id += self.step
                if self.total_frames and src_id >= self.total_frames:
                    break
        finally:
            cap.release()

    def _iter_camera(self) -> Iterator[FramePacket]:
        """摄像头：连续读帧，按时间间隔决定是否采样。

        与文件读取的关键区别：
            - 不能 set(POS_FRAMES) 跳帧，只能一直读
            - 每读一帧检查是否到了采样时刻，没到就丢弃继续读
            - 实时性优先：读帧后立刻丢弃积压，保证拿到的是当前画面
        """
        backend = cv2.CAP_DSHOW if isinstance(self.camera_source, int) else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.camera_source, backend)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_source)
        if not cap.isOpened():
            raise IOError(f"无法打开摄像头：{self.camera_source}")

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # 逐帧模式：间隔为 0，读到就处理
        interval = 0.0 if self.full_frame_mode else 1.0 / self.sample_fps
        t0 = time.time()
        last_emit = -interval
        emitted = 0
        src_id = 0
        try:
            while True:
                # 关键：先把缓冲里的旧帧全部读掉，只保留最新一帧。
                # 否则处理速度跟不上时，画面会越来越滞后，表现为「卡顿」。
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.05)
                    if time.time() - t0 > 5 and emitted == 0:
                        break
                    continue
                src_id += 1

                now = time.time() - t0
                if now - last_emit < interval:
                    continue          # 还没到采样时刻，丢弃这一帧继续读

                yield FramePacket(
                    index=emitted, frame=frame,
                    timestamp=round(now, 3),
                    source_frame_id=src_id,
                )
                emitted += 1
                last_emit = now
                if self.max_frames > 0 and emitted >= self.max_frames:
                    break
        finally:
            cap.release()

    def read_all(self) -> list[FramePacket]:
        """一次性读取全部采样帧（仅建议用于短文件，摄像头慎用）。"""
        return list(self.iter_frames())
