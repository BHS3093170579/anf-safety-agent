# -*- coding: utf-8 -*-
"""
视频播放器模块 —— 让检测结果像播放器一样流畅播放、暂停、拖动

对应工单：CV专高5-11（工单 03 - 任务 7：Streamlit 界面）

为什么需要这个模块：
    Streamlit 的 st.image() 每次调用都要走一遍 WebSocket + 浏览器重绘，
    逐帧刷新根本跟不上视频帧率，看起来就是「一卡一卡的」。

    解决办法是把「处理」和「播放」分开：
        1. 先用检测流程把所有帧处理完，生成带检测框的视频文件（H.264）
        2. 再用 st.video() 播放这个文件，浏览器原生解码，流畅且能暂停/拖动

    代价是必须等处理完才能播，好处是播放体验和普通播放器一致。

H.264 编码来源：
    OpenCV 自带的 FFmpeg 缺少 libx264 / libopenh264，写出的 mp4v 浏览器不支持。
    因此用 imageio-ffmpeg 自带的静态 ffmpeg.exe 做管道编码。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np


def _get_ffmpeg() -> str | None:
    """拿到可用的 ffmpeg 可执行文件路径，优先用 imageio-ffmpeg 自带的。"""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if Path(exe).exists():
            return exe
    except Exception:
        pass
    import shutil
    return shutil.which("ffmpeg")


def h264_available() -> bool:
    """浏览器可播放的编码是否可用。"""
    return _get_ffmpeg() is not None


class VideoWriter:
    """把帧序列写成浏览器可播放的 H.264 MP4。

    用法：
        writer = VideoWriter("out.mp4", width=1280, height=720, fps=30)
        writer.write(bgr_frame)
        ...
        writer.close()
    """

    def __init__(self, out_path: str | Path, width: int, height: int,
                 fps: float = 30.0, crf: int = 26) -> None:
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.crf = int(crf)
        self._proc: subprocess.Popen | None = None
        self._fallback: cv2.VideoWriter | None = None
        self.frames_written = 0

        # H.264 要求宽高为偶数
        if self.width % 2:
            self.width -= 1
        if self.height % 2:
            self.height -= 1

        ffmpeg = _get_ffmpeg()
        if ffmpeg:
            cmd = [
                ffmpeg, "-y", "-loglevel", "error",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{self.width}x{self.height}",
                "-pix_fmt", "bgr24", "-r", str(self.fps),
                "-i", "-",
                "-an",
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",        # 浏览器兼容必须
                "-preset", "veryfast",
                "-crf", str(self.crf),
                "-movflags", "+faststart",    # 边下边播
                str(self.out_path),
            ]
            try:
                self._proc = subprocess.Popen(
                    cmd, stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                )
            except Exception:
                self._proc = None

        if self._proc is None:
            # 兜底：用 OpenCV 的 mp4v（浏览器可能播不了，但至少文件是完整的）
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._fallback = cv2.VideoWriter(
                str(self.out_path), fourcc, self.fps, (self.width, self.height))

    def write(self, frame: np.ndarray) -> None:
        if frame is None:
            return
        h, w = frame.shape[:2]
        if (w, h) != (self.width, self.height):
            frame = cv2.resize(frame, (self.width, self.height))
        if self._proc is not None and self._proc.stdin is not None:
            try:
                self._proc.stdin.write(frame.tobytes())
                self.frames_written += 1
            except (BrokenPipeError, OSError):
                self._proc = None
        elif self._fallback is not None:
            self._fallback.write(frame)
            self.frames_written += 1

    def close(self) -> Path:
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                self._proc.wait(timeout=60)
            except Exception:
                pass
            self._proc = None
        if self._fallback is not None:
            self._fallback.release()
            self._fallback = None
        return self.out_path


def is_playable(path: str | Path) -> bool:
    """粗查文件是否是有效视频（能被 OpenCV 打开且有帧）。"""
    p = Path(path)
    if not p.exists() or p.stat().st_size < 1024:
        return False
    cap = cv2.VideoCapture(str(p))
    ok = cap.isOpened()
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if ok else 0
    cap.release()
    return ok and n > 0


def burn_frame_into_video(src_video: Path, detections_by_frame: dict[int, list[dict]],
                          draw_fn, out_path: Path, max_frames: int = 0) -> Path:
    """把检测结果画到原视频上，输出新视频。

    这张备用路径用于「只想画出检测框、不重新跑模型」的场景。
    """
    cap = cv2.VideoCapture(str(src_video))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 2
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    writer = VideoWriter(out_path, w, h, fps)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets = detections_by_frame.get(idx, [])
        writer.write(draw_fn(frame, dets) if dets else frame)
        idx += 1
        if max_frames and idx >= max_frames:
            break
    cap.release()
    return writer.close()
