"""边缘端人体检测+追踪服务（ModelPool 共享 + 关键点重构框 + ByteTrack）。

这是把三块能力串起来的入口：
    ModelPool（唯一一份 pose 权重）
        ↓ 每帧推理一次
    keypoint_bbox（17 关键点 → 人体外接框，不另加载 detect 模型）
        ↓
    PoseByteTracker（每路一个实例，给稳定 track_id）

为什么每路要单独一个 tracker：
    ByteTrack 内部用卡尔曼滤波维护每条轨迹的位置和速度，状态是跨帧累积的。
    多路共用会把 A 相机的人关联到 B 相机的轨迹上，ID 直接错乱。
    但**模型**必须共用——这正是「显存只留一份、状态各管各的」。

性能取舍：
    推理是串行排队的（见 ModelPool 注释）。8 路 × 5FPS = 40 帧/秒，
    单帧 10ms 左右，串行利用率约 40%，完全够用；
    并发推理虽然看着快，但显存会随线程数翻倍，违背设计前提。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tracking.bytetrack_wrapper import PoseByteTracker
from tracking.keypoint_bbox import aspect_ok, bbox_from_result
from utils.model_pool import ModelPool

PROJECT_ROOT = Path(__file__).resolve().parent.parent

@dataclass
class CamResult:
    """一路视频一帧的处理结果。"""

    cam_id: str
    frame_id: int
    ts: float
    tracks: list[dict] = field(default_factory=list)
    infer_ms: float = 0.0
    pose_count: int = 0

class ModelService:
    """对外只暴露 process()，内部共享模型、隔离跟踪状态。"""

    def __init__(
        self,
        conf: float = 0.35,
        kpt_conf: float = 0.25,
        imgsz: int = 640,
        weight: str | Path | None = None,
        lying_ok: bool = False,
    ) -> None:
        self.pool = ModelPool.get(weight=weight)
        self.conf = float(conf)
        self.kpt_conf = float(kpt_conf)
        self.imgsz = int(imgsz)
        # 睡岗场景专用：人趴在桌上时关键点只有头和肩，算出来的框宽大于高，
        # 会被「站立人形」的宽高比筛掉。打开这个开关就跳过该筛。
        self.lying_ok = bool(lying_ok)

        self._trackers: dict[str, PoseByteTracker] = {}
        self._frames: dict[str, int] = {}
        self._lock = threading.Lock()

    def _tracker(self, cam_id: str) -> PoseByteTracker:
        with self._lock:
            trk = self._trackers.get(cam_id)
            if trk is None:
                trk = PoseByteTracker(cam_id=cam_id)
                self._trackers[cam_id] = trk
                self._frames[cam_id] = 0
            return trk

    def reset(self, cam_id: str | None = None) -> None:
        """换视频/换会话时调用，别让上一段视频的 ID 残留在新视频里。"""
        with self._lock:
            if cam_id is None:
                for trk in self._trackers.values():
                    trk.reset()
                self._frames = {k: 0 for k in self._frames}
            elif cam_id in self._trackers:
                self._trackers[cam_id].reset()
                self._frames[cam_id] = 0

    def process(self, cam_id: str, frame: np.ndarray) -> CamResult:
        """单帧处理。线程安全。"""
        h, w = frame.shape[:2]
        trk = self._tracker(cam_id)

        t0 = time.perf_counter()
        res = self.pool.predict(frame, conf=self.conf, **({} if self.imgsz <= 0 else {"imgsz": self.imgsz}))
        infer_ms = (time.perf_counter() - t0) * 1000

        kd = None
        if res and getattr(res[0], "keypoints", None) is not None:
            kd = res[0].keypoints.data

        items = [
            it for it in bbox_from_result(kd, (h, w), conf_thr=self.kpt_conf)
            if (self.lying_ok or aspect_ok(it[0]))
        ]
        tracks = trk.update(items, (h, w))

        with self._lock:
            self._frames[cam_id] = self._frames.get(cam_id, 0) + 1
            fid = self._frames[cam_id]

        return CamResult(
            cam_id=cam_id,
            frame_id=fid,
            ts=time.time(),
            tracks=tracks,
            infer_ms=infer_ms,
            pose_count=len(items),
        )

    # ---------- 多路循环 ----------
    def run(
        self,
        sources: dict[str, str | int],
        fps: float = 5.0,
        max_frames: int | None = None,
        on_frame: Callable[[CamResult, np.ndarray], None] | None = None,
        stop_event: threading.Event | None = None,
    ) -> dict[str, dict]:
        """顺序轮询多路（单进程单线程，最简单也最省显存）。

        Args:
            sources: {cam_id: 视频路径 或 摄像头索引}
            fps: 每路的抽帧目标帧率
            max_frames: 每路最多处理多少帧
            on_frame: 回调，拿结果和原帧去画图/推流
            stop_event: 置位即退出

        Returns:
            {cam_id: 统计}
        """
        import cv2

        caps: dict[str, Any] = {}
        steps: dict[str, int] = {}
        for cid, src in sources.items():
            cap = cv2.VideoCapture(str(src))
            if not cap.isOpened():
                print(f"[ModelService] {cid} 打不开：{src}")
                continue
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            caps[cid] = cap
            steps[cid] = max(1, int(round(src_fps / fps)))

        counters = {cid: 0 for cid in caps}
        src_idx = {cid: 0 for cid in caps}
        stats: dict[str, dict] = {
            cid: {"frames": 0, "infer_ms": [], "ids": set(), "fps": 0.0} for cid in caps
        }

        stop_event = stop_event or threading.Event()
        t_start = time.perf_counter()

        try:
            while caps and not stop_event.is_set():
                alive = False
                for cid in list(caps.keys()):
                    if stop_event.is_set():
                        break
                    cap = caps[cid]
                    ok, frame = cap.read()
                    if not ok:
                        caps.pop(cid, None)
                        cap.release()
                        continue
                    src_idx[cid] += 1
                    if src_idx[cid] % steps[cid]:
                        continue
                    if max_frames is not None and counters[cid] >= max_frames:
                        caps.pop(cid, None)
                        cap.release()
                        continue
                    counters[cid] += 1
                    alive = True

                    r = self.process(cid, frame)
                    st = stats[cid]
                    st["frames"] += 1
                    st["infer_ms"].append(r.infer_ms)
                    st["ids"].update(t["track_id"] for t in r.tracks)
                    if on_frame:
                        on_frame(r, frame)
                # caps 空了 while 自然退出；这里只需防「还在读但没到抽帧点」的空转
                if not alive and not caps:
                    break
        finally:
            for cap in caps.values():
                try:
                    cap.release()
                except Exception:
                    pass

        total_s = max(1e-6, time.perf_counter() - t_start)
        for cid, st in stats.items():
            ms = st["infer_ms"]
            st["avg_ms"] = round(float(np.mean(ms)), 2) if ms else 0.0
            st["ids"] = sorted(st["ids"])
            st["fps"] = round(st["frames"] / total_s, 2)
        return stats

    def stats(self) -> dict:
        return {
            "pool": self.pool.stats(),
            "trackers": {c: t.stats() for c, t in self._trackers.items()},
            "frames": dict(self._frames),
        }

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    svc = ModelService()
    video = PROJECT_ROOT / "datasets" / "videos" / "helmet" / "111111.mp4"
    print("处理：", video)
    st = svc.run({"Camera_01": str(video)}, fps=5, max_frames=20)
    print("统计：", st)
    print("显存：", svc.pool.vram_json())
