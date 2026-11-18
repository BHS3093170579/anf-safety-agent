"""ByteTrack 跟踪器封装（关键点重构框 → 稳定 track_id → 骨架回填）。

工单编号：CV项目-安防智能体-03-边缘端—ModelPool+显存控制与人体 BBox 重构追踪

核心矛盾（对应工单第 3 节「动态追踪 ID 分配」）：
    Pose 模型逐帧独立推理，同一个人在第 1 帧和第 2 帧没有任何联系。
    要让 track_id 稳定，必须给它配上运动模型 + 外观匹配。

为什么用 ByteTrack：
    工地场景遮挡极多（钢管、脚手架、其他人）。DeepSORT/BoT-SORT 依赖
    ReID 特征，Nano 上跑 ReID 网络显存和算力都吃紧；ByteTrack 只用
    运动 + 低分框二次关联，零额外显存，正好符合工单的「极限瘦身」约束。

关键实现点（全部是踩过的坑）：
1.  **每路视频必须各自持有一个 tracker 实例**
    tracker 内部维护 `tracked/pool/lost` 三个列表和帧号。若多路共用，
    Camera_02 的人会被关联到 Camera_01 的轨迹上，ID 直接错乱。
2.  **没有检出时也要调 update**
    传空 Boxes 进去，tracker 才会执行卡尔曼预测、把丢了的人标记为 lost。
    如果直接 return，短时遮挡后就再也接不回来了。
3.  **用第 8 列 `idx` 回填关键点**
    tracker 输出 `(M, 8)`：`[x1,y1,x2,y2,track_id,score,cls,idx]`，
    idx 指回你喂进去的那批框的下标，这样骨架不会串到别人身上。
4.  **必须调 `BaseTrack.reset_id()` 吗**
    跨相机不需要，但每次新建测试会话建议重置，否则 ID 会一直往上加。
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_bt_config(cfg_path: str | Path | None = None):
    """加载 ByteTrack 参数。

    优先用项目内自己的配置文件（便于改阈值），没有就退回 ultralytics 自带的。
    """
    from types import SimpleNamespace

    import yaml

    if cfg_path and Path(cfg_path).exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return SimpleNamespace(**data)

    try:
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace, yaml_load
        from ultralytics.utils.checks import check_yaml

        return IterableSimpleNamespace(**yaml_load(check_yaml("bytetrack.yaml")))
    except Exception:
        # 兜底：用 ultralytics bytetrack.yaml 的默认值
        return SimpleNamespace(
            tracker_type="bytetrack",
            track_high_thresh=0.25,
            track_low_thresh=0.1,
            new_track_thresh=0.25,
            track_buffer=30,
            match_thresh=0.8,
            fuse_score=True,
        )


class PoseByteTracker:
    """一路视频对应一个实例。**不要跨路复用。**

    用法：
        trk = PoseByteTracker()
        tracks = trk.update(items, frame.shape[:2])
        # items: [(bbox, quality, kpts_17x2), ...]  ← keypoint_bbox.bbox_from_result 的输出
        # tracks: [{"track_id": 5, "bbox": [...], "score": 0.8, "keypoints": ndarray(17,2)}, ...]
    """

    def __init__(self, cfg_path: str | Path | None = None, cam_id: str = "") -> None:
        self.cam_id = cam_id
        self._cfg = _load_bt_config(cfg_path)
        self._lock = threading.Lock()
        self._tracker = self._new_tracker()

        self.total_frames = 0
        self.total_tracks = 0
        self._seen_ids: set[int] = set()
        # 追踪器内部异常只提示一次，避免每帧刷屏；但必须留痕，
        # 否则 ultralytics 版本不兼容时会静默退化成「没有 track_id」
        self._err_logged = False

    def _note_update_error(self, exc: Exception) -> None:
        if self._err_logged:
            return
        self._err_logged = True
        print(f"[PoseByteTracker{('/' + self.cam_id) if self.cam_id else ''}] "
              f"update() 抛异常，本帧及后续失败帧将返回空结果："
              f"{type(exc).__name__}: {exc}", flush=True)

    def _new_tracker(self):
        from ultralytics.trackers.byte_tracker import BYTETracker

        # 注意：ultralytics 8.4 的 BYTETracker 签名只有 (self, args)，
        # 没有 frame_rate 参数（老版本 DeepSORT 风格才有）。
        # tracker 在内部自己累加 frame_id，跨帧状态靠 persist 维护。
        return BYTETracker(self._cfg)

    def reset(self) -> None:
        """新会话开始时清空轨迹，避免旧 ID 串到新视频。"""
        with self._lock:
            from ultralytics.trackers.basetrack import BaseTrack

            BaseTrack.reset_id()
            self._tracker = self._new_tracker()
            self.total_frames = 0
            self.total_tracks = 0
            self._seen_ids.clear()

    def update(
        self,
        items: list[tuple[list[float], float, np.ndarray]],
        img_shape: tuple[int, int],
    ) -> list[dict]:
        """喂入本帧所有重构框，返回带 track_id 的结果。

        Args:
            items: [(bbox_xyxy, quality, kpts_17x2), ...]
            img_shape: (h, w)

        Returns:
            每项含 track_id / bbox / score / keypoints / is_new
        """
        with self._lock:
            self.total_frames += 1

            if not items:
                # 关键：空帧也要喂，让卡尔曼预测推进、把丢目标标记 lost
                try:
                    self._tracker.update(self._empty_boxes(img_shape))
                except Exception as exc:      # noqa: BLE001
                    self._note_update_error(exc)
                return []

            from ultralytics.engine.results import Boxes

            # (N, 6) = x1,y1,x2,y2,conf,cls
            data = np.zeros((len(items), 6), dtype=np.float32)
            for i, (bbox, quality, _k) in enumerate(items):
                data[i, 0:4] = bbox
                data[i, 4] = max(float(quality), 0.05)
                data[i, 5] = 0.0        # 统一当作「人」一类

            boxes = Boxes(data, img_shape)
            try:
                tracked = self._tracker.update(boxes)
            except Exception as exc:          # noqa: BLE001
                self._note_update_error(exc)
                return []

            if tracked is None or len(tracked) == 0:
                return []

            results: list[dict] = []
            for row in tracked:
                row = np.asarray(row, dtype=np.float64).reshape(-1)
                if row.shape[0] < 8:
                    continue
                tid = int(row[4])
                score = float(row[5])
                idx = int(row[7]) if row.shape[0] > 7 else -1
                kpts = items[idx][2] if 0 <= idx < len(items) else None

                is_new = tid not in self._seen_ids
                if is_new:
                    self._seen_ids.add(tid)
                    self.total_tracks += 1

                results.append({
                    "track_id": tid,
                    "bbox": [float(row[0]), float(row[1]), float(row[2]), float(row[3])],
                    "score": score,
                    "keypoints": kpts,
                    "is_new": is_new,
                    "cam_id": self.cam_id,
                })

            return results

    @staticmethod
    def _empty_boxes(img_shape: tuple[int, int]):
        from ultralytics.engine.results import Boxes

        return Boxes(np.zeros((0, 6), dtype=np.float32), img_shape)

    def stats(self) -> dict:
        return {
            "cam_id": self.cam_id,
            "frames": self.total_frames,
            "unique_ids": len(self._seen_ids),
            "assignments": self.total_tracks,
        }
