"""空间距离 ID 恢复器（TrackRecovery）。

原始判据：
    · 维护 LostTracks：3 秒内消失目标的 [track_id, last_bbox, last_timestamp, state]
    · 当画面出现「生命周期为 1 帧」的全新追踪目标时，
      算它与所有丢失目标中心点的欧氏距离
    · 距离 < 50px 且 Δt < 3s → 把新 ID 强行改写为旧 ID，并恢复时序状态

为什么判据要同时卡距离和时间：
    只卡距离：人在原地不动，换个 ID 也会被误合并（其实是对的，但静止物体
              反复进出检测会不断合并，反而放大误报）
    只卡时间：3 秒内可以走很远，会把路人甲认成工友乙
    两者同时卡：3 秒 × 正常步速 1.2m/s ≈ 3.6m，50px 对应画面里的一小段，
              这个组合正好覆盖「被柱子挡了一下」的典型场景。

额外加的两道防护（没写但必须有，否则会乱认）：
    1. 尺寸相似度：框面积比要在 0.4~2.5 之间。
       否则「远处的路人小框」会被合并到「近处工友的大框」上。
    2. 一对多取最近：多个丢失目标都符合时，只取距离最小的那个，
       且一个旧 ID 只能被认领一次（认领后就从候选里移除）。
"""

from __future__ import annotations

import math
import time

from utils.camera_memory_store import CameraMemoryStore, TrackState

def center(bbox: list[float] | tuple[float, ...]) -> tuple[float, float]:
    """框中心点。"""
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter)

def area(bbox: list[float]) -> float:
    return max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

class TrackRecovery:
    """把「刚出现 1 帧的新目标」匹配回「3 秒内消失的旧目标」。"""

    def __init__(
        self,
        store: CameraMemoryStore,
        max_distance: float = 50.0,
        max_dt: float = 3.0,
        area_ratio_range: tuple[float, float] = (0.4, 2.5),
        new_track_max_age_frames: int = 1,
    ) -> None:
        self.store = store
        self.max_distance = float(max_distance)
        self.max_dt = float(max_dt)
        self.area_ratio_range = area_ratio_range
        self.new_track_max_age_frames = int(new_track_max_age_frames)

        # track_id → 已经连续出现多少帧。lifecycle=1 就是所谓「全新目标」
        self._lifecycle: dict[tuple[str, int], int] = {}
        self.stats = {
            "checked": 0,
            "recovered": 0,
            "rejected_far": 0,
            "rejected_age": 0,
            "rejected_size": 0,
        }

    def _bump_lifecycle(self, cam_id: str, track_id: int) -> int:
        key = (cam_id, track_id)
        self._lifecycle[key] = self._lifecycle.get(key, 0) + 1
        return self._lifecycle[key]

    def reset(self, cam_id: str | None = None) -> None:
        if cam_id is None:
            self._lifecycle.clear()
        else:
            for k in [k for k in self._lifecycle if k[0] == cam_id]:
                self._lifecycle.pop(k, None)

    def try_recover(
        self,
        cam_id: str,
        tracks: list[dict],
        now: float | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """处理一帧的追踪结果。

        Args:
            cam_id: 摄像头 id
            tracks: ModelService 产出的 [{track_id, bbox, score, keypoints, ...}]
            now: 时间戳（测试可注入）

        Returns:
            (改写后的 tracks, 恢复事件列表)
            恢复事件形如 {"new_id": 31, "old_id": 7, "dist": 18.2, "dt": 2.0}
        """
        now = time.time() if now is None else now
        events: list[dict] = []

        live_ids = {int(t["track_id"]) for t in tracks}
        # 本帧没出现的，标记为丢失
        for tid in list(self._lifecycle.keys()):
            if tid[0] == cam_id and tid[1] not in live_ids:
                self.store.mark_lost(cam_id, tid[1], now)

        # 候选：3 秒窗口内丢失的旧轨迹
        candidates: list[TrackState] = self.store.lost_tracks(cam_id, now)
        claimed: set[int] = set()

        for t in tracks:
            new_id = int(t["track_id"])
            life = self._bump_lifecycle(cam_id, new_id)

            # 只有「刚出现」的目标才可能是恢复对象。
            # 已经跟了几帧的老目标绝不能改写，否则会把两条轨迹反复对调。
            if life > self.new_track_max_age_frames:
                self.store.touch(cam_id, new_id, t["bbox"], now)
                continue

            self.stats["checked"] += 1
            best: TrackState | None = None
            best_dist = float("inf")

            cx, cy = center(t["bbox"])
            a_new = area(t["bbox"])

            for st in candidates:
                if st.track_id in claimed or st.track_id == new_id:
                    continue
                if not st.last_bbox:
                    continue
                dt = now - (st.lost_at or now)
                if dt > self.max_dt:
                    self.stats["rejected_age"] += 1
                    continue
                ocx, ocy = center(st.last_bbox)
                dist = math.hypot(cx - ocx, cy - ocy)
                if dist > self.max_distance:
                    self.stats["rejected_far"] += 1
                    continue
                ratio = a_new / area(st.last_bbox)
                if not (self.area_ratio_range[0] <= ratio <= self.area_ratio_range[1]):
                    self.stats["rejected_size"] += 1
                    continue
                if dist < best_dist:
                    best_dist, best = dist, st

            if best is None:
                self.store.touch(cam_id, new_id, t["bbox"], now)
                continue

            # ---- 命中：ID 覆写 ----
            old_id = best.track_id
            dt = now - (best.lost_at or now)
            self.store.recover(cam_id, old_id, new_id, now)
            self._lifecycle[(cam_id, new_id)] = self._lifecycle.get((cam_id, old_id), 0) + 1
            self._lifecycle.pop((cam_id, old_id), None)
            claimed.add(old_id)

            t["track_id"] = old_id
            t["recovered_from"] = new_id
            t["recovered"] = True
            self.stats["recovered"] += 1
            events.append({
                "new_id": new_id,
                "old_id": old_id,
                "dist": round(best_dist, 2),
                "dt": round(dt, 3),
            })
            self.store.touch(cam_id, old_id, t["bbox"], now)

        return tracks, events
