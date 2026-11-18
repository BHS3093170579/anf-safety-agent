"""单镜头时序状态缓存（CameraMemoryStore）。

工单编号：CV项目-安防智能体-04-边缘端—时序规则引擎与ID空间恢复任务工单

要解决的问题（工单 2.1 核心痛点）：
    工地里脚手架、材料堆、车辆随处可见，工友移动时视觉信号中断 1~3 秒很常见。
    ByteTrack 断轨超过 track_buffer 就会丢掉这条轨迹，人再出现时拿一个**全新 ID**。
    一旦 ID 换了，这条命从「已侵入 ROI 12 秒」变成「刚进来 0 秒」，
    告警永远攒不够时间，等于漏报。

    所以要有一块「记忆」：ID 消失前把自己的时序状态存下来，
    新 ID 若是同一个人（空间位置对得上），就把旧状态原样继承过去。

三层结构：
    CameraMemoryStore          按摄像头隔离的容器
      └─ TrackState           某一路里某个 track_id 的全部时序状态
           ├─ violation_timers  每条规则各自的计时起点
           ├─ counters          规则内部计数器（滑窗、连续帧数）
           └─ lost_at           何时被标记为丢失（供恢复器判断 Δt）

为什么必须按摄像头隔离：
    两个摄像头的像素坐标系是独立的。A 机位 (300,400) 和 B 机位 (300,400)
    完全是两个地方。跨相机比较距离会把完全无关的人当成同一个。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrackState:
    """一个 track_id 的完整时序状态。"""

    track_id: int
    first_seen: float
    last_seen: float
    last_bbox: list[float] = field(default_factory=list)

    # 每条违规规则各自的计时起点（秒级时间戳），规则名 → 起始时刻
    violation_timers: dict[str, float] = field(default_factory=dict)
    # 规则内部计数器：滑窗队列、连续帧数、累计位移等
    counters: dict[str, Any] = field(default_factory=dict)
    # 规则当前判定结果：normal / suspected / abnormal
    states: dict[str, str] = field(default_factory=dict)

    # 丢失标记。None 表示当前仍被判为在画面中
    lost_at: float | None = None
    # 从别的 ID 继承过来的次数（排查用）
    inherited: int = 0

    def age(self, now: float) -> float:
        return now - self.first_seen

    def is_lost(self) -> bool:
        return self.lost_at is not None


class CameraMemoryStore:
    """一路摄像头一张状态表。多个摄像头互不可见。

    用法：
        store = CameraMemoryStore()
        st = store.touch("Camera_02", 7, bbox, now)   # 每帧刷新
        store.mark_lost("Camera_02", 7, now)          # 这一帧没检到
        old = store.pop_lost("Camera_02", 7)          # 取走并删除
    """

    def __init__(self, lost_ttl: float = 3.0) -> None:
        # LostTracks 只保留工单要求的 3 秒窗口
        self.lost_ttl = float(lost_ttl)
        self._data: dict[str, dict[int, TrackState]] = {}
        self._lock = threading.RLock()

    # ---------- 基本读写 ----------
    def touch(
        self,
        cam_id: str,
        track_id: int,
        bbox: list[float] | None = None,
        now: float | None = None,
    ) -> TrackState:
        """这一帧看到这个 ID，刷新存活时间并清除丢失标记。"""
        now = time.time() if now is None else now
        with self._lock:
            table = self._data.setdefault(cam_id, {})
            st = table.get(track_id)
            if st is None:
                st = TrackState(track_id=track_id, first_seen=now, last_seen=now)
                table[track_id] = st
            st.last_seen = now
            st.lost_at = None
            if bbox is not None:
                st.last_bbox = [float(v) for v in bbox]
            return st

    def get(self, cam_id: str, track_id: int) -> TrackState | None:
        with self._lock:
            return self._data.get(cam_id, {}).get(track_id)

    def get_or_create(self, cam_id: str, track_id: int, now: float | None = None) -> TrackState:
        return self.get(cam_id, track_id) or self.touch(cam_id, track_id, None, now)

    # ---------- 丢失与恢复 ----------
    def mark_lost(self, cam_id: str, track_id: int, now: float | None = None) -> None:
        """这一帧没检到这个 ID。只打标记，不删——可能马上要恢复。"""
        now = time.time() if now is None else now
        with self._lock:
            st = self._data.get(cam_id, {}).get(track_id)
            if st is not None and st.lost_at is None:
                st.lost_at = now

    def lost_tracks(self, cam_id: str, now: float | None = None) -> list[TrackState]:
        """当前仍在 3 秒窗口内、可以被恢复的丢失轨迹。"""
        now = time.time() if now is None else now
        with self._lock:
            out = []
            for st in self._data.get(cam_id, {}).values():
                if st.lost_at is not None and (now - st.lost_at) <= self.lost_ttl:
                    out.append(st)
            return out

    def recover(self, cam_id: str, old_id: int, new_id: int, now: float | None = None) -> TrackState | None:
        """把 old_id 的全部时序状态搬到 new_id 上。

        工单要求「将新目标 ID 强行改写为旧 ID」——实现上有两种做法：
            a) 下游直接把 new_id 当作 old_id 用（ID 覆写）
            b) 复制状态过去，ID 保持新值
        这里用 (a)：把状态挂到 old_id 上并删掉 new_id 的空壳，
        这样下游看到的 ID 就是旧的，历史计时天然连续。
        """
        now = time.time() if now is None else now
        with self._lock:
            table = self._data.setdefault(cam_id, {})
            old = table.get(old_id)
            if old is None:
                return None
            old.last_seen = now
            old.lost_at = None
            old.inherited += 1
            # 新 ID 的空壳直接丢掉，避免表里留垃圾
            if new_id != old_id:
                table.pop(new_id, None)
            return old

    def reap(self, now: float | None = None) -> int:
        """清理超时的死轨迹，防止长时间跑下去内存无界增长。"""
        now = time.time() if now is None else now
        removed = 0
        with self._lock:
            for cam_id, table in self._data.items():
                dead = [
                    tid for tid, st in table.items()
                    if st.lost_at is not None and (now - st.lost_at) > self.lost_ttl
                ]
                for tid in dead:
                    table.pop(tid, None)
                    removed += 1
        return removed

    # ---------- 整表操作 ----------
    def reset(self, cam_id: str | None = None) -> None:
        """换视频/新会话时清空。"""
        with self._lock:
            if cam_id is None:
                self._data.clear()
            else:
                self._data.pop(cam_id, None)

    def snapshot(self, cam_id: str) -> dict[int, dict]:
        with self._lock:
            return {
                tid: {
                    "first_seen": round(st.first_seen, 3),
                    "lost": st.is_lost(),
                    "states": dict(st.states),
                    "timers": {k: round(v, 3) for k, v in st.violation_timers.items()},
                    "inherited": st.inherited,
                }
                for tid, st in self._data.get(cam_id, {}).items()
            }

    def stats(self) -> dict:
        with self._lock:
            return {
                "cams": len(self._data),
                "tracks": sum(len(t) for t in self._data.values()),
                "lost": sum(1 for t in self._data.values() for s in t.values() if s.is_lost()),
            }
