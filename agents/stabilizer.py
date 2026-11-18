# -*- coding: utf-8 -*-
"""
DetectionStabilizer —— 让检测框在时间上连续稳定

解决的问题：
    no-hardhat 的单帧置信度只有 0.1~0.35，模型每帧都在犹豫，
    导致框一闪一闪。hardhat 置信度 0.5~0.8，框很稳。

思路（不重训模型）：
    1. 低阈值检出也纳入轨迹（no-hardhat 可以低到 0.08）
    2. 用 IOU + 质心距离做跨帧匹配，给每个目标分配 track_id
    3. 某一帧漏检时，用轨迹上一帧的位置补框（最多补 max_lost 帧）
    4. 连续命中 min_hits 次才正式输出（过滤单帧误检）

输出：
    稳定后的检测列表。补出来的框带 "tracked": True 标记，
    调用方可据此画虚线，和真实检出区分。

用法：
    stab = DetectionStabilizer()
    for frame in frames:
        raw = detector.detect(frame)
        stable = stab.update(raw)
        draw(frame, stable)
"""
from __future__ import annotations

import math

# 几何计算统一走 agents.geometry，保留旧名字方便调用点不变
from agents.geometry import bbox_iou as _iou

class Track:
    """单条轨迹。

    cx / cy 是缓存下来的质心，每次更新 bbox 时同步刷新。
    匹配阶段先用它做一次便宜的排除，避免对每条轨迹都算 IOU
    —— 目标数一多，纯 O(n²) 的 IOU 计算会从 0.4ms 涨到 40ms。
    """
    __slots__ = ("tid", "bbox", "cls", "score", "hits", "lost", "class_id",
                 "cx", "cy", "w")

    def __init__(self, tid: int, bbox: list, cls: str, class_id: int, score: float):
        self.tid = tid
        self.cls = cls
        self.class_id = class_id
        self.score = score
        self.hits = 1
        self.lost = 0
        self.set_bbox(bbox)

    def set_bbox(self, bbox: list) -> None:
        self.bbox = list(bbox)
        self.cx = (bbox[0] + bbox[2]) / 2.0
        self.cy = (bbox[1] + bbox[3]) / 2.0
        self.w = max(1.0, bbox[2] - bbox[0])

    @property
    def confirmed(self) -> bool:
        return self.lost == 0

class DetectionStabilizer:
    """
    检测结果稳定器。

    Args:
        iou_thres:      IOU 匹配阈值（低于此值不算同一目标）
        dist_ratio:     质心距离阈值（相对于框宽度的倍数）
        max_lost:       允许连续漏检多少帧（超过则丢弃轨迹）
        min_hits:       连续命中多少次才输出（过滤单帧误检）
        low_conf_keep:  低置信度但已确认的轨迹，保留阈值（低于此值也丢弃）
    """

    def __init__(self, iou_thres: float = 0.20, dist_ratio: float = 1.5,
                 max_lost: int = 2, min_hits: int = 2,
                 low_conf_keep: float = 0.04, track_min_score: float = 0.15):
        """
        Args:
            iou_thres:       IOU 匹配阈值
            dist_ratio:      质心距离阈值（相对框宽倍数）
            max_lost:        允许补框的最大连续漏检帧数。
                             **必须很小**——补框用的是旧位置，丢得越久偏得越远。
            min_hits:        连续命中多少次才输出
            low_conf_keep:   低于此分的轨迹直接丢弃
            track_min_score: **补框的最低置信度**。低于此值不补，
                             否则会出现「0.06 的框也标跟踪」这种离谱情况。
        """
        self.iou_thres = iou_thres
        self.dist_ratio = dist_ratio
        self.max_lost = max_lost
        self.min_hits = min_hits
        self.low_conf_keep = low_conf_keep
        self.track_min_score = track_min_score

        self._tracks: dict[int, Track] = {}
        self._next_tid = 1

    def reset(self):
        self._tracks.clear()
        self._next_tid = 1

    def update(self, detections: list[dict]) -> list[dict]:
        """
        输入本帧原始检测，返回稳定后的检测列表。

        返回的每个 dict 在原字段基础上可能多了：
            "track_id": int     轨迹 ID
            "tracked":  bool    True 表示这是补出来的框（本帧没检出）
        """
        used_tids: set[int] = set()
        matched: list[tuple[int, dict]] = []   # (track_id, detection)

        # ---------- 1. 检测与轨迹匹配 ----------
        for d in detections:
            bbox = d.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            cls = d.get("class", "")
            best_tid = None
            best_score = 0.0

            # 新框的质心和宽度，供下面的快速排除用
            ncx = (bbox[0] + bbox[2]) / 2.0
            ncy = (bbox[1] + bbox[3]) / 2.0
            nw = max(1.0, bbox[2] - bbox[0])

            for tid, tr in self._tracks.items():
                if tid in used_tids:
                    continue
                # 类别必须一致，避免安全帽和人互相串
                if tr.cls != cls:
                    continue

                # 快速排除：只要质心在「允许最大距离」之外就直接跳过，
                # 连 IOU 都不用算。dist_ratio 是相对框宽的倍数，
                # 再乘 2 留足余量（尺寸比检查会再筛一次）。
                max_d = nw * self.dist_ratio * 2.0
                dx = ncx - tr.cx
                if dx > max_d or dx < -max_d:
                    continue
                dy = ncy - tr.cy
                if dy > max_d or dy < -max_d:
                    continue

                # 尺寸一致性：新框和轨迹框大小不能差太多。
                # 没有这道检查时，一个误检的大框会把轨迹拖走，
                # 后续补框就会补出「横跨半屏的巨大虚线框」。
                old_w = tr.w
                old_h = max(1.0, tr.bbox[3] - tr.bbox[1])
                new_h = max(1.0, bbox[3] - bbox[1])
                if not (0.6 <= nw / old_w <= 1.7 and 0.6 <= new_h / old_h <= 1.7):
                    continue

                iou = _iou(tr.bbox, bbox)
                dist = math.hypot(dx, dy)
                dist_norm = dist / nw

                # 综合评分：IOU 高 或 质心近
                ok = (iou >= self.iou_thres) or (dist_norm <= self.dist_ratio)
                if not ok:
                    continue

                score = max(iou, 1.0 - dist_norm / self.dist_ratio)
                if score > best_score:
                    best_score = score
                    best_tid = tid

            if best_tid is not None:
                tr = self._tracks[best_tid]
                tr.set_bbox(bbox)      # 同步刷新缓存的质心与宽度
                tr.score = float(d.get("score", 0.0))
                tr.hits += 1
                tr.lost = 0
                used_tids.add(best_tid)
                matched.append((best_tid, d))
            else:
                # 新目标
                tid = self._next_tid
                self._next_tid += 1
                self._tracks[tid] = Track(
                    tid, bbox, cls, int(d.get("class_id", 0)),
                    float(d.get("score", 0.0)))
                used_tids.add(tid)
                matched.append((tid, d))

        # ---------- 2. 未匹配的轨迹：老化 / 补框 ----------
        for tid in list(self._tracks.keys()):
            if tid in used_tids:
                continue
            tr = self._tracks[tid]
            tr.lost += 1
            # 低分轨迹快速淘汰
            if tr.score < self.low_conf_keep:
                del self._tracks[tid]
                continue
            if tr.lost > self.max_lost:
                del self._tracks[tid]
                continue

        # ---------- 3. 组装输出 ----------
        out: list[dict] = []

        # 3a. 本帧真实检出的（按轨迹命中数过滤）
        for tid, d in matched:
            tr = self._tracks.get(tid)
            if tr is None:
                continue
            if tr.hits < self.min_hits:
                continue
            item = dict(d)
            item["track_id"] = tid
            item["tracked"] = False
            out.append(item)

        # 3b. 本帧漏检但轨迹仍活着的 → 补框
        # 严格条件：只补「刚丢 1~2 帧」且「上次置信度够高」的目标。
        # 放宽会导致人走开了框还飘在画面上的跟踪漂移。
        matched_tids = {tid for tid, _ in matched}
        for tid, tr in self._tracks.items():
            if tid in matched_tids:
                continue
            if tr.hits < self.min_hits:
                continue
            if tr.lost > self.max_lost:
                continue
            if tr.score < self.track_min_score:
                continue          # 低分轨迹不补，避免 0.06 的框也画出来
            out.append({
                "class": tr.cls,
                "class_id": tr.class_id,
                "bbox": list(tr.bbox),
                "score": tr.score,
                "track_id": tid,
                "tracked": True,          # 标记：这是补出来的
            })

        return out

    def stats(self) -> dict:
        """当前轨迹统计（调试用）"""
        return {
            "tracks": len(self._tracks),
            "confirmed": sum(1 for t in self._tracks.values() if t.confirmed),
            "lost": sum(1 for t in self._tracks.values() if t.lost > 0),
        }
