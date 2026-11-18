# -*- coding: utf-8 -*-
"""
JudgmentAgent —— 工单编号：CV专高5-10（工单 02 - 任务 5）

职责：
    根据检测结果判断是否触发告警（规则引擎）。

支持三种规则类型，可单独使用，也可多条并存：

    1. class    —— 类别规则（安全帽 / 火灾）
                   检出 violation_classes 即违规，配合连续帧确认

    2. posture  —— 姿态规则（睡岗）
                   用关键点算出的姿态指标判断低头，持续 N 帧才告警

    3. tracking —— 跟踪规则（徘徊）
                   IOU + 质心距离给每人分配 ID，停留够久才告警

【多规则并存】—— 值班室综合场景的核心
    同一个摄像头、同一个模型，可以同时跑「睡岗」和「徘徊」两条规则，
    各自独立计数、独立告警。一路摄像头一个模型覆盖两个隐患，
    比各跑一个模型省一半算力。

为什么要「连续帧确认」：
    单帧误检在实际项目里很常见（反光、遮挡、模型抖动）。
    连续 N 帧都命中，才认为事件真实发生，这是压误报最有效的手段。

状态说明：
    计数器、跟踪器都保存在对象内部，不放进 LangGraph 的 State。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

# IOU 统一由 agents.geometry 提供，避免三个文件各写一份实现
from agents.geometry import bbox_iou as _iou


class SimpleTracker:
    """极简跟踪器：给画面里的每个人分配稳定 ID，统计停留时长。

    匹配策略 = IOU 与质心距离取较大值：
        低帧率（1~5fps）下人走动幅度大，IOU 常常直接归零，
        单靠 IOU 会让同一个人的 ID 反复断裂；质心距离能兜住这种情况。

    刻意不引入 DeepSORT/ByteTrack：
        徘徊检测只需要「同一个人停留多久」，固定机位下这套逻辑已足够，
        零依赖且可解释。复杂跟踪器是过度设计。
    """

    def __init__(self, iou_threshold: float = 0.25, max_missing: int = 3) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_missing = int(max_missing)
        self._next_id = 1
        self.tracks: dict[int, dict] = {}
        # 每个轨迹额外记录：
        #   center  —— 上一帧质心
        #   travel  —— 累计位移（归一化：除以人体框宽度）
        # 用来区分「来回走动」和「静止伫立」

    @staticmethod
    def _match_score(a: list[float], b: list[float]) -> float:
        iou_v = _iou(a, b)
        if iou_v > 0:
            return iou_v
        acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
        bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        dist = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5
        aw, ah = a[2] - a[0], a[3] - a[1]
        bw, bh = b[2] - b[0], b[3] - b[1]
        diag = (((aw + bw) / 2) ** 2 + ((ah + bh) / 2) ** 2) ** 0.5
        if diag <= 0:
            return 0.0
        return max(0.0, 1.0 - dist / (diag * 2.0)) * 0.8

    def update(self, detections: list[dict], timestamp: float) -> list[dict]:
        assigned: set[int] = set()
        for tr in self.tracks.values():
            best_i, best_v = -1, 0.0
            for i, det in enumerate(detections):
                if i in assigned:
                    continue
                v = self._match_score(tr["bbox"], det["bbox"])
                if v > best_v:
                    best_v, best_i = v, i
            if best_v >= self.iou_threshold and best_i >= 0:
                det = detections[best_i]
                assigned.add(best_i)
                tr["bbox"] = det["bbox"]
                tr["frames"] += 1
                tr["missing"] = 0
                tr["score"] = max(tr["score"], det["score"])
                tr["last_ts"] = timestamp
                tr["det"] = det
                # 累计位移：用框宽度归一化，避免远近影响
                ncx, ncy = (det["bbox"][0] + det["bbox"][2]) / 2, (det["bbox"][1] + det["bbox"][3]) / 2
                pcx, pcy = tr.get("center", (ncx, ncy))
                step = ((ncx - pcx) ** 2 + (ncy - pcy) ** 2) ** 0.5
                body_w = max(1.0, abs(det["bbox"][2] - det["bbox"][0]))
                tr["travel"] = tr.get("travel", 0.0) + step / body_w
                tr["center"] = (ncx, ncy)
            else:
                tr["missing"] += 1

        for i, det in enumerate(detections):
            if i in assigned:
                continue
            self.tracks[self._next_id] = {
                "bbox": det["bbox"], "frames": 1, "missing": 0,
                "score": det["score"], "first_ts": timestamp,
                "last_ts": timestamp, "class_id": det.get("class_id", 0),
                "det": det,
                "center": ((det["bbox"][0] + det["bbox"][2]) / 2,
                           (det["bbox"][1] + det["bbox"][3]) / 2),
                "travel": 0.0,
            }
            self._next_id += 1

        for tid in [t for t, tr in self.tracks.items() if tr["missing"] > self.max_missing]:
            del self.tracks[tid]

        return [
            {"track_id": tid, "bbox": tr["bbox"], "frames": tr["frames"],
             "duration": round(tr["last_ts"] - tr["first_ts"], 2),
             "score": tr["score"], "class_id": tr["class_id"], "det": tr["det"],
             "travel": round(tr.get("travel", 0.0), 2)}
            for tid, tr in self.tracks.items() if tr["missing"] == 0
        ]

    def summary(self) -> list[dict]:
        out = [
            {"track_id": tid, "frames": tr["frames"],
             "duration": round(tr["last_ts"] - tr["first_ts"], 1),
             "travel": round(tr.get("travel", 0.0), 2),
             "bbox": tr["bbox"]}
            for tid, tr in self.tracks.items() if tr["missing"] == 0
        ]
        return sorted(out, key=lambda x: -x["frames"])


class Rule:
    """一条判断规则，独立维护自己的计数与跟踪状态。"""

    def __init__(self, name: str, cfg: dict, class_map: dict[int, str]) -> None:
        self.name = name
        # 规则类型：显式声明优先；未声明时按配置里的特征自动推断，
        # 避免「写了跟踪参数却因为忘了写 type 而退化成类别规则」这种坑。
        self.type = cfg.get("type")
        if not self.type:
            if cfg.get("track_iou_threshold") is not None or cfg.get("track_max_missing") is not None:
                self.type = "tracking"
            elif cfg.get("posture"):
                self.type = "posture"
            else:
                self.type = "class"
        self.class_map = class_map
        self.min_frames = max(1, int(cfg.get("min_consecutive_frames", 1)))
        self.posture = cfg.get("posture")
        self.alert_cfg = cfg["alert"]
        self.title = self.alert_cfg.get("title", name)

        self.tracker: SimpleTracker | None = None
        # 徘徊要求「确实在走动」：累计位移低于该阈值视为静止伫立，不告警。
        # 单位是「人体框宽度」的倍数，抵消远近影响。
        self.min_travel = float(cfg.get("min_travel", 0.0))
        if self.type == "tracking":
            self.tracker = SimpleTracker(
                iou_threshold=cfg.get("track_iou_threshold", 0.25),
                max_missing=cfg.get("track_max_missing", 3),
            )

        self._streak: dict[Any, int] = defaultdict(int)
        self._fired: set[Any] = set()

    # ---------- 姿态判断 ----------
    def _passes_posture(self, det: dict) -> bool:
        if not self.posture:
            return True
        value = det.get(self.posture.get("metric", "head_drop"))
        if value is None:
            return False          # 关键点不可靠 → 不算违规，避免误报
        thr = float(self.posture.get("threshold", 0.0))
        direction = self.posture.get("direction", "greater")
        return value > thr if direction == "greater" else value < thr

    # ---------- 告警构造 ----------
    def _make_alert(self, cid: int, hits: list[dict], timestamp: float,
                    key: Any, frame_detections: list[dict],
                    extra: dict | None = None) -> dict:
        best = max(hits, key=lambda d: d["score"])
        alert = {
            "event_type": self.alert_cfg["event_type"],
            "level": self.alert_cfg["level"],
            "title": self.alert_cfg["title"],
            "message": self.alert_cfg["message"],
            "rule_name": self.name,
            "class_name": self.class_map.get(cid, str(cid)),
            "class_id": cid,
            "timestamp": round(float(timestamp), 3),
            "count": len(hits),
            "max_score": round(float(best["score"]), 4),
            "bbox": best["bbox"],
            "mask_polygons": best.get("mask_polygons"),
            "area_ratio": best.get("area_ratio"),
            "all_targets": [
                {
                    "class_name": self.class_map.get(int(d["class_id"]), ""),
                    "class_id": int(d["class_id"]),
                    "bbox": d["bbox"],
                    "score": d["score"],
                    "mask_polygons": d.get("mask_polygons"),
                    "area_ratio": d.get("area_ratio"),
                }
                for d in sorted(hits, key=lambda x: -x["score"])
            ],
            # 本帧全部检测（含合规目标），截图时一并画出
            "frame_detections": [
                {
                    "class_name": self.class_map.get(int(d["class_id"]), ""),
                    "class_id": int(d["class_id"]),
                    "bbox": d["bbox"],
                    "score": d["score"],
                }
                for d in frame_detections
            ],
            "consecutive_frames": self._streak[key],
        }
        if extra:
            alert.update(extra)
        return alert

    # ---------- 主判断 ----------
    def judge(self, detections: list[dict], timestamp: float,
              violation_ids: set[int]) -> list[dict]:
        suspects = [
            d for d in detections
            if int(d["class_id"]) in violation_ids and self._passes_posture(d)
        ]
        if self.type == "tracking":
            return self._judge_tracking(suspects, timestamp, detections)
        return self._judge_streak(suspects, timestamp, detections)

    def _judge_streak(self, suspects: list[dict], timestamp: float,
                      frame_detections: list[dict]) -> list[dict]:
        """按类别累计连续命中帧数（class / posture 规则）。"""
        present: dict[int, list[dict]] = defaultdict(list)
        for d in suspects:
            present[int(d["class_id"])].append(d)

        alerts = []
        for cid in list(present.keys()):
            self._streak[cid] += 1
            if self._streak[cid] >= self.min_frames and cid not in self._fired:
                self._fired.add(cid)
                alerts.append(self._make_alert(cid, present[cid], timestamp,
                                               cid, frame_detections))
        # 本帧没有出现的类别清零，允许下次重新触发
        for key in list(self._streak.keys()):
            if key not in present:
                self._streak[key] = 0
                self._fired.discard(key)
        return alerts

    def _judge_tracking(self, suspects: list[dict], timestamp: float,
                        frame_detections: list[dict]) -> list[dict]:
        """同一个人停留够久才告警。"""
        assert self.tracker is not None
        alive = self.tracker.update(suspects, timestamp)
        active = {t["track_id"] for t in alive}

        for tid in list(self._streak.keys()):
            if tid not in active:
                self._streak.pop(tid, None)
                self._fired.discard(tid)

        alerts = []
        for t in alive:
            tid = t["track_id"]
            self._streak[tid] = t["frames"]
            moved_enough = t.get("travel", 0.0) >= self.min_travel
            if (t["frames"] >= self.min_frames and moved_enough
                    and tid not in self._fired):
                self._fired.add(tid)
                alerts.append(self._make_alert(
                    t["class_id"], [t["det"]], timestamp, tid, frame_detections,
                    extra={
                        "track_id": tid,
                        "dwell_seconds": t["duration"],
                        "travel": round(t.get("travel", 0.0), 2),
                        "message": (f"人员 #{tid} 在区域停留 {t['duration']:.0f} 秒，"
                                    f"累计走动 {t.get('travel', 0):.1f} 个身位，疑似徘徊"),
                    },
                ))
        return alerts

    def summary(self) -> list[dict]:
        return self.tracker.summary() if self.tracker else []


class JudgmentAgent:
    """规则判断 Agent：一条或多条规则并行判断。"""

    def __init__(self, scene_cfg: dict) -> None:
        self.scene = scene_cfg
        self.name = scene_cfg["name"]
        self.violation_ids = set(int(c) for c in scene_cfg["violation_classes"])
        self.class_map = {int(k): v for k, v in scene_cfg["classes"].items()}
        rules_cfg = scene_cfg.get("rules", {})

        multi = rules_cfg.get("multi_rules")
        if multi:
            # 多规则并存：值班室综合场景（睡岗 + 徘徊）
            self.rules = [Rule(r.get("name", f"rule{i}"), r, self.class_map)
                          for i, r in enumerate(multi)]
        else:
            # 单规则场景：把顶层 rules + alert 包装成一条规则
            single = dict(rules_cfg)
            single["alert"] = scene_cfg["alert"]
            self.rules = [Rule("main", single, self.class_map)]

        self._frame_detections: list[dict] = []

    @property
    def rule_names(self) -> list[str]:
        return [r.name for r in self.rules]

    def judge(self, detections: list[dict], timestamp: float) -> list[dict]:
        """所有规则各判一次，合并告警结果。"""
        self._frame_detections = detections
        alerts: list[dict] = []
        for rule in self.rules:
            alerts.extend(rule.judge(detections, timestamp, self.violation_ids))
        return alerts

    def track_summary(self) -> list[dict]:
        """所有跟踪规则的状态合并（供界面展示）。"""
        out = []
        for rule in self.rules:
            for item in rule.summary():
                out.append({**item, "rule": rule.name})
        return sorted(out, key=lambda x: -x["frames"])

    def rule_progress(self) -> dict[str, int]:
        """各规则当前命中的连续帧数（界面显示用）。"""
        result = {}
        for rule in self.rules:
            if rule.type == "tracking" and rule.tracker:
                items = rule.summary()
                result[rule.name] = max((i["frames"] for i in items), default=0)
            else:
                result[rule.name] = max(rule._streak.values(), default=0)
        return result

    def reset(self) -> None:
        for rule in self.rules:
            rule._streak.clear()
            rule._fired.clear()
            if rule.type == "tracking":
                rule.tracker = SimpleTracker(
                    rule.tracker.iou_threshold if rule.tracker else 0.25,
                    rule.tracker.max_missing if rule.tracker else 3,
                )
