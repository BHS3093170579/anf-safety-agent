# -*- coding: utf-8 -*-
"""边缘端主流水线调度器。

工单编号：CV项目-安防智能体-05（工单 05 · Pipeline 协同）

一帧的处理顺序
--------------
    ModelService.process   →  YOLO-pose 检测 + ByteTrack 追踪（工单 03）
    TrackRecovery          →  3 秒内遮挡的 ID 空间恢复（工单 04）
    RulesEngine.update     →  五大业务规则的时序状态机（工单 04）
    状态迁移检测           →  某 Track 由 normal/suspected 首次变为 abnormal
    ReidExtractor.extract  →  **仅此一刻**裁剪人体、跑 OSNet、取 512 维特征
    EventDispatcher.push   →  拼装事件、非阻塞入队，后台线程异步 POST

核心约束（工单验收标准第 1 条）
------------------------------
严禁在非 abnormal 状态下运行 Re-ID 特征提取网络。实现上靠三重保证：
  1. 只有 `rs.state == "abnormal"` 才进入提取分支；
  2. 必须是状态**跃迁**（上一帧不是 abnormal），持续 abnormal 不会重复提取；
  3. 每个 Track 只提取一次，用 `reid_extracted` 标志位兜住。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from edge.event_dispatcher import AlertEvent, EventDispatcher
from utils.reid_extractor import ReidExtractor

# 状态严重度：数值越大越严重
SEVERITY = {"normal": 0, "suspected": 1, "abnormal": 2}


def _iou(a, b) -> float:
    """两个 xyxy 框的 IoU；任一为空返回 0。"""
    try:
        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        bx1, by1, bx2, by2 = [float(v) for v in b]
    except Exception:
        return 0.0
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


# ---------------------------------------------------------------------- #
@dataclass
class FrameReport:
    """单帧处理结果。"""

    cam_id: str
    frame_idx: int = 0
    video_time: float = 0.0
    bodies: int = 0
    rule_results: list = field(default_factory=list)
    alerts: list[dict] = field(default_factory=list)
    recovered: int = 0
    reid_count: int = 0             # 本帧新提取的 Re-ID 特征条数
    elapsed_ms: float = 0.0
    # 本帧的人体框（含 track_id），供界面画框用；规则结果里不带 bbox
    tracks: list[dict] = field(default_factory=list)

    @property
    def abnormal_rules(self) -> list:
        return [r for r in self.rule_results if r.state == "abnormal"]


# ---------------------------------------------------------------------- #
class EdgePipeline:
    """边缘端主流水线。"""

    def __init__(
        self,
        cams: dict[str, dict],
        rules: dict,
        model_service,
        reid_extractor: ReidExtractor | None = None,
        dispatcher: EventDispatcher | None = None,
        memory_store=None,
        snapshot_dir: str | Path | None = None,
        save_snapshot: bool = True,
        reid_on_rules: tuple[str, ...] | None = None,
        detectors: dict[str, Any] | None = None,
        verbose: bool = False,
    ) -> None:
        from edge.rules_engine import RulesEngine  # noqa: PLC0415
        from utils.camera_memory_store import CameraMemoryStore  # noqa: PLC0415
        from utils.track_recovery import TrackRecovery  # noqa: PLC0415

        self.cams = cams
        self.rules = rules
        self.svc = model_service
        self.reid = reid_extractor or ReidExtractor()
        self.dispatcher = dispatcher
        self.verbose = verbose

        self.store = memory_store or CameraMemoryStore(
            lost_ttl=float(rules.get("track_recovery", {}).get("lost_ttl", 3.0))
        )
        # 额外检测器：{"helmet": DetectionAgent, "fire": DetectionAgent}
        # pose 由 model_service 负责；头盔/火灾是按相机 scene 叠加的第二、三路推理
        self.detectors = detectors or {}
        self._scenes = {cid: list(cfg.get("scene") or []) for cid, cfg in cams.items()}
        self._fire_last_alert: dict[str, float] = {}
        rc = rules.get("track_recovery", {})
        self.recovery = TrackRecovery(
            self.store,
            max_distance=float(rc.get("max_distance", 50.0)),
            max_dt=float(rc.get("max_dt", 3.0)),
        )

        root = Path(__file__).resolve().parent.parent
        self.snapshot_dir = Path(snapshot_dir or (root / "outputs" / "snapshots"))
        if save_snapshot:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.save_snapshot = save_snapshot

        # 哪些规则触发 Re-ID 建档（None = 全部规则）
        self.reid_on_rules = set(reid_on_rules) if reid_on_rules else None

        self.engines: dict[str, RulesEngine] = {}
        for cam_id, cfg in cams.items():
            self.engines[cam_id] = RulesEngine(cam_id, cfg, rules, store=self.store)

        # 状态机：记录每个 (cam, track) 上一帧的聚合状态
        self._prev_state: dict[tuple[str, int], str] = {}
        # 已建档标志：每个 (cam, track) 只提取一次 Re-ID
        self._reid_done: dict[tuple[str, int], dict] = {}
        # 告警冷却：同一个 (cam, track, rule) 在 cooldown 秒内只上报一次。
        # 规则状态会在 abnormal/suspected 之间抖动（滑窗投票、跟踪短暂丢失都会），
        # 不设冷却的话同一个人会被反复建档、反复推送。
        self.alert_cooldown = 30.0
        self._last_alert: dict[tuple[str, int, str], float] = {}
        self._recent_boxes: dict[tuple[str, str], list] = {}

        self.frame_idx = 0
        self.timings: list[float] = []

    # ------------------------------------------------------------------ #
    # 单帧
    # ------------------------------------------------------------------ #
    def process(
        self,
        cam_id: str,
        frame: np.ndarray,
        now: float | None = None,
        video_time: float | None = None,
        demo: bool = False,
    ) -> FrameReport:
        """处理一帧：检测→追踪→恢复→规则→按需 Re-ID→入队。"""
        t0 = time.perf_counter()
        self.frame_idx += 1
        now = time.time() if now is None else now
        video_time = 0.0 if video_time is None else video_time

        rep = FrameReport(cam_id=cam_id, frame_idx=self.frame_idx, video_time=video_time)

        # 1) 检测 + 追踪
        res = self.svc.process(cam_id, frame)
        rep.bodies = len(res.tracks)

        # 2) ID 空间恢复（遮挡后找回旧 ID，状态计时不清零）
        tracks, rec_evts = self.recovery.try_recover(cam_id, res.tracks, now=now)
        rep.recovered = len(rec_evts)

        # 3) 时序规则
        engine = self.engines.get(cam_id)
        if engine is None:
            rep.elapsed_ms = (time.perf_counter() - t0) * 1000
            return rep

        rep.tracks = tracks

        # 2.5) 按相机 scene 叠加安全帽 / 火灾两路能力
        scenes = self._scenes.get(cam_id, [])
        helmet_det = self.detectors.get("helmet")
        if helmet_det is not None and "helmet" in scenes and tracks:
            try:
                self._attach_helmet(tracks, helmet_det.detect(frame))
            except Exception as exc:  # noqa: BLE001
                if self.verbose:
                    print(f"[Pipeline] 安全帽检测失败 {cam_id}: {exc}")
        fire_det = self.detectors.get("fire")
        if fire_det is not None and "fire" in scenes:
            try:
                self._run_fire(cam_id, fire_det.detect(frame), frame, now, video_time, rep)
            except Exception as exc:  # noqa: BLE001
                if self.verbose:
                    print(f"[Pipeline] 火灾检测失败 {cam_id}: {exc}")

        rule_results = engine.update(
            tracks,
            now=now,
            frame_shape=(frame.shape[0], frame.shape[1]),
            demo=demo,
        )
        rep.rule_results = rule_results

        # 4) 按 Track 聚合出最严重的状态
        latest: dict[int, Any] = {}
        for rs in rule_results:
            k = rs.track_id
            if k not in latest or SEVERITY.get(rs.state, 0) > SEVERITY.get(latest[k].state, 0):
                latest[k] = rs

        # 5) 状态跃迁 → 按需提取 Re-ID（这是整个工单的核心）
        for tid, rs in latest.items():
            key = (cam_id, tid)
            prev = self._prev_state.get(key, "normal")
            self._prev_state[key] = rs.state

            if rs.state != "abnormal":
                continue
            if prev == "abnormal":          # 持续 abnormal，不重复提取
                continue
            if key in self._reid_done:      # 已建档
                continue
            if self.reid_on_rules and rs.rule not in self.reid_on_rules:
                continue

            # RuleResult 本身不带 bbox（规则只管判定），从本帧 tracks 里按 track_id 取，
            # 否则抓拍永远拿不到框、只能存整帧。
            tbox = None
            for t in tracks:
                if int(t.get("track_id", -1)) == tid:
                    tbox = t.get("bbox")
                    break

            # 冷却 1：同一 (cam, track, rule) 30 秒内只报一次
            ck = (cam_id, tid, rs.rule)
            last = self._last_alert.get(ck)
            if last is not None and now - last < self.alert_cooldown:
                continue

            # 冷却 2：跟踪 ID 跳变时同一位置的人会换 tid 再报一遍，
            # 用「上次告警框 + 30 秒内 + IoU>0.3」认定是同一个人。
            if tbox is not None:
                # 同一个 (相机, 规则) 可能有多个不同的人在报，所以要存「最近若干条框」，
                # 只存最后一条会被别人覆盖，导致同一个人被反复上报。
                bk = (cam_id, rs.rule)
                lst = self._recent_boxes.setdefault(bk, [])
                lst[:] = [(b, t) for b, t in lst if now - t < self.alert_cooldown]
                if any(_iou(b, tbox) > 0.3 for b, _ in lst):
                    continue
                lst.append((list(tbox), now))

            self._last_alert[ck] = now

            self._mint_identity(cam_id, tid, rs, frame, now, video_time, rep, tbox)

        # 6) 收尾：清理已消失目标的追踪状态，避免字典无限增长
        alive = {(cam_id, t["track_id"]) for t in tracks if "track_id" in t}
        for key in list(self._prev_state.keys()):
            if key[0] == cam_id and key not in alive:
                self._prev_state.pop(key, None)

        rep.elapsed_ms = (time.perf_counter() - t0) * 1000
        self.timings.append(rep.elapsed_ms)
        return rep

    # ------------------------------------------------------------------ #
    def _attach_helmet(self, tracks: list[dict], dets: list[dict]) -> None:
        """把头盔检测框贴到人体轨迹上：头框中心落在身体框内即视为同一目标。

        只写 `has_helmet` 字段，真正的规则判定交给 RulesEngine._rule_helmet，
        这样滑窗投票、离岗抑制等逻辑不用重复实现。
        """
        heads = [d for d in dets if d.get("class") in ("hardhat", "no-hardhat")]
        if not heads:
            return
        used: set[int] = set()
        for t in tracks:
            tb = t.get("bbox") or [0, 0, 0, 0]
            bw = max(1.0, float(tb[2]) - float(tb[0]))
            cx = (float(tb[0]) + float(tb[2])) / 2.0
            best_i, best_d, best_dist = -1, None, float("inf")
            for i, d in enumerate(heads):
                if i in used:
                    continue
                hb = d.get("bbox") or [0, 0, 0, 0]
                hx = (float(hb[0]) + float(hb[2])) / 2.0
                hy = (float(hb[1]) + float(hb[3])) / 2.0
                if not (float(tb[0]) <= hx <= float(tb[2]) and float(tb[1]) <= hy <= float(tb[3])):
                    continue
                dist = abs(hx - cx) / bw
                if dist < best_dist:
                    best_dist, best_i, best_d = dist, i, d
            if best_d is not None:
                used.add(best_i)
                t["has_helmet"] = best_d.get("class") == "hardhat"

    def _run_fire(
        self,
        cam_id: str,
        dets: list[dict],
        frame: np.ndarray,
        now: float,
        video_time: float,
        rep: FrameReport,
    ) -> None:
        """火灾/烟雾命中即告警；带 5 秒冷却，避免每帧都建一条。"""
        from edge.rules_engine import RuleResult  # noqa: PLC0415

        fires = [d for d in dets if d.get("class") in ("fire", "smoke")]
        if not fires:
            return
        if now - self._fire_last_alert.get(cam_id, -1e9) < 5.0:
            return
        self._fire_last_alert[cam_id] = now
        d = max(fires, key=lambda x: float(x.get("score", 0)))
        rs = RuleResult("fire", "abnormal", 0.0,
                        f"检出 {d.get('class')}（{float(d.get('score', 0)):.2f}）")
        rep.rule_results.append(rs)
        self._mint_identity(cam_id, None, rs, frame, now, video_time, rep, d.get("bbox"))

    # ------------------------------------------------------------------ #
    def _mint_identity(
        self,
        cam_id: str,
        track_id: int,
        rs,
        frame: np.ndarray,
        now: float,
        video_time: float,
        rep: FrameReport,
        track_bbox: list | None = None,
    ) -> None:
        """为刚转为 abnormal 的目标建档：裁剪 → OSNet → 512 维 → 入队。"""
        bbox = track_bbox or getattr(rs, "bbox", None)
        feature: list[float] = []

        try:
            if bbox:
                feat = self.reid.embed_boxes(frame, [bbox])
                if len(feat):
                    feature = feat[0].tolist()
            else:
                # 没有 bbox（如离岗规则无目标）时，用整帧顶部区域兜底
                h, w = frame.shape[:2]
                feat = self.reid.embed_boxes(frame, [[0, 0, w, h]])
                if len(feat):
                    feature = feat[0].tolist()
        except Exception as exc:  # Re-ID 失败不能拖垮主流水线
            if self.verbose:
                print(f"[Pipeline] Re-ID 失败 {cam_id}#{track_id}: {exc}")

        event_id = uuid.uuid4().hex
        snap_path = None
        if self.save_snapshot and bbox:
            snap_path = self._save_snapshot(event_id, frame, bbox)

        self._reid_done[(cam_id, track_id)] = {
            "event_id": event_id,
            "feature_dim": len(feature),
            "at": now,
            "rule": rs.rule,
        }
        rep.reid_count += 1

        evt = AlertEvent(
            event_id=event_id,
            camera_id=cam_id,
            track_id=int(track_id) if track_id is not None else 0,
            violation_type=rs.rule,
            severity="abnormal",
            timestamp=time.time(),          # 上报用墙钟时间
            video_time=round(video_time, 2),
            reid_feature=feature,
            snapshot_path=str(snap_path) if snap_path else None,
            message=getattr(rs, "reason", "") or f"{rs.rule} 告警",
            extra={"elapsed_sum": round(float(getattr(rs, "elapsed", 0.0)), 2)},
        )

        rep.alerts.append(evt.to_payload())
        if self.dispatcher is not None:
            ok = self.dispatcher.push(evt)
            if self.verbose:
                state = "已入队" if ok else "队列满丢弃"
                print(
                    f"[Pipeline] 建档 {cam_id}#{track_id} {rs.rule} "
                    f"特征{len(feature)}维 {self.reid.last_ms:.1f}ms → {state}"
                )

    def _save_snapshot(self, event_id: str, frame: np.ndarray, bbox) -> Path | None:
        try:
            img = frame.copy()
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            path = self.snapshot_dir / f"{event_id}.jpg"
            cv2.imwrite(str(path), img)
            return path
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # 视频
    # ------------------------------------------------------------------ #
    def process_video(
        self,
        cam_id: str,
        video_path: str | Path,
        sample_fps: float = 5.0,
        max_frames: int | None = None,
        demo: bool = False,
        real_time_sleep: bool = False,
    ) -> dict:
        """跑一段视频，返回统计（平均 FPS、建档数、队列情况等）。"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"打不开视频：{video_path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(round(src_fps / sample_fps)))
        interval = 1.0 / sample_fps

        t_start = time.perf_counter()
        wall0 = time.time()
        idx = 0
        kept = 0
        fwd: list[float] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % step:
                continue
            kept += 1
            if max_frames and kept > max_frames:
                break

            video_time = kept / sample_fps
            if real_time_sleep:
                # 模拟真实 5FPS 节拍：处理完再补足到下一帧的时间点
                due = wall0 + kept * interval
                gap = due - time.time()
                if gap > 0:
                    time.sleep(gap)

            self.process(
                cam_id,
                frame,
                # 规则引擎用「视频内时间」算停留/持续时长，而不是墙钟：
                # 跑录像时不按实时节拍，墙钟只走 1.5 秒，视频却过了 12 秒，
                # 用墙钟会导致 15 秒停留永远触发不了。
                now=video_time,
                video_time=video_time,
                demo=demo,
            )
            fwd.append(1.0 / max(1e-6, (time.perf_counter() - t_start) / max(1, kept)))

        cap.release()
        wall = time.perf_counter() - t_start
        fps = kept / wall if wall > 0 else 0.0

        return {
            "cam_id": cam_id,
            "frames": kept,
            "wall_sec": round(wall, 2),
            "fps": round(fps, 2),
            "avg_ms": round(float(np.mean(self.timings)) if self.timings else 0.0, 2),
            "reid_count": len(self._reid_done),
            "reid_stats": self.reid.stats(),
            "queue": self.dispatcher.stats() if self.dispatcher else None,
        }

    # ------------------------------------------------------------------ #
    def reset(self, cam_id: str | None = None) -> None:
        if cam_id is None:
            self._prev_state.clear()
            self._reid_done.clear()
            self.timings.clear()
            self.frame_idx = 0
            for e in self.engines.values():
                e.reset()
            self.recovery.reset(None)
            self.svc.reset(None)
        else:
            for k in [k for k in self._prev_state if k[0] == cam_id]:
                self._prev_state.pop(k, None)
            for k in [k for k in self._reid_done if k[0] == cam_id]:
                self._reid_done.pop(k, None)
            if cam_id in self.engines:
                self.engines[cam_id].reset()
            self.recovery.reset(cam_id)
            self.svc.reset(cam_id)

    def stats(self) -> dict:
        return {
            "frames": self.frame_idx,
            "reid_done": len(self._reid_done),
            "avg_ms": round(float(np.mean(self.timings)) if self.timings else 0.0, 2),
            "reid": self.reid.stats(),
            "queue": self.dispatcher.stats() if self.dispatcher else None,
        }
