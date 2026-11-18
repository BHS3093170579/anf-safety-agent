"""边缘端五大行为时序判定 + 业务互斥状态机。

五条业务规则（ 2.2）：
    1. 安全帽违规     5 帧滑窗内未戴帽 ≥3 帧才判违规（挡掉黄色背包/反光衣的单帧误判）
    2. 危险区域入侵   质心落入多边形 ROI，穿行 5s 内不报，停留 15s 报警
    3. 工友跌倒       躯干与竖直夹角 > 30°，持续 2s 降 medium，10s 升 critical
    4. 值班离岗       ROI 内无人累计 180 秒
    5. 值班睡岗       头部下沉比例超阈值 + 持续 60 秒（演示值 8 秒）

业务互斥（ 2.2 最后一条，也是最容易写错的一条）：
    判到「离岗」时必须**强制关闭**本机位的睡岗检测——人都不在，
    椅子上的衣物、挂着的反光背心会被当成「头趴在桌上」，必然误报。
    反过来，一旦 ROI 里重新有人，离岗计时清零、睡岗检测重新打开。

为什么用三段式状态机而不是布尔量：
    boolean 只能表达「是/否」，表达不了「已经不对劲但还没到报警线」。
    normal / suspected / abnormal 让上层可以先提醒、后告警，
    也方便答辩时解释「为什么不是一进去就报」。

与 V1.0 的关系：
    V1.0 的规则逻辑写死在 app.py 的 detect 循环里，一路摄像头一套。
    这里抽成纯函数式引擎：输入 tracks，输出判定，不碰画面也不碰模型。
    所以追踪结果可以直接喂进来，告警中心也能直接消费。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from utils.camera_memory_store import CameraMemoryStore

# COCO-17 里用到的关键点下标
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHO, R_SHO, L_HIP, R_HIP = 5, 6, 11, 12

NORMAL, SUSPECTED, ABNORMAL = "normal", "suspected", "abnormal"

# ============================================================== 数据结构
@dataclass
class RuleResult:
    """一条规则在某一帧的判定结果。"""

    rule: str
    state: str
    elapsed: float = 0.0
    reason: str = ""
    track_id: int | None = None
    suppressed: bool = False
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def is_alert(self) -> bool:
        """真正要送告警中心的结果：abnormal 且没被互斥规则压制。"""
        return self.state == ABNORMAL and not self.suppressed

    def to_dict(self) -> dict:
        d = {
            "rule": self.rule,
            "state": self.state,
            "elapsed": round(self.elapsed, 2),
            "reason": self.reason,
            "track_id": self.track_id,
            "suppressed": self.suppressed,
        }
        if self.detail:
            d["detail"] = self.detail
        return d

# ============================================================== 几何工具
def point_in_polygon(x: float, y: float, poly) -> bool:
    """射线法判断点是否在多边形内。

    既定用射线法。相比凸包/矩阵判定，射线法对任意凹多边形都成立——
    现场的「基坑边缘」「脚手架下方」经常是不规则形状，凹多边形很常见。
    """
    if not poly or len(poly) < 3:
        return False
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = float(poly[i][0]), float(poly[i][1])
        x2, y2 = float(poly[(i + 1) % n][0]), float(poly[(i + 1) % n][1])
        if (y1 > y) != (y2 > y):
            xin = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if x < xin:
                inside = not inside
    return inside

def scale_polygon(poly, w: float, h: float) -> list[list[float]]:
    """归一化 ROI → 像素坐标。配置里一律存 [0,1]，与分辨率无关。"""
    return [[float(p[0]) * w, float(p[1]) * h] for p in poly]

def _kpts(kp) -> np.ndarray | None:
    if kp is None:
        return None
    k = np.asarray(kp, dtype=np.float32)
    if k.ndim != 2 or k.shape[0] < 17:
        return None
    return k

def hip_center(kp, conf_thr: float = 0.2) -> tuple[float, float] | None:
    """左右髋中点 = 既定的入侵质心。看不到髋时退化到肩中点。"""
    k = _kpts(kp)
    if k is None:
        return None
    c = k[:, 2] if k.shape[1] >= 3 else None

    def ok(i: int) -> bool:
        return c is None or c[i] >= conf_thr

    if ok(L_HIP) and ok(R_HIP):
        return float((k[L_HIP, 0] + k[R_HIP, 0]) / 2), float((k[L_HIP, 1] + k[R_HIP, 1]) / 2)
    if ok(L_SHO) and ok(R_SHO):
        return float((k[L_SHO, 0] + k[R_SHO, 0]) / 2), float((k[L_SHO, 1] + k[R_SHO, 1]) / 2)
    return None

def shoulder_mid(kp, conf_thr: float = 0.2) -> tuple[float, float] | None:
    k = _kpts(kp)
    if k is None:
        return None
    c = k[:, 2] if k.shape[1] >= 3 else None

    def ok(i: int) -> bool:
        return c is None or c[i] >= conf_thr

    if ok(L_SHO) and ok(R_SHO):
        return float((k[L_SHO, 0] + k[R_SHO, 0]) / 2), float((k[L_SHO, 1] + k[R_SHO, 1]) / 2)
    return None

def torso_angle_deg(kp, conf_thr: float = 0.2) -> float | None:
    """躯干（肩中点→髋中点）与竖直方向的夹角，单位度。

    为什么不用框的长宽比判跌倒：
        人蹲下、弯腰搬东西时框也会变「扁」，长宽比误判率很高。
        躯干向量在站立时接近竖直（≈0°）、躺地时接近水平（≈90°），
        这个变化来自人体姿态本身，蹲下也不会横过来。
    """
    k = _kpts(kp)
    if k is None:
        return None
    c = k[:, 2] if k.shape[1] >= 3 else None

    def ok(i: int) -> bool:
        return c is None or c[i] >= conf_thr

    if not (ok(L_SHO) and ok(R_SHO) and ok(L_HIP) and ok(R_HIP)):
        return None
    nx = (k[L_SHO, 0] + k[R_SHO, 0]) / 2
    ny = (k[L_SHO, 1] + k[R_SHO, 1]) / 2
    hx = (k[L_HIP, 0] + k[R_HIP, 0]) / 2
    hy = (k[L_HIP, 1] + k[R_HIP, 1]) / 2
    dx, dy = abs(hx - nx), abs(hy - ny)
    if dx < 1e-6 and dy < 1e-6:
        return None
    return float(math.degrees(math.atan2(dx, dy)))

def head_drop_ratio(kp, conf_thr: float = 0.2) -> float | None:
    """头部下沉比 = (nose.y - shoulder_mid.y) / shoulder_width。

    与 V1.0 的 sleep 判定完全一致（rules.yaml 里 head_drop_threshold: -0.32）：
    比值越大表示头越低。用肩宽做分母是为了抵消远近尺度变化，
    远处的人和近处的人肩宽不同，直接比像素偏移会失真。
    """
    k = _kpts(kp)
    if k is None:
        return None
    c = k[:, 2] if k.shape[1] >= 3 else None

    def ok(i: int) -> bool:
        return c is None or c[i] >= conf_thr

    if not (ok(NOSE) and ok(L_SHO) and ok(R_SHO)):
        return None
    sx = (k[L_SHO, 0] + k[R_SHO, 0]) / 2
    sy = (k[L_SHO, 1] + k[R_SHO, 1]) / 2
    shoulder_w = float(abs(k[L_SHO, 0] - k[R_SHO, 0]))
    if shoulder_w < 4.0:
        return None
    return float((k[NOSE, 1] - sy) / shoulder_w)

def bbox_center(bbox) -> tuple[float, float]:
    return (float(bbox[0]) + float(bbox[2])) / 2, (float(bbox[1]) + float(bbox[3])) / 2

def bbox_bottom_center(bbox) -> tuple[float, float]:
    """脚底中点。要求入侵检测用这个点，避免上半身探入就误报。"""
    return (float(bbox[0]) + float(bbox[2])) / 2, float(bbox[3])

def clamp_point(x: float, y: float, shape: tuple[int, int] | None) -> tuple[float, float]:
    """把点收进画面内。

    为什么需要：人被画面底边截断时 bbox[3] 正好等于图像高度，
    而 ROI 坐标归一化后最大也就到 h（甚至 0.95h），这个点会永远落在
    ROI 外面——工地上人贴着镜头走是常态，不 clamp 会大面积漏判入侵。
    """
    if not shape:
        return x, y
    h, w = float(shape[0]), float(shape[1])
    return min(max(x, 0.0), w - 1.0), min(max(y, 0.0), h - 1.0)

# ============================================================== 引擎本体
class RulesEngine:
    """一路摄像头一个实例。持有 CameraMemoryStore，按帧推进所有规则。

    用法：
        eng = RulesEngine("Camera_01", cam_cfg, rules_cfg)
        results = eng.update(tracks, now=ts, frame_shape=(h, w))
        for r in results:
            if r.is_alert: ...
    """

    def __init__(
        self,
        cam_id: str,
        cam_cfg: dict | None = None,
        rules_cfg: dict | None = None,
        store: CameraMemoryStore | None = None,
    ) -> None:
        self.cam_id = cam_id
        self.cam_cfg = cam_cfg or {}
        self.rules_cfg = rules_cfg or {}
        self.store = store or CameraMemoryStore()

        rc = self.rules_cfg
        self.cfg_helmet = rc.get("helmet", {}) or {}
        self.cfg_intrude = rc.get("intrusion", {}) or {}
        self.cfg_fall = rc.get("fall", {}) or {}
        self.cfg_away = rc.get("away", {}) or {}
        self.cfg_sleep = rc.get("sleep", {}) or {}
        self.cfg_loiter = rc.get("loiter", {}) or {}
        self.cfg_link = rc.get("linkage", {}) or {}

        # ---- 规则 1：安全帽滑窗 ----
        # 要求 5 帧滑窗 / ≥3 帧未戴帽；rules.yaml 里的
        # min_consecutive_frames 是早期版本留下的连续帧，两者取较严的。
        self.helmet_window = int(self.cfg_helmet.get("window_frames", 5))
        self.helmet_abnormal = int(self.cfg_helmet.get("abnormal_frames", 3))
        self.helmet_suspect = int(self.cfg_helmet.get("suspect_frames", 2))
        self.helmet_min_consec = int(self.cfg_helmet.get("min_consecutive_frames", 2))

        # ---- 规则 2：入侵 ----
        self.intrude_dwell = float(self.cfg_intrude.get("dwell_sec", 15.0))
        self.intrude_transient = float(self.cfg_intrude.get("transient_sec", 5.0))
        self.use_bottom_center = bool(self.cfg_intrude.get("use_bottom_center", True))

        # ---- 规则 3：跌倒 ----
        self.fall_theta = float(self.cfg_fall.get("theta_deg", 30.0))
        self.fall_medium = float(self.cfg_fall.get("medium_after_sec", 2.0))
        self.fall_critical = float(self.cfg_fall.get("critical_after_sec", 10.0))

        # ---- 规则 4：离岗 ----
        self.away_sec = float(self.cfg_away.get("accumulate_sec", 180.0))
        self.away_min_px = float(self.cfg_away.get("min_person_px", 40.0))
        self.away_suspect = float(self.cfg_away.get("suspect_after_sec", self.away_sec / 3))

        # ---- 规则 5：睡岗 ----
        self.sleep_drop_thr = float(self.cfg_sleep.get("head_drop_threshold", -0.32))
        self.sleep_direction = str(self.cfg_sleep.get("head_drop_direction", "greater"))
        self.sleep_frames = int(self.cfg_sleep.get("min_consecutive_frames", 300))
        self.sleep_demo_frames = int(self.cfg_sleep.get("demo_min_consecutive_frames", 40))
        self.sleep_suspend_when_away = bool(self.cfg_sleep.get("suspend_when_away", True))
        # 一个实例按 fps 换算秒数，默认 5FPS（与 system.yaml stream.target_fps 一致）
        self.sleep_fps = float(self.cfg_sleep.get("fps_baseline", 5.0))
        self.sleep_after_s = self.sleep_frames / self.sleep_fps
        self.sleep_demo_s = self.sleep_demo_frames / self.sleep_fps

        # ---- 规则 6：徘徊 ----
        self.loiter_sec = float(self.cfg_loiter.get("loiter_sec", 20.0))
        self.loiter_suspect = float(self.cfg_loiter.get("suspect_sec", 8.0))
        self.loiter_reentry = int(self.cfg_loiter.get("reentry_thresh", 2))
        self.loiter_child_ratio = float(self.cfg_loiter.get("child_ratio", 0.35))
        self.loiter_child_weight = float(self.cfg_loiter.get("child_weight", 0.2))
        self.loiter_min_px = float(self.cfg_loiter.get("min_person_px", 40.0))
        self.loiter_demo = float(self.cfg_loiter.get("demo_loiter_sec", 5.0))
        self.loiter_suspect_demo = float(self.cfg_loiter.get("demo_suspect_sec", 2.0))
        self._loiter: dict[int, dict] = {}

        # ---- ROI：从 cameras.yaml 取，按 purpose 区分用途 ----
        rois = self.cam_cfg.get("roi") or []
        self.intrude_rois: list[dict] = []
        self.away_rois: list[dict] = []
        self.loiter_rois: list[dict] = []
        for r in rois:
            if not isinstance(r, dict):
                continue
            purpose = str(r.get("purpose", "")).lower()
            if purpose == "away":
                self.away_rois.append(r)
            elif purpose == "loiter":
                self.loiter_rois.append(r)
            else:
                self.intrude_rois.append(r)
        self._scaled = False

        # ---- 互斥状态 ----
        self.sleep_detect_enabled = True
        self.away_active = False
        self._away_since: float | None = None
        self._sleep_run: dict[int, int] = {}      # track_id -> 连续命中帧数
        self._scaled_polys: dict[str, list[list[list[float]]]] = {}
        self._frame_shape: tuple[int, int] | None = None

        self.last_results: dict[str, RuleResult] = {}
        self.events: list[dict] = []
        self.frames = 0

    # ---------------------------------------------------------- 主入口
    def update(
        self,
        tracks: list[dict],
        now: float | None = None,
        frame_shape: tuple[int, int] | None = None,
        demo: bool = False,
    ) -> list[RuleResult]:
        """推进一帧，返回本轮所有规则的判定。

        Args:
            tracks: [{track_id, bbox, keypoints?, has_helmet?}, ...]
            now: 时间戳（测试可注入固定值，生产用 time.time()）
            frame_shape: (h, w)，ROI 归一化坐标按它换算成像素
            demo: True 时睡岗用演示值（8 秒）而不是正式值（60 秒）
        """
        now = time.time() if now is None else now
        self.frames += 1
        if frame_shape is not None:
            self._prepare_polygons(frame_shape)

        results: list[RuleResult] = []

        # 只跑这台相机 scene 里声明的规则；离岗必须最先跑，它决定 sleep_detect_enabled
        scenes = set(self.cam_cfg.get("scene") or [])
        if not scenes:                     # 没写 scene 时退回全跑（兼容老配置）
            scenes = {"away", "sleep", "intrusion", "fall", "loiter", "helmet"}

        if "away" in scenes:
            results.extend(self._rule_away(tracks, now))
        if "sleep" in scenes:
            results.extend(self._rule_sleep(tracks, now, demo))
        if "intrusion" in scenes:
            results.extend(self._rule_intrusion(tracks, now))
        if "fall" in scenes:
            results.extend(self._rule_fall(tracks, now))
        if "loiter" in scenes:
            results.extend(self._rule_loiter(tracks, now, demo))
        if "helmet" in scenes and tracks and any("has_helmet" in t for t in tracks):
            results.extend(self._rule_helmet(tracks, now))

        for r in results:
            self.last_results[r.rule] = r
        return results

    # ---------------------------------------------------------- 多边形准备
    def _prepare_polygons(self, frame_shape: tuple[int, int]) -> None:
        h, w = float(frame_shape[0]), float(frame_shape[1])
        self._frame_shape = (int(h), int(w))
        if self._scaled and self._scaled_polys:
            return
        self._scaled_polys = {
            "intrusion": [scale_polygon(r["polygon"], w, h) for r in self.intrude_rois if r.get("polygon")],
            "away": [scale_polygon(r["polygon"], w, h) for r in self.away_rois if r.get("polygon")],
            "loiter": [scale_polygon(r["polygon"], w, h) for r in self.loiter_rois if r.get("polygon")],
        }
        self._scaled = True

    def _polygons(self, kind: str) -> list[list[list[float]]]:
        return self._scaled_polys.get(kind, [])

    # ---------------------------------------------------------- 规则 4：离岗
    def _rule_away(self, tracks: list[dict], now: float) -> list[RuleResult]:
        """ROI 内无人累计 180 秒 → 离岗，同时强制关闭睡岗检测。"""
        polys = self._polygons("away") or self._polygons("intrusion")
        polys = [p for p in polys if len(p) >= 3]

        present = 0
        for t in tracks:
            bbox = t.get("bbox")
            if not bbox:
                continue
            # 滤掉远处路人：框太矮的不算在岗
            if float(bbox[3]) - float(bbox[1]) < self.away_min_px:
                continue
            c = hip_center(t.get("keypoints")) or bbox_center(bbox)
            if not polys or any(point_in_polygon(c[0], c[1], p) for p in polys):
                present += 1

        if present > 0:
            self._away_since = None
            was = self.away_active
            self.away_active = False
            self.sleep_detect_enabled = True
            if was:
                self.events.append({"type": "away_cleared", "cam": self.cam_id, "ts": now})
            return [RuleResult("away", NORMAL, 0.0, f"在岗 {present} 人",
                               detail={"in_roi": present})]

        if self._away_since is None:
            self._away_since = now
        gone = now - self._away_since

        if gone >= self.away_sec:
            state, reason = ABNORMAL, f"无人 {gone:.0f}s"
            if not self.away_active:
                self.away_active = True
                # ← 业务互斥：离岗期间必须关掉睡岗，否则椅子上的衣服会误报
                if self.sleep_suspend_when_away and self.cfg_link.get("away_suspends_sleep", True):
                    self.sleep_detect_enabled = False
                self.events.append({"type": "away_alert", "cam": self.cam_id,
                                    "ts": now, "elapsed": gone})
        elif gone >= self.away_suspect:
            state, reason = SUSPECTED, f"无人 {gone:.0f}s"
        else:
            state, reason = NORMAL, f"无人 {gone:.0f}s"

        return [RuleResult("away", state, gone, reason,
                           detail={"in_roi": 0, "sleep_detect_enabled": self.sleep_detect_enabled})]

    # ---------------------------------------------------------- 规则 5：睡岗
    def _rule_sleep(self, tracks: list[dict], now: float, demo: bool = False) -> list[RuleResult]:
        """头部下沉 + 连续命中 → 睡岗。

        用 V1.0 已验证的 head_drop 指标：(nose.y - shoulder_mid.y) / shoulder_width。
        比值越大 = 头越低。threshold -0.32 / direction greater 在 rules.yaml 里写死，
        与文档口径一致，不在这里改判据。
        """
        if not self.sleep_detect_enabled:
            return [RuleResult("sleep", NORMAL, 0.0, "离岗期间强制抑制睡岗", suppressed=True)]

        polys = self._polygons("away")
        polys = [p for p in polys if len(p) >= 3]

        need_frames = self.sleep_demo_frames if demo else self.sleep_frames
        after_s = self.sleep_demo_s if demo else self.sleep_after_s

        hit_any = False
        best: RuleResult | None = None

        for t in tracks:
            tid = int(t.get("track_id", -1))
            bbox = t.get("bbox") or [0, 0, 0, 0]
            if polys:
                c = hip_center(t.get("keypoints")) or bbox_center(bbox)
                if not any(point_in_polygon(c[0], c[1], p) for p in polys):
                    self._sleep_run.pop(tid, None)
                    continue

            ratio = head_drop_ratio(t.get("keypoints"))
            if ratio is None:
                self._sleep_run.pop(tid, None)
                continue

            if self.sleep_direction == "greater":
                hit = ratio > self.sleep_drop_thr
            else:
                hit = ratio < self.sleep_drop_thr

            if hit:
                hit_any = True
                run = self._sleep_run.get(tid, 0) + 1
                self._sleep_run[tid] = run
            else:
                run = 0
                self._sleep_run[tid] = 0

            st = self.store.touch(self.cam_id, tid, bbox, now)
            start = st.violation_timers.setdefault("sleep", now)
            if run == 0:
                st.violation_timers.pop("sleep", None)
                start = now
            hold = now - start

            if run >= need_frames:
                state = ABNORMAL
                reason = f"头部下沉 {run} 帧（阈值 {self.sleep_drop_thr}），持续 {hold:.0f}s"
            elif run >= max(1, need_frames // 3):
                state = SUSPECTED
                reason = f"头部下沉 {run}/{need_frames} 帧"
            else:
                state = NORMAL
                reason = ""

            st.states["sleep"] = state
            t["sleep_state"] = state          # 回写给画框层，人体框按它上色
            r = RuleResult("sleep", state, hold, reason, track_id=tid,
                           detail={"head_drop": round(ratio, 3), "frames": run,
                                   "need_frames": need_frames})
            if best is None or (r.state != NORMAL and best.state == NORMAL):
                best = r

        if not hit_any:
            return [RuleResult("sleep", NORMAL, 0.0, "头部高度正常",
                               detail={"need_frames": need_frames})]
        return [best] if best else [RuleResult("sleep", NORMAL, 0.0, "")]

    # ---------------------------------------------------------- 规则 2：入侵
    def _rule_intrusion(self, tracks: list[dict], now: float) -> list[RuleResult]:
        """质心落入 ROI：穿行期（<5s）不报，5s 起 suspected，15s 起 abnormal。"""
        polys = [p for p in self._polygons("intrusion") if len(p) >= 3]
        if not polys:
            return []

        out: list[RuleResult] = []
        for t in tracks:
            tid = int(t.get("track_id", -1))
            bbox = t.get("bbox")
            if not bbox:
                continue
            if self.use_bottom_center:
                c = clamp_point(*bbox_bottom_center(bbox), self._frame_shape)
            else:
                c = hip_center(t.get("keypoints")) or clamp_point(*bbox_center(bbox), self._frame_shape)

            inside = any(point_in_polygon(c[0], c[1], p) for p in polys)
            st = self.store.touch(self.cam_id, tid, bbox, now)

            if not inside:
                st.violation_timers.pop("intrusion", None)
                st.states["intrusion"] = NORMAL
                out.append(RuleResult("intrusion", NORMAL, 0.0, "不在 ROI 内", track_id=tid))
                continue

            start = st.violation_timers.setdefault("intrusion", now)
            dwell = now - start
            if dwell >= self.intrude_dwell:
                state, reason = ABNORMAL, f"ROI 内停留 {dwell:.1f}s"
            elif dwell >= self.intrude_transient:
                state, reason = SUSPECTED, f"ROI 内停留 {dwell:.1f}s（超过穿行期）"
            else:
                state, reason = NORMAL, f"ROI 内穿行 {dwell:.1f}s"
            st.states["intrusion"] = state
            out.append(RuleResult("intrusion", state, dwell, reason, track_id=tid,
                                  detail={"center": [round(c[0], 1), round(c[1], 1)]}))
        return out

    # ---------------------------------------------------------- 规则 3：跌倒
    def _rule_fall(self, tracks: list[dict], now: float) -> list[RuleResult]:
        """躯干与竖直夹角 > theta_deg：2s 降 medium，10s 升 critical。"""
        out: list[RuleResult] = []
        for t in tracks:
            tid = int(t.get("track_id", -1))
            bbox = t.get("bbox") or [0, 0, 0, 0]
            st = self.store.touch(self.cam_id, tid, bbox, now)

            theta = torso_angle_deg(t.get("keypoints"))
            lying = theta is not None and theta > self.fall_theta

            if not lying:
                st.violation_timers.pop("fall", None)
                st.states["fall"] = NORMAL
                out.append(RuleResult("fall", NORMAL, 0.0, "", track_id=tid,
                                      detail={"theta": round(theta, 1) if theta is not None else None}))
                continue

            start = st.violation_timers.setdefault("fall", now)
            hold = now - start
            if hold >= self.fall_critical:
                state, reason = ABNORMAL, f"倒地 {hold:.1f}s（躯干 {theta:.0f}°）"
            elif hold >= self.fall_medium:
                state, reason = SUSPECTED, f"倒地 {hold:.1f}s（躯干 {theta:.0f}°）"
            else:
                state, reason = NORMAL, f"姿态异常 {hold:.1f}s"
            st.states["fall"] = state
            out.append(RuleResult("fall", state, hold, reason, track_id=tid,
                                  detail={"theta": round(theta, 1)}))
        return out

    # ---------------------------------------------------------- 规则 6：徘徊
    def _rule_loiter(self, tracks: list[dict], now: float, demo: bool = False) -> list[RuleResult]:
        """ROI 内停留超时 或 折返次数达标 → 徘徊。

        与跨镜 Re-ID 不是一回事：这里是单路「同一目标在警戒区里
        反复进出/长时间逗留」。三个防误报手段：
          1. 只算 ROI 内的停留，ROI 外路过不计；
          2. 小孩（框高 < child_ratio × 画面高）按 child_weight 衰减；
          3. 折返要求「先离开 ROI 再回来」，原地抖动不算。
        """
        polys = self._polygons("loiter") or self._polygons("intrusion")
        polys = [p for p in polys if len(p) >= 3]

        need_sec = self.loiter_demo if demo else self.loiter_sec
        suspect_sec = self.loiter_suspect_demo if demo else self.loiter_suspect

        frame_h = float(self._frame_shape[0]) if self._frame_shape else 720.0
        out: list[RuleResult] = []

        for t in tracks:
            tid = int(t.get("track_id", -1))
            bbox = t.get("bbox") or [0, 0, 0, 0]
            h = float(bbox[3]) - float(bbox[1])
            if h < self.loiter_min_px:
                continue

            c = bbox_center(bbox)
            in_roi = (not polys) or any(point_in_polygon(c[0], c[1], p) for p in polys)
            weight = self.loiter_child_weight if h < self.loiter_child_ratio * frame_h else 1.0

            st = self._loiter.get(tid)
            if st is None:
                st = {"sec": 0.0, "last": now, "in_roi": in_roi,
                      "reentries": 0.0, "entered": now}
                self._loiter[tid] = st

            dt = max(0.0, now - st["last"])
            st["last"] = now
            if in_roi:
                if dt > 2.0:
                    # 消失超过 2 秒又出现 = 一次折返；空档时间不算停留
                    st["reentries"] += weight
                    st["entered"] = now
                elif not st["in_roi"]:
                    # 从 ROI 外回来 = 一次折返
                    st["reentries"] += weight
                    st["entered"] = now
                    st["sec"] += dt * weight
                else:
                    st["sec"] += dt * weight
            st["in_roi"] = in_roi

            if not in_roi:
                continue

            hold = now - st["entered"]
            if st["sec"] >= need_sec or st["reentries"] >= self.loiter_reentry:
                state = ABNORMAL
                reason = f"停留 {st['sec']:.0f}s、折返 {st['reentries']:.0f} 次"
            elif st["sec"] >= suspect_sec:
                state = SUSPECTED
                reason = f"停留 {st['sec']:.0f}s（阈值 {need_sec:.0f}s）"
            else:
                state = NORMAL
                reason = ""

            t["loiter_state"] = state         # 回写给画框层
            t["loiter_sec"] = round(st["sec"], 1)
            out.append(RuleResult("loiter", state, st["sec"], reason, track_id=tid,
                                  detail={"loiter_sec": round(st["sec"], 1),
                                          "reentries": round(st["reentries"], 1),
                                          "weight": weight,
                                          "hold": round(hold, 1)}))

        # 清理长时间没再出现的轨迹，避免字典无限增长
        stale = [k for k, v in self._loiter.items() if now - v["last"] > 600]
        for k in stale:
            self._loiter.pop(k, None)
        return out

    # ---------------------------------------------------------- 规则 1：安全帽
    def _rule_helmet(self, tracks: list[dict], now: float) -> list[RuleResult]:
        """5 帧滑窗，未戴帽帧数 ≥3 才 abnormal。

        要求挡掉「黄色背包 / 反光衣头帽」造成的单帧误判——
        这类误检的共同特点是只闪一两帧，滑窗投票天然能滤掉。
        """
        out: list[RuleResult] = []
        for t in tracks:
            tid = int(t.get("track_id", -1))
            has = bool(t.get("has_helmet", True))
            st = self.store.touch(self.cam_id, tid, t.get("bbox"), now)

            win: list[int] = st.counters.setdefault("helmet_window", [])
            win.append(0 if has else 1)
            if len(win) > self.helmet_window:
                del win[: len(win) - self.helmet_window]
            bad = sum(win)

            if bad >= self.helmet_abnormal and len(win) >= self.helmet_abnormal:
                state = ABNORMAL
                reason = f"滑窗 {len(win)} 帧内 {bad} 帧未戴帽"
            elif bad >= self.helmet_suspect:
                state = SUSPECTED
                reason = f"滑窗 {len(win)} 帧内 {bad} 帧未戴帽"
            else:
                state = NORMAL
                reason = ""

            if state == NORMAL:
                st.violation_timers.pop("helmet", None)
                hold = 0.0
            else:
                start = st.violation_timers.setdefault("helmet", now)
                hold = now - start
                # 兼容连续帧口径
                if st.counters.get("helmet_consec", 0) < self.helmet_min_consec and bad < self.helmet_abnormal:
                    state = SUSPECTED

            st.counters["helmet_consec"] = 0 if has else st.counters.get("helmet_consec", 0) + 1
            st.states["helmet"] = state
            out.append(RuleResult("helmet", state, hold, reason, track_id=tid,
                                  detail={"bad_frames": bad, "window": len(win)}))
        return out

    # ---------------------------------------------------------- 辅助
    def drain_events(self) -> list[dict]:
        ev, self.events = self.events, []
        return ev

    def reset(self) -> None:
        self.store.reset(self.cam_id)
        self._away_since = None
        self.away_active = False
        self.sleep_detect_enabled = True
        self._sleep_run.clear()
        self.last_results.clear()
        self.frames = 0

    def stats(self) -> dict:
        return {
            "cam_id": self.cam_id,
            "frames": self.frames,
            "sleep_detect_enabled": self.sleep_detect_enabled,
            "away_active": self.away_active,
            "intrude_rois": len(self.intrude_rois),
            "away_rois": len(self.away_rois),
            "last_states": {k: v.state for k, v in self.last_results.items()},
            "store": self.store.stats(),
        }
