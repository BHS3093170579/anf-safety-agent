"""
V2.0 边缘流水线面板 —— 把各能力模块真正接到界面上。

为什么需要这个文件：
    edge/model_service.py（ModelPool + 人体 BBox 重构 + ByteTrack）、
    edge/rules_engine.py（五大规则 + 离岗睡岗互斥）、
    edge/pipeline.py（按需 Re-ID + 事件分发）、
    edge/event_dispatcher.py（异步上报）、
    utils/track_recovery.py（遮挡 ID 恢复）、
    utils/reid_extractor.py（OSNet 512 维特征）

    这六个模块写完后只在 test_ticketXX.py 里被调用，主界面一直是 V1.0
    的单路检测页，从界面完全看不出 V2.0 做了什么。本文件补上这个入口。

和 V1.0 标签页的分工：
    「实时监控」  —— V1.0：YOLO 检测 + 告警 + 报告（按类别置信度）
    「规则引擎」—— Pose 单模型 + ByteTrack + 五条规则时序判定
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent

# 五条规则的中文名，和 rules.yaml 的键一一对应
RULE_CN = {
    "helmet": "安全帽",
    "intrusion": "危险区入侵",
    "fall": "工友跌倒",
    "away": "值班离岗",
    "sleep": "值班睡岗",
}
STATE_CN = {"normal": "正常", "suspected": "疑似", "abnormal": "告警"}
STATE_COLOR = {"normal": (110, 200, 110), "suspected": (60, 190, 250), "abnormal": (70, 70, 245)}

# ---------------------------------------------------------------- 中文绘制
# cv2.putText 不支持中文，直接画会变成一串问号。
# 这里用 PIL 渲染，但只渲染一次：结果按 (文本, 颜色, 字号) 缓存成小图，
# 之后每帧只做一次 numpy 覆盖，所以对角色的实时性没有影响。
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",     # 微软雅黑
    r"C:\Windows\Fonts\simhei.ttf",   # 黑体
    r"C:\Windows\Fonts\Deng.ttf",     # 等线
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]
_font_cache: dict = {}

def _font(size: int):
    """按字号取字体对象，找不到中文字体就退回默认（英文仍可读）。"""
    from PIL import ImageFont

    if size in _font_cache:
        return _font_cache[size]
    for path in _FONT_CANDIDATES:
        try:
            _font_cache[size] = ImageFont.truetype(path, size)
            return _font_cache[size]
        except Exception:                                  # noqa: BLE001
            continue
    _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]

@lru_cache(maxsize=512)
def _text_tile(text: str, bgr: tuple, size: int):
    """把一行中文渲染成 (彩色图, 透明度蒙版)，按参数缓存。"""
    from PIL import Image, ImageDraw

    font = _font(size)
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    l, t, r, b = probe.textbbox((0, 0), text, font=font)
    w, h = max(1, r - l), max(1, b - t)
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).text((-l, -t), text, font=font, fill=255)
    m = np.array(mask, dtype=np.uint8)
    tile = np.empty((h, w, 3), dtype=np.uint8)
    tile[:] = bgr
    return tile, m

def _put_cn(vis: np.ndarray, text: str, x: int, y: int,
            bgr: tuple, size: int = 19) -> None:
    """在 vis 上贴一行中文（带透明度），坐标越界自动裁剪。"""
    if not text:
        return
    tile, mask = _text_tile(str(text), tuple(int(c) for c in bgr), int(size))
    h, w = mask.shape
    H, W = vis.shape[:2]
    x, y = int(x), int(y)
    if x >= W or y >= H or x + w <= 0 or y + h <= 0:
        return
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    sx, sy = x0 - x, y0 - y
    m = (mask[sy:sy + (y1 - y0), sx:sx + (x1 - x0)].astype(np.float32) / 255.0)[..., None]
    region = vis[y0:y1, x0:x1]
    region[:] = (tile[sy:sy + (y1 - y0), sx:sx + (x1 - x0)].astype(np.float32) * m
                 + region.astype(np.float32) * (1.0 - m)).astype(np.uint8)

# 演示用阈值：正式值（15s 入侵 / 10s 跌倒 / 180s 离岗 / 60s 睡岗）
# 在几十秒的测试视频里等不到，所以界面默认用演示值，可勾选切回正式值。
DEMO_OVERRIDE = {
    "intrusion": {"dwell_sec": 1.5, "transient_sec": 0.6},
    "fall": {"medium_after_sec": 0.8, "critical_after_sec": 2.0},
    "away": {"accumulate_sec": 3.0, "suspect_after_sec": 1.5},
    "sleep": {"min_consecutive_frames": 10, "demo_min_consecutive_frames": 10},
    "loiter": {"loiter_sec": 5.0, "suspect_sec": 2.0},
}

def _load_yaml():
    # 走 config_loader 才会展开 ${CAM01_URL:...} 这类占位，直接 yaml.safe_load
    # 拿到的是字面量，前端就看不到现场 RTSP 地址了。
    from config_loader import load_all

    data = load_all()
    return data["cameras"], data["rules"]

@st.cache_resource(show_spinner=False)
def _get_dispatcher():
    """全局一个上报调度器：JSONL 落盘 + 可选 Webhook。"""
    from config_loader import load_all
    from edge.event_dispatcher import EventDispatcher
    from edge.notifier import make_post_fn

    alert = (load_all().get("system") or {}).get("alert") or {}
    hook = str(alert.get("webhook_url") or "")
    disp = EventDispatcher(
        endpoint="local://edge-alerts",
        post_fn=make_post_fn(ROOT / "outputs" / "events" / "edge_alerts.jsonl", hook),
        timeout=3.0,
    )
    return disp.start()

# 多路页「检测能力」下拉可覆盖相机自带的 scene
CAP_SCENES = {
    "按相机配置": None,
    "安全帽检测": ["helmet"],
    "火灾检测": ["fire"],
    "安全帽 + 火灾": ["helmet", "fire"],
    "睡岗 / 离岗": ["away", "sleep"],
    "徘徊检测": ["loiter"],
}

@st.cache_resource(show_spinner=False)
def _get_pipeline(cam_id: str, video: str, demo: bool,
                  capability: str = "按相机配置"):
    """按 (摄像头, 视频, 是否演示阈值, 能力) 缓存一条流水线。

    用 cache_resource 是因为 ModelPool 是进程级单例，
    每次重建会重新加载 4 秒，切标签页也会卡。
    """
    from edge.model_service import ModelService
    from edge.pipeline import EdgePipeline

    cams, rules = _load_yaml()
    if demo:
        for k, v in DEMO_OVERRIDE.items():
            if k in rules and isinstance(rules[k], dict):
                rules[k].update(v)

    cam_cfg = dict(cams["cameras"][cam_id])
    # 只给这一条流水线这一路相机，避免把别的相机的引擎也建起来
    # 睡岗场景放宽宽高比：人趴在桌上只有头肩可见，框是宽大于高的
    override = CAP_SCENES.get(capability)
    if override:
        cam_cfg["scene"] = list(override)
    scenes = cam_cfg.get("scene") or []
    svc = ModelService(conf=0.35, kpt_conf=0.25, imgsz=640,
                       lying_ok="sleep" in scenes)

    # 按相机 scene 叠加安全帽 / 火灾两路检测器（各自独立模型，共用模型池）
    from agents.detector import DetectionAgent
    from config_loader import load_all

    scene_cfgs = load_all()["scenes"]["scenes"]
    detectors: dict = {}
    if "helmet" in scenes and "helmet" in scene_cfgs:
        cfg = dict(scene_cfgs["helmet"])
        cfg.setdefault("name", "helmet")
        detectors["helmet"] = DetectionAgent(cfg)
    if "fire" in scenes and "fire" in scene_cfgs:
        cfg = dict(scene_cfgs["fire"])
        cfg.setdefault("name", "fire")
        detectors["fire"] = DetectionAgent(cfg)

    pipe = EdgePipeline(
        cams={cam_id: cam_cfg},
        rules=rules,
        model_service=svc,
        detectors=detectors,
        dispatcher=_get_dispatcher(),
        save_snapshot=True,
        verbose=False,
    )
    return pipe, cam_cfg

def _draw(frame: np.ndarray, rep, cam_cfg: dict) -> np.ndarray:
    """画框 + ID + 左上角规则状态。

    ROI 多边形不画在画面上——它只是规则判定用的区域，
    画出来会让人误以为是告警框。
    """
    vis = frame.copy()
    h, w = frame.shape[:2]

    # ROI 多边形（默认不画，保留代码方便调试时打开）
    # for roi in cam_cfg.get("roi") or []:
    #     ...

    # 人体框 + track_id + 状态标签
    # 颜色约定：绿=戴帽 / 红=未戴帽、睡岗 / 橙=徘徊、疑似 / 灰=未判定
    C_OK = (60, 180, 75)
    C_BAD = (40, 40, 220)
    C_WARN = (0, 140, 255)
    C_IDLE = (150, 150, 150)

    for t in getattr(rep, "tracks", []) or []:
        bb = t.get("bbox")
        if not bb:
            continue
        x1, y1, x2, y2 = [int(v) for v in bb]

        tags: list[str] = []
        color = C_IDLE
        hh = t.get("has_helmet")
        if hh is True:
            tags.append("安全帽")
            color = C_OK
        elif hh is False:
            tags.append("未戴安全帽")
            color = C_BAD

        s_sleep = t.get("sleep_state")
        if s_sleep == "abnormal":
            tags.insert(0, "睡岗")
            color = C_BAD
        elif s_sleep == "suspected":
            tags.insert(0, "疑似睡岗")
            color = C_WARN

        s_loiter = t.get("loiter_state")
        if s_loiter == "abnormal":
            tags.append("徘徊")
            color = C_WARN if color != C_BAD else color
        elif s_loiter == "suspected":
            tags.append("疑似徘徊")
            if color == C_IDLE:
                color = C_WARN

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        head = f"ID{t.get('track_id')}"
        if tags:
            head += " " + " ".join(tags)
        # 标签底色：深色半透明不做了，直接圆角实底，投影也看得清
        tw = max(52, 9 * len(head))
        cv2.rectangle(vis, (x1, max(0, y1 - 22)), (x1 + tw, max(18, y1 - 2)),
                      color, -1)
        _put_cn(vis, head, x1 + 4, max(0, y1 - 21), (255, 255, 255), 15)

    # 状态面板
    order = ["helmet", "intrusion", "fall", "away", "sleep"]
    by_rule = {r.rule: r for r in rep.rule_results}
    lines = []
    for rule in order:
        r = by_rule.get(rule)
        if r is None:
            continue
        lines.append((rule, r))
    if lines:
        # 面板放大到投影可读：宽度 470、行高 34、中文实绘
        panel_w, header_h, row_h = 470, 38, 34
        box_h = header_h + row_h * len(lines) + 10
        cv2.rectangle(vis, (0, 0), (panel_w, box_h), (18, 18, 18), -1)
        cv2.line(vis, (panel_w, 0), (panel_w, box_h), (65, 65, 65), 1)
        cv2.line(vis, (0, box_h), (panel_w, box_h), (65, 65, 65), 1)

        # 表头（英文用 cv2 画更快，且不会糊）
        cv2.putText(vis, f"{rep.cam_id}  bodies={rep.bodies}  rec={rep.recovered}",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 190, 190), 1, cv2.LINE_AA)
        cv2.line(vis, (0, header_h), (panel_w, header_h), (55, 55, 55), 1)

        y = header_h + 4
        for rule, r in lines:
            c = STATE_COLOR.get(r.state, (180, 180, 180))
            # 状态色点
            cv2.circle(vis, (20, y + 15), 6, c, -1, cv2.LINE_AA)
            # 规则名（中文）
            _put_cn(vis, RULE_CN.get(rule, rule), 36, y + 4, (238, 238, 238), 19)
            # 状态（中文，告警时加粗感：画两遍错位 1px）
            _put_cn(vis, STATE_CN.get(r.state, r.state), 152, y + 4, c, 19)
            # 原因（小一号、灰一点，避免抢视觉）
            _put_cn(vis, str(r.reason)[:13], 246, y + 7, (165, 165, 165), 16)
            y += row_h
    return vis

def render_edge_pipeline_tab() -> None:
    """规则引擎标签页主体。"""
    cams, _ = _load_yaml()
    cam_ids = list(cams["cameras"].keys())
    fallback = cams.get("fallback") or {}

    c1, c2, c3 = st.columns([1.1, 1.6, 1.2])
    with c1:
        cam_id = st.selectbox(
            "摄像头", cam_ids,
            format_func=lambda k: f"{k} · {cams['cameras'][k].get('name', '')}",
            key="edge_cam",
        )
    with c2:
        videos = sorted((ROOT / "datasets" / "videos").glob("*/*.mp4"))
        # 该摄像头的默认视频排在最前
        default_v = str((ROOT / fallback.get(cam_id, "")).resolve()) if fallback.get(cam_id) else ""
        opts = [str(v.resolve()) for v in videos]
        if default_v in opts:
            opts.remove(default_v)
            opts.insert(0, default_v)
        video = st.selectbox("视频源", opts, format_func=lambda p: f"{Path(p).parent.name}/{Path(p).name}",
                             key="edge_video")
    with c3:
        demo = st.checkbox("演示阈值（秒级触发）", value=True, key="edge_demo",
                           help="关掉后按正式值：入侵15s / 跌倒10s / 离岗180s / 睡岗60s")
        max_frames = st.number_input("最多帧数（0=全部）", 0, 3000, 200, 50, key="edge_max")

    b1, b2, b3 = st.columns([1, 1, 3])
    with b1:
        run = st.button("▶ 启动流水线", type="primary", key="edge_run")
    with b2:
        if st.button("↺ 重置状态", key="edge_reset"):
            _get_pipeline.clear()
            st.success("已清空流水线与追踪状态")

    if not run:
        st.markdown(
            '<div class="empty-state"><div class="big">🧩</div>'
            '<b>规则引擎未启动</b><br><br>'
            '选择摄像头与视频后点「启动流水线」，可以看到五条规则的实时状态</div>',
            unsafe_allow_html=True,
        )
        return

    try:
        pipe, cam_cfg = _get_pipeline(cam_id, video, demo)
    except Exception as e:  # noqa: BLE001
        st.error(f"流水线初始化失败：{type(e).__name__}: {e}")
        return

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        st.error(f"打不开视频：{video}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    limit = int(max_frames) if max_frames else total
    st.caption(f"共 {total} 帧，本次最多处理 {limit} 帧")

    ph_img = st.empty()
    ph_prog = st.progress(0.0)
    ph_rule = st.empty()

    idx = 0
    all_alerts: list[dict] = []
    last_rep = None
    while idx < limit:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        rep = pipe.process(cam_id, frame, video_time=idx / 24.0, demo=demo)
        # EdgePipeline 把 tracks 存在 svc 侧的 CamResult 里，这里补一份给绘制用
        last_rep = rep
        if getattr(rep, "alerts", None):
            all_alerts.extend(rep.alerts)

        if idx % 3 == 1 or idx == limit:
            vis = _draw(frame, rep, cam_cfg)
            ph_img.image(vis, channels="BGR", width="stretch")
            ph_prog.progress(min(idx / max(limit, 1), 1.0))

    cap.release()

    st.success(f"处理完成：{idx} 帧，产生告警 {len(all_alerts)} 条")

    if last_rep is not None:
        rows = [
            {
                "规则": RULE_CN.get(r.rule, r.rule),
                "状态": STATE_CN.get(r.state, r.state),
                "依据": r.reason or "-",
                "track": getattr(r, "track_id", "-"),
            }
            for r in last_rep.rule_results
        ]
        st.markdown("### 最后一帧的规则状态")
        ph_rule.table(rows)

    if all_alerts:
        st.markdown("### 告警事件")
        st.dataframe(all_alerts, width="stretch")

    st.caption("流水线统计：")
    st.json(pipe.stats())
