# -*- coding: utf-8 -*-
"""
安防智能体 V1.0 · Streamlit Web 界面
工单编号：CV专高5-11（工单 03 - 任务 7）

功能对应工单要求：
    实时监控画面（VideoCapture）—— 「实时监控」标签页，逐帧显示检测框与姿态骨架
    告警列表（AlertList）       —— 「告警列表」标签页，带等级徽章与证据截图
    报告查看（ReportViewer）    —— 「报告查看」标签页，可在线阅读与下载

启动：
    E:\\ANACONDA\\envs\\smoke_trt\\python.exe -m streamlit run app.py
    或双击 启动界面.bat

环境说明：
    .engine 权重绑定 TensorRT 10.16，必须用 smoke_trt 环境启动。
"""

from __future__ import annotations

import sys
import time
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config_loader import get_output_dirs, get_scene, list_scenes, load_config  # noqa: E402
from agents.logger import get_logger  # noqa: E402

_LOG = get_logger(__name__)

# ============================================================
# 页面配置与主题
# ============================================================
st.set_page_config(
    page_title="安防智能体 V1.0",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* ============================================================
   安防智能体 · 浅色专业后台
   取向：白底、柔和阴影、单一蓝强调；按钮做成清晰的实心主操作
   ============================================================ */
.block-container { padding-top: 1.4rem; padding-bottom: 2.6rem; max-width: 1600px; }

/* ---------- 顶部标题栏 ---------- */
.hero {
    background: #ffffff;
    border: 1px solid #e3e8ee;
    border-left: 4px solid #0969da;
    border-radius: 12px; padding: 20px 24px; margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(16,24,40,.05), 0 1px 2px rgba(16,24,40,.03);
}
.hero h1 {
    color: #1f2328; font-size: 1.4rem; margin: 0 0 5px 0;
    font-weight: 650; letter-spacing: .1px;
    display: flex; align-items: center; gap: 10px;
}
.hero p { color: #656d76; margin: 0; font-size: .87rem; }
.hero .chips { margin-top: 13px; display: flex; gap: 8px; flex-wrap: wrap; }
.chip {
    background: #f6f8fa; color: #57606a;
    padding: 4px 11px; border-radius: 7px; font-size: .75rem;
    border: 1px solid #e3e8ee;
    font-variant-numeric: tabular-nums;
}
.chip-strong { background: #eef6ff; color: #0969da; border-color: #cfe3fb; font-weight: 600; }

/* ---------- 指标卡片 ---------- */
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e3e8ee; border-radius: 12px;
    padding: 16px 18px;
    box-shadow: 0 1px 2px rgba(16,24,40,.04);
}
div[data-testid="stMetricLabel"] p {
    font-size: .75rem !important; color: #656d76 !important;
    letter-spacing: .4px;
}
div[data-testid="stMetricValue"] {
    font-size: 1.6rem; font-weight: 650; color: #1f2328;
    font-variant-numeric: tabular-nums;
}

/* ---------- 侧边栏 ---------- */
section[data-testid="stSidebar"] {
    background: #f8fafc;
    border-right: 1px solid #e3e8ee;
}
section[data-testid="stSidebar"] h2 {
    font-size: .76rem !important; color: #57606a; margin-top: .9rem;
    padding-bottom: .45rem; letter-spacing: .9px; font-weight: 650;
    text-transform: uppercase; border-bottom: 1px solid #e3e8ee;
}
section[data-testid="stSidebar"] label p { color: #24292f !important; font-size: .83rem; }
section[data-testid="stSidebar"] div[data-testid="stExpander"] {
    background: #ffffff; border: 1px solid #e3e8ee; border-radius: 10px;
}
section[data-testid="stSidebar"] div[data-testid="stExpander"] summary p {
    font-size: .82rem; color: #57606a; font-weight: 600;
}

/* ---------- 告警等级 ---------- */
.badge {
    display: inline-block; padding: 3px 10px; border-radius: 6px;
    font-size: .71rem; font-weight: 650; letter-spacing: .3px;
    border: 1px solid;
}
.badge-critical { background: #ffebe9; color: #cf222e; border-color: #ffcecb; }
.badge-high     { background: #fff2e5; color: #bc4c00; border-color: #ffd9b3; }
.badge-medium   { background: #fff8c5; color: #9a6700; border-color: #f2e08a; }
.badge-low      { background: #dafbe1; color: #1a7f37; border-color: #aee8bd; }

/* ---------- 空状态 + 三步引导 ---------- */
.empty-state {
    text-align: center; padding: 52px 28px 40px;
    border: 1px dashed #d8dee4; border-radius: 12px;
    background: linear-gradient(180deg, #fcfdfe 0%, #f6f8fa 100%);
    color: #656d76;
}
.empty-state .big { font-size: 2.3rem; margin-bottom: 12px; opacity: .6; }
.empty-state b { color: #24292f; font-size: 1.02rem; font-weight: 650; }
.empty-steps {
    display: flex; gap: 14px; justify-content: center;
    flex-wrap: wrap; margin-top: 26px;
}
.empty-step {
    flex: 1 1 190px; max-width: 250px; text-align: left;
    background: #ffffff; border: 1px solid #e3e8ee; border-radius: 10px;
    padding: 15px 17px; box-shadow: 0 1px 2px rgba(16,24,40,.04);
}
.empty-step .n {
    display: inline-flex; width: 22px; height: 22px; border-radius: 50%;
    background: #0969da; color: #ffffff; font-size: .74rem; font-weight: 700;
    align-items: center; justify-content: center; margin-bottom: 9px;
}
.empty-step b {
    display: block; color: #1f2328; font-size: .88rem;
    font-weight: 650; margin-bottom: 4px;
}
.empty-step span { color: #656d76; font-size: .78rem; line-height: 1.55; }

/* ---------- 图像 ---------- */
div[data-testid="stImage"] img {
    border-radius: 10px; border: 1px solid #e3e8ee;
    box-shadow: 0 1px 3px rgba(16,24,40,.05);
}

/* ---------- 进度条 ---------- */
div[data-testid="stProgress"] > div > div > div {
    border-radius: 4px; background: #0969da;
}

/* ---------- 标签页 ---------- */
button[data-baseweb="tab"] {
    font-size: .9rem; font-weight: 550; color: #57606a;
}
button[data-baseweb="tab"][aria-selected="true"] { color: #0969da; }
div[data-baseweb="tab-highlight"] { background: #0969da; }

/* ============================================================
   按钮：主按钮实心蓝、次按钮描边白底，统一 8px 圆角
   ============================================================ */
div.stButton > button,
div.stDownloadButton > button {
    width: 100%;
    border-radius: 8px;
    font-weight: 600;
    font-size: .9rem;
    padding: .52rem .9rem;
    border: 1px solid #d8dee4;
    background: #ffffff;
    color: #24292f;
    transition: background .15s, border-color .15s, box-shadow .15s;
}
div.stButton > button:hover,
div.stDownloadButton > button:hover {
    background: #f6f8fa; border-color: #b6bec6;
    box-shadow: 0 1px 2px rgba(16,24,40,.06);
}

/* 主按钮：实心蓝 */
div.stButton > button[kind="primary"] {
    background: #0969da;
    border: 1px solid #0969da;
    color: #ffffff;
    box-shadow: 0 1px 2px rgba(9,105,218,.28);
}
div.stButton > button[kind="primary"]:hover {
    background: #0860ca;
    border-color: #0860ca;
    box-shadow: 0 2px 6px rgba(9,105,218,.32);
}
div.stButton > button[kind="primary"]:active { background: #0757b8; }

/* 下载按钮 */
div.stDownloadButton > button {
    background: #ffffff; color: #0969da; border-color: #b6d3f5;
}
div.stDownloadButton > button:hover { background: #f0f7ff; }

/* ---------- 表格 ---------- */
div[data-testid="stDataFrame"] {
    border-radius: 10px; overflow: hidden;
    border: 1px solid #e3e8ee;
}

/* ---------- 提示 ---------- */
div[data-testid="stAlert"] { border-radius: 10px; }

/* ---------- 状态点 ---------- */
.dot { display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:6px; }
.dot-on  { background:#1a7f37; }
.dot-off { background:#8c959f; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

LEVEL_STYLE = {
    "critical": ("badge-critical", "🔴", "危急"),
    "high": ("badge-high", "🟠", "高"),
    "medium": ("badge-medium", "🟡", "中"),
    "low": ("badge-low", "🟢", "低"),
}

# COCO 17 关键点骨架连线
SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
]


# ============================================================
# 资源缓存
# ============================================================
@st.cache_resource(show_spinner=False)
def load_scene_configs() -> dict:
    return load_config()


@st.cache_resource(show_spinner="正在加载检测模型…")
def get_detector(scene_name: str, conf: float, weight_path: str, device_key: str,
                 pipeline: str = "single"):
    """加载检测器。weight_path 参与缓存 key，切换权重会自动重新加载。

    注意：这里的 conf 会被压到 DETECT_FLOOR，让低置信度的 no-hardhat 也能进
    稳定器（否则模型一犹豫就检不到，跟踪器没东西可跟）。
    真正的阈值过滤在 filter_by_class_conf() 里按类别做。
    """
    from agents.detector import DetectionAgent

    cfg = load_scene_configs()
    scene = get_scene(cfg, scene_name)
    scene["conf"] = min(float(conf), DETECT_FLOOR)   # 推理下限
    scene["user_conf"] = float(conf)                 # 用户设定，用于正常类别过滤
    if weight_path:
        scene["model"] = weight_path

    # 安全帽是三路可选：端到端 / 两步法 / 姿态兜底。
    # 其余场景一律走默认的 YOLO 检测器。
    if scene_name == "helmet" and pipeline != "single":
        from capabilities.helmet import HelmetCascadeDetector
        return HelmetCascadeDetector(scene, pipeline=pipeline, device="0")

    return DetectionAgent(scene)


# 检测器的最低阈值：低到能让 no-hardhat 的弱信号也进入稳定器
DETECT_FLOOR = 0.08


def filter_by_class_conf(dets: list[dict], user_conf: float,
                         violation_ids: set[int]) -> list[dict]:
    """
    按类别分别卡阈值。

    为什么分开：
        hardhat 模型很自信（0.5~0.8），用用户设定的阈值即可；
        no-hardhat 模型犹豫（0.1~0.35），用同样阈值会大量漏检。
        违规类别的门槛放低，靠稳定器的时间连续性压住误报。
    """
    out = []
    for d in dets:
        score = float(d.get("score", 0.0))
        is_violation = int(d.get("class_id", -1)) in violation_ids
        thr = DETECT_FLOOR if is_violation else user_conf
        if score >= thr:
            out.append(d)
    return out


WEIGHT_EXTS = (".engine", ".onnx", ".pt")


def list_weights(scene_name: str) -> list[Path]:
    """列出该场景模型目录下所有可用权重（engine / onnx / pt）。

    同目录下通常并存三种格式，让用户能在界面上直接切换：
        .engine 最快（需 TensorRT 环境）
        .onnx   跨平台
        .pt     最慢但最通用
    """
    cfg = load_scene_configs()
    scene = get_scene(cfg, scene_name)
    default = Path(scene["model"])
    wdir = default.parent
    found: list[Path] = []
    if wdir.exists():
        # 只保留与默认权重同源的模型：
        # 模型目录里常混着别的预训练权重（yolo26n.pt 之类），
        # 误选会因类别表不匹配直接报错，必须在列表阶段就过滤掉。
        stem = default.stem
        found = [p for p in wdir.iterdir()
                 if p.is_file()
                 and p.suffix.lower() in WEIGHT_EXTS
                 and (stem in p.stem or p.stem == stem)]
    if default.exists() and default not in found:
        found.append(default)
    # 排序：engine 优先，其次 onnx，最后 pt
    order = {".engine": 0, ".onnx": 1, ".pt": 2}
    return sorted(found, key=lambda p: (order.get(p.suffix.lower(), 9), p.name))


def weight_label(p: Path) -> str:
    """权重下拉框的显示文本：文件名 + 大小 + 速度提示。"""
    mb = p.stat().st_size / 1024 / 1024
    hint = {"engine": "最快 · 需 TensorRT", "onnx": "跨平台", "pt": "通用 · 较慢"}
    return f"{p.name}   {mb:.1f} MB   {hint.get(p.suffix.lstrip('.').lower(), '')}"


def get_scene_cfg(scene_name: str, conf: float) -> dict:
    cfg = load_scene_configs()
    scene = get_scene(cfg, scene_name)
    scene["conf"] = conf
    return scene


@st.cache_resource(show_spinner=False)
def get_output_paths() -> dict:
    cfg = load_scene_configs()
    return {k: str(v) for k, v in get_output_dirs(cfg).items()}


def get_alert_webhook() -> str:
    """读 system.yaml 的 alert.webhook_url（钉钉/企业微信机器人地址）。"""
    try:
        from config_loader import load_all

        alert = (load_all().get("system") or {}).get("alert") or {}
        return str(alert.get("webhook_url") or "").strip()
    except Exception:
        return ""


# ============================================================
# 绘制
# ============================================================
def draw_skeleton(vis: np.ndarray, keypoints: list[list[float]],
                  head_drop: float | None = None) -> None:
    """画姿态骨架 + 低头指标（睡岗场景）。"""
    kp = keypoints
    # 用 is None 判断而不是 not kp：kp 有可能来自板子端，是 numpy 数组，
    # 「非空数组的真值」会抛 ValueError: truth value ... is ambiguous
    if kp is None or len(kp) < 17:
        return

    for a, b in SKELETON:
        if kp[a][2] > 0.3 and kp[b][2] > 0.3:
            cv2.line(vis, (int(kp[a][0]), int(kp[a][1])),
                     (int(kp[b][0]), int(kp[b][1])), (0, 220, 255), 2)

    for i, p in enumerate(kp):
        if p[2] > 0.3:
            color = (0, 0, 255) if i == 0 else (0, 255, 0)
            cv2.circle(vis, (int(p[0]), int(p[1])), 4 if i == 0 else 3, color, -1)

    if kp[5][2] > 0.3 and kp[6][2] > 0.3:
        cv2.line(vis, (int(kp[5][0]), int(kp[5][1])),
                 (int(kp[6][0]), int(kp[6][1])), (255, 128, 0), 2)

    if head_drop is not None:
        pts = [p for p in kp if p[2] > 0.3]
        if pts:
            x0 = int(min(p[0] for p in pts))
            y0 = int(min(p[1] for p in pts))
        else:
            x0, y0 = 10, 30
        text = f"head_drop {head_drop:+.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ty = max(th + 6, y0 - 8)
        cv2.rectangle(vis, (x0, ty - th - 4), (x0 + tw + 6, ty + 2), (20, 20, 20), -1)
        cv2.putText(vis, text, (x0 + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 128, 255) if head_drop <= -0.32 else (0, 255, 255), 2)


def draw_detections(frame: np.ndarray, detections: list[dict],
                    violation_ids: set[int]) -> np.ndarray:
    """画检测框、分割掩码与姿态骨架，配色与告警截图一致。

    按任务类型分区绘制：
        分割（火灾）   → 半透明掩码填充 + 粗轮廓线，框只做细描边
        检测（安全帽） → 矩形框
        姿态（睡岗）   → 骨架 + 框
    """
    from agents.alert import color_for

    vis = frame.copy()
    h, w = vis.shape[:2]

    # 同类嵌套框去重：一个头同时被框成 0.88 和 0.55 时只留分数高的那个。
    # NMS 默认 IoU 阈值 0.7 管不到这种情况——小框整个落在大框里，
    # 面积比只有 0.5 左右，IoU 够不着阈值，于是画面上会出现「头上再叠一个小框」。
    detections = _suppress_nested(detections)

    # ---------- ① 分割掩码（先画，让框和标签压在上面）----------
    # 性能关键：所有掩码先画到同一张 overlay 上，最后只做一次 addWeighted。
    # 之前的写法是「每个目标 copy 一次全图 + 融合一次」，
    # 火灾场景有 9 个烟雾目标时会拷 9 次 1280x720（实测画框 14.7 ms），
    # 改成一次性处理后降到 1 ms 级别。
    mask_items = [d for d in detections if d.get("mask_polygons")]
    if mask_items:
        overlay = vis.copy()
        outlines = []           # (pts, color)
        for d in mask_items:
            cls_name = d.get("class", "")
            is_bad = int(d.get("class_id", -1)) in violation_ids
            color = color_for(cls_name, "high" if is_bad else "low")
            for poly in d["mask_polygons"]:
                if not poly or len(poly) < 3:
                    continue
                pts = np.array(poly, dtype=np.int32)
                # 28% 半透明填充：能看出火势范围，又不遮住火焰细节
                cv2.fillPoly(overlay, [pts], color)
                outlines.append((pts, color))
        vis = cv2.addWeighted(overlay, 0.28, vis, 0.72, 0)

        # 轮廓线加粗，不规则形状才看得清
        for pts, color in outlines:
            cv2.polylines(vis, [pts], True, color, 2)

    # ---------- ② 姿态骨架 ----------
    for d in detections:
        if d.get("keypoints"):
            draw_skeleton(vis, d["keypoints"], d.get("head_drop"))

    # ---------- ③ 矩形框与标签 ----------
    for d in detections:
        bbox = d.get("bbox")
        # 防御：检测结果缺 bbox 就别画了，否则一个脏数据会让整帧渲染失败
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = (int(v) for v in bbox)
        cls_name = d.get("class", "")
        is_bad = int(d.get("class_id", -1)) in violation_ids
        color = color_for(cls_name, "high" if is_bad else "low")
        has_mask = bool(d.get("mask_polygons"))
        is_tracked = bool(d.get("tracked"))     # True = 本帧漏检，靠轨迹补的框

        # 框按模型原始输出画，不做显示层缩放。
        # 旧版这里会把 no-hardhat 框外扩 35%（再按画面比例二次缩放），
        # 缩放以框中心为基准，而人头偏向框上部，结果框整体下移、
        # 罩住脖子和背景，看起来像是框错了。模型输出本身是贴合的。

        if is_tracked:
            # 补框：虚线 + 颜色减淡到 45%，一眼就能看出这是「顶住的」不是「看到的」
            dim = tuple(int(c * 0.45) for c in color)
            _draw_dashed_rect(vis, (x1, y1, x2, y2), dim, 1)
            color = dim
        else:
            # 有掩码时框画细一点，避免和轮廓抢视觉焦点
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1 if has_mask else 2)

        label = f"{cls_name} {float(d.get('score', 0.0)):.2f}"
        if is_tracked:
            label += " (跟踪)"
        # 分割任务附面积占比，这是判断火势大小的关键数据
        if d.get("area_ratio"):
            label += f"  面积{d['area_ratio'] * 100:.1f}%"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        lx = max(0, min(x1, w - tw - 6))
        if y1 - th - 6 < 0:
            ly0, ly1 = y1 + 1, y1 + th + 7
        else:
            ly0, ly1 = y1 - th - 6, y1
        cv2.rectangle(vis, (lx, ly0), (lx + tw + 4, ly1), (20, 20, 20), -1)
        cv2.putText(vis, label, (lx + 2, ly1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return vis


def _display_name(video_path) -> str:
    """视频源的人类可读名字。

    - 摄像头序号（int）→ ``camera0``
    - RTSP/HTTP 地址（str）→ ``rtsp_192.168.1.101_554``
    - 本地文件 → 文件名
    """
    if isinstance(video_path, int):
        return f"camera{video_path}"
    s = str(video_path)
    low = s.lower()
    if low.startswith(("rtsp://", "http://", "https://")):
        host = s.split("//", 1)[-1].split("@")[-1].split("/")[0]
        return "rtsp_" + host.replace(":", "_")
    return Path(s).name


def _display_stem(video_path) -> str:
    """同 ``_display_name``，但本地文件去掉扩展名。"""
    if isinstance(video_path, int) or str(video_path).lower().startswith(
        ("rtsp://", "http://", "https://")
    ):
        return _display_name(video_path)
    return Path(str(video_path)).stem


def _suppress_nested(detections: list[dict]) -> list[dict]:
    """丢掉两类「看着乱」的框：

    ① 同类嵌套：小框有 60% 以上面积落在同类高分框里（一个头画了两个框）。
    ② 低分虚框：分数低于 0.60 且与别的框重叠超过自身面积 20%
       （几个人挨得近时，模型会在两人中间的空白处补一个虚框）。

    只用于显示，不改检测结果与告警判断。
    """
    boxes = []
    for d in detections:
        b = d.get("bbox")
        if not b or len(b) != 4:
            continue
        try:
            x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        except (TypeError, ValueError):
            continue
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area <= 0:
            continue
        boxes.append((d, str(d.get("class", "")), float(d.get("score", 0.0) or 0.0),
                      x1, y1, x2, y2, area))

    keep = []
    for i, (d, cls_i, sc_i, x1, y1, x2, y2, area_i) in enumerate(boxes):
        covered = False
        for j, (_, cls_j, sc_j, a1, b1, a2, b2, _) in enumerate(boxes):
            if i == j:
                continue
            ix = min(x2, a2) - max(x1, a1)
            iy = min(y2, b2) - max(y1, b1)
            if ix <= 0 or iy <= 0:
                continue
            frac = (ix * iy) / area_i
            # ① 同类嵌套
            if cls_j == cls_i and sc_j > sc_i and frac > 0.60:
                covered = True
                break
            # ② 低分虚框压在别人身上
            if sc_i < 0.60 and frac > 0.20:
                covered = True
                break
        if not covered:
            keep.append(d)
    return keep


def _draw_dashed_rect(img: np.ndarray, rect: tuple[int, int, int, int],
                      color, thickness: int = 1, dash: int = 8, gap: int = 6):
    """画虚线框（用于「跟踪补框」和「真实检出」的视觉区分）。"""
    x1, y1, x2, y2 = rect

    # 上边 + 下边
    for y in (y1, y2):
        x = x1
        while x < x2:
            cv2.line(img, (x, y), (min(x + dash, x2), y), color, thickness)
            x += dash + gap

    # 左边 + 右边
    for x in (x1, x2):
        y = y1
        while y < y2:
            cv2.line(img, (x, y), (x, min(y + dash, y2)), color, thickness)
            y += dash + gap


def list_videos() -> list[Path]:
    vroot = PROJECT_ROOT / "datasets" / "videos"
    return sorted(p for p in vroot.rglob("*.mp4")) if vroot.exists() else []


def save_uploaded_video(uploaded) -> Path:
    tmp_dir = PROJECT_ROOT / "outputs" / "_uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / Path(uploaded.name).name
    path.write_bytes(uploaded.getbuffer())
    return path


# ============================================================
# 核心处理
# ============================================================
def run_detection(video_path: Path | int, scene_name: str, conf: float,
                  sample_fps: float, max_frames: int, location: str,
                  weight_path: str = "", pipeline: str = "single") -> dict:
    """执行一次完整检测，实时更新界面。video_path 为 int 时表示摄像头索引。"""
    from agents.alert import AlertAgent
    from agents.judgment import JudgmentAgent
    from agents.reporter import ReportAgent
    from agents.video_capture import VideoCaptureAgent
    from agents.video_writer import VideoWriter, h264_available
    from graph_def import build_graph

    scene = get_scene_cfg(scene_name, conf)
    violation_ids = set(int(c) for c in scene["violation_classes"])
    out_paths = get_output_paths()

    video_agent = VideoCaptureAgent(video_path, sample_fps=sample_fps,
                                    max_frames=max_frames,
                                    camera_width=640, camera_height=480)
    info = video_agent.info()

    detector = get_detector(scene_name, conf, weight_path, "gpu0", pipeline)
    # 换视频必须清空轨迹，否则上一段的框会串到这一段的头几帧。
    # 旧版权重可能没有这个方法，属于预期内的兼容情况，记一条 debug 即可。
    try:
        detector.reset_stabilizer()
    except AttributeError as e:
        _LOG.debug("检测器不支持 reset_stabilizer，跳过轨迹重置: %s", e)
    judgment = JudgmentAgent(scene)
    alert_agent = AlertAgent(scene, snapshot_dir=Path(out_paths["snapshots"]),
                             channels=("console", "file", "webhook"),
                             location=location, webhook_url=get_alert_webhook())
    reporter = ReportAgent(Path(out_paths["reports"]), scene)
    video_name = _display_name(video_path)

    graph, ref = build_graph(scene, detector, judgment, alert_agent, reporter,
                             video_agent, video_name=video_name)
    ref["iter"] = video_agent.iter_frames()

    # ---------- 录像器：边检测边录，处理完用播放器回放 ----------
    rec_path = Path(out_paths.get("videos", "outputs/videos")) / f"{scene_name}_{video_name}.mp4"
    _rec_can_write = h264_available()
    rec_writer = None
    _rec_size: tuple[int, int] | None = None
    # 写入 fps 必须等于「真实采样率」，否则回放会变速/顿挫。
    #   step = round(source_fps / sample_fps)，实际采样率 = source_fps / step
    #   例：24fps 源 + 抽帧 10fps → step=2 → 真实采样 12fps
    #   若写成 10fps，输出时长会比原视频长，画面像是慢放+顿挫。
    _src_fps = float(info.get("source_fps") or 0)
    _step = int(info.get("frame_step") or 1)
    _sample = float(info.get("sample_fps") or 0)
    if info.get("full_frame_mode") or _sample <= 0:
        rec_fps = _src_fps if _src_fps > 0 else 25.0   # 逐帧：用源帧率
    elif _src_fps > 0 and _step > 0:
        rec_fps = _src_fps / _step                     # 抽帧：用真实采样率
    else:
        rec_fps = _sample                              # 兜底
    # 录像分辨率跟随源帧：不要把 1280×720 压成 640×480，
    # 压缩后框线和标签会跟着缩小，回放看着又糊又小，还容易误以为框偏了。

    # ---------- 界面占位 ----------
    col_frame, col_stat = st.columns([3, 1], gap="medium")
    with col_stat:
        st.markdown("##### 实时状态")
        stat_box = st.empty()
        alert_feed = st.empty()
    frame_box = col_frame.empty()

    progress = st.progress(0.0, text="准备中…")
    expected = max(1, info["expected_samples"] or max_frames or 1)

    # 性能曲线占位（每 3 帧刷新一次，避免拖慢主循环）
    curve_box = st.empty()
    latency_hist: list[float] = []
    fps_hist: list[float] = []

    records: list[dict] = []
    processed = 0
    t_start = time.time()

    # 保留最后一帧，循环结束后继续显示，不能处理完就变空白
    last_vis: np.ndarray | None = None
    last_caption: str = ""
    # 界面推送节流：每秒最多推 12 次画面
    ui_min_interval = 1.0 / 12.0
    last_ui_time = 0.0

    while True:
        state = graph.invoke({
            "video_path": video_name,
            "scene": scene_name,
            "alerts": [],
            "alert_records": [],
        })
        idx = state.get("frame_index", -1)
        if idx is None or idx < 0:
            break

        processed += 1
        new_records = state.get("alert_records") or []
        records.extend(new_records)

        frame = state.get("frame")
        dets = state.get("detections") or []
        if frame is not None:
            vis = draw_detections(frame, dets, violation_ids)
            # 每一帧都写进录像文件（不受界面刷新节奏影响），分辨率取源帧
            if _rec_can_write:
                fh, fw = vis.shape[:2]
                if _rec_size != (fw, fh):
                    if rec_writer is not None:
                        rec_writer.release()
                    rec_writer = VideoWriter(rec_path, width=fw, height=fh, fps=rec_fps)
                    _rec_size = (fw, fh)
                rec_writer.write(vis)

            # 界面刷新：按「真实时间」节流，不按帧数。
            # 推理只要 8ms，瓶颈是 Streamlit 的 WebSocket 推送 + 浏览器重绘。
            # 因此控制每秒推送次数（默认 12 次），既流畅又不拖慢检测。
            now_ts = time.time()
            if (now_ts - last_ui_time >= ui_min_interval) or new_records:
                # 缩小到 720p 以内，减少传输体积
                h, w = vis.shape[:2]
                if w > 960:
                    scale = 960.0 / w
                    vis_small = cv2.resize(vis, (960, int(h * scale)))
                else:
                    vis_small = vis
                last_vis = cv2.cvtColor(vis_small, cv2.COLOR_BGR2RGB)
                last_caption = (f"第 {processed} / 约 {expected} 帧　"
                                f"时间 {state.get('timestamp', 0):.2f}s")
                frame_box.image(last_vis, caption=last_caption,
                                width="stretch")
                last_ui_time = now_ts

        bad = sum(1 for d in dets if int(d["class_id"]) in violation_ids)
        lines = [
            f"**本帧目标** {len(dets)}　|　**违规** {bad}",
            f"**累计告警** {len(records)}",
            f"**已处理** {processed} 帧",
        ]
        tracks = judgment.track_summary()
        if tracks:
            need = int(scene["rules"].get("min_consecutive_frames", 1))
            lines.append("")
            lines.append(f"**跟踪中** {len(tracks)} 人")
            for tr in tracks[:3]:
                pct = min(100, int(100 * tr["frames"] / max(1, need)))
                mv = tr.get("travel", 0)
                lines.append(
                    f"　#{tr['track_id']}　{tr['duration']:.1f}s　走了{mv:.1f}身　"
                    f"({tr['frames']}/{need}) {'▓' * (pct // 20)}{'░' * (5 - pct // 20)}"
                )
        if scene["task"] == "pose":
            vals = [d.get("head_drop") for d in dets if d.get("head_drop") is not None]
            if vals:
                thr = scene["rules"].get("posture", {}).get("threshold", -0.32)
                lines.append("")
                lines.append(f"**低头指标** {', '.join(f'{v:+.2f}' for v in vals)}")
                lines.append(f"　阈值 {thr}（超过即低头）")
        stat_box.markdown("  \n".join(lines))

        # 采集本帧耗时并定期画曲线
        if detector.infer_ms:
            latency_hist.append(round(detector.infer_ms[-1], 2))
        elapsed_so_far = max(0.001, time.time() - t_start)
        fps_hist.append(round(processed / elapsed_so_far, 1))
        if processed % 3 == 0 and len(latency_hist) >= 2:
            df_perf = pd.DataFrame({
                "推理耗时 ms": latency_hist,
                "等效 FPS": fps_hist,
            })
            with curve_box.container():
                c1, c2 = st.columns(2)
                with c1:
                    st.caption("推理耗时（ms）")
                    st.line_chart(df_perf["推理耗时 ms"], height=130)
                with c2:
                    st.caption("等效 FPS")
                    st.line_chart(df_perf["等效 FPS"], height=130)

        if new_records:
            latest = new_records[-1]
            _, icon, _ = LEVEL_STYLE.get(latest.get("level", "high"), ("", "🟠", ""))
            alert_feed.warning(
                f"{icon} **{latest['title']}**　{latest['video_time']}\n\n"
                f"{latest['class_name']} · {latest['max_score']:.2f}"
            )

        progress.progress(min(1.0, processed / expected),
                          text=f"处理中 {processed}/{expected} 帧　告警 {len(records)} 条")

    elapsed = time.time() - t_start
    progress.progress(1.0, text=f"完成　{processed} 帧　{len(records)} 条告警")

    # 先算出性能数据，播放器侧栏要用
    result_perf_median = 0.0
    result_fps = 0.0
    if detector.infer_ms:
        result_perf_median = float(np.median(detector.infer_ms))
        result_fps = 1000.0 / result_perf_median if result_perf_median else 0.0

    # 关闭录像器，得到完整视频
    rec_file: Path | None = None
    if rec_writer is not None:
        try:
            rec_file = rec_writer.close()
        except Exception:
            rec_file = None

    # 处理结束后把最后一帧留在画面上，并标注「处理完成」
    # 注意：这里只构造图，不再渲染回放。
    # 回放统一由 render_result_panel() 在 tab 里画——因为 Streamlit 每次交互都会
    # 重跑整个脚本，本函数内的 st.video 在重跑后会消失，用户就"看不到东西"了。
    final_img: np.ndarray | None = None
    if last_vis is not None:
        final_img = last_vis.copy()
        banner = "处理完成"
        (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(final_img, (8, 8), (tw + 24, th + 22), (20, 20, 20), -1)
        cv2.putText(final_img, banner, (16, th + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 100), 2)
        frame_box.image(final_img,
                        caption=f"{last_caption}　（处理完成）",
                        width="stretch")

    perf = detector.stats()
    perf["elapsed_sec"] = round(elapsed, 2)
    info_out = dict(info)
    info_out["processed_frames"] = processed
    report = reporter.build(info_out, records, perf)
    stem = _display_stem(video_path)
    paths = reporter.save(report, stem=f"{scene_name}_{stem}")

    return {
        "records": records, "report": report, "paths": paths,
        "perf": perf, "processed": processed, "elapsed": elapsed,
        "info": info_out, "scene": scene,
        "video": str(rec_file) if rec_file else None,
        "last_frame": final_img,
    }


def render_result_panel(result: dict) -> None:
    """把一次检测的结果渲染出来：指标 + 视频回放 + 下载。

    为什么要单独一层：Streamlit 每次交互都会重跑整个脚本，
    run_detection() 内部画的 st.video 在重跑后会消失，
    用户点一下别的地方就"什么都看不到了"。
    所以回放必须从 session_state 里存着的 result["video"] 路径重新渲染。
    """
    p = result.get("perf") or {}
    records = result.get("records") or []

    vid = result.get("video")
    vp = Path(vid) if vid else None
    if vp is not None and vp.exists() and vp.stat().st_size > 4096:
        st.markdown("##### 🎬 检测结果回放")
        st.caption("浏览器原生播放器：可暂停、拖动进度条、全屏、下载")
        vc1, vc2 = st.columns([2, 1])
        with vc1:
            st.video(str(vp))
        with vc2:
            st.markdown("**回放说明**")
            st.markdown(
                f"- 总帧数：**{result.get('processed', 0)}**\n"
                f"- 告警事件：**{len(records)}** 条\n"
                f"- 检测耗时：{p.get('median_ms', 0):.1f} ms / 帧\n"
                f"- 等效速度：**{p.get('fps', 0):.0f} FPS**\n"
            )
            st.markdown("---")
            with open(vp, "rb") as fh:
                st.download_button("⬇️ 下载检测视频", data=fh.read(),
                                   file_name=vp.name, mime="video/mp4",
                                   width="stretch")
            st.caption("文件存在 outputs/videos/ 目录")
    elif result.get("last_frame") is not None:
        st.image(result["last_frame"],
                 caption="最后一帧（本次未生成回放视频）",
                 width="stretch")
    else:
        st.warning("本次没有生成回放视频：可能是抽帧帧数过少，或本机缺少 H.264 编码器")

    # 指标卡紧贴视频下方：先看画面，再看数字，符合观看顺序
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("处理帧数", result.get("processed", 0))
    m2.metric("告警总数", len(records))
    m3.metric("检测耗时", f"{p.get('median_ms', 0):.1f} ms")
    m4.metric("等效 FPS", f"{p.get('fps', 0):.0f}")

    st.success(f"✅ 处理完成，耗时 {result.get('elapsed', 0):.1f} 秒")


# ============================================================
# 摄像头持续运行模式
# ============================================================
CAM_BATCH = 8          # 每批处理的帧数，批间交出控制权以便响应暂停


def _get_camera(idx):
    """获取或创建摄像头对象，跨批次复用，避免反复开关设备。

    ``idx`` 可以是本地设备序号（int），也可以是 RTSP 地址（str）。
    """
    cap = st.session_state.get("cam_cap")
    if cap is not None and cap.isOpened():
        return cap
    if isinstance(idx, str):
        cap = cv2.VideoCapture(idx, cv2.CAP_FFMPEG)
        if cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        st.session_state["cam_cap"] = cap
        return cap if cap.isOpened() else None
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    st.session_state["cam_cap"] = cap
    return cap


def release_camera() -> None:
    """释放摄像头。"""
    cap = st.session_state.get("cam_cap")
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    st.session_state["cam_cap"] = None


def run_camera_frame(scene_name: str, conf: float, cam_idx: int,
                     location: str, weight_path: str = "",
                     detect_every: int = 1) -> dict | None:
    """处理摄像头的一帧，返回本帧结果。

    为什么每次只处理一帧：
        Streamlit 的脚本是同步执行的，一个长循环期间无法响应按钮点击。
        改成「每次 rerun 处理一帧」后，浏览器上的「暂停」按钮才能生效。

    detect_every>1 时做「检测跳帧」：画面每帧都取最新的，推理每 N 帧做一次，
    中间复用上一次的检测框。画面更跟手，推理负载降到 1/N。
    """
    from agents.alert import AlertAgent
    from agents.judgment import JudgmentAgent

    cap = _get_camera(cam_idx)
    if cap is None:
        st.error(f"无法打开摄像头 {cam_idx}")
        return None

    ok, frame = cap.read()
    if not ok or frame is None:
        return {"frame": None, "detections": [], "alerts": []}

    n = int(st.session_state.get("cam_frames_read", 0))
    st.session_state["cam_frames_read"] = n + 1
    every = max(1, int(detect_every or 1))

    if every > 1 and n % every != 0:
        last = st.session_state.get("cam_last_ctx")
        if last is not None:
            # 复用上一次检测框，不重复判定/推送
            return {"frame": frame,
                    "detections": last.get("detections") or [],
                    "alerts": [], "records": [],
                    "judge": last.get("judge"), "scene": last.get("scene"),
                    "cached": True}

    scene = get_scene_cfg(scene_name, conf)
    detector = get_detector(scene_name, conf, weight_path, "gpu0")
    detections = detector.detect(frame)

    # 判断逻辑需要跨帧状态：跟踪器/计数器保存在 session_state 里
    judge = st.session_state.get("cam_judge")
    if judge is None or judge.name != scene_name:
        judge = JudgmentAgent(scene)
        st.session_state["cam_judge"] = judge

    ts = time.time() - st.session_state.get("cam_t0", time.time())
    alerts = judge.judge(detections, ts)

    out_paths = get_output_paths()
    alert_agent = st.session_state.get("cam_alert")
    if alert_agent is None or alert_agent.scene.get("name") != scene_name:
        alert_agent = AlertAgent(scene, snapshot_dir=Path(out_paths["snapshots"]),
                                 channels=("file", "webhook"), location=location,
                                 webhook_url=get_alert_webhook())
        st.session_state["cam_alert"] = alert_agent

    records = [alert_agent.push(a, frame=frame, video_name=_display_name(cam_idx))
               for a in alerts]

    ctx = {"frame": frame, "detections": detections, "alerts": alerts,
           "records": records, "judge": judge, "scene": scene}
    if every > 1:
        st.session_state["cam_last_ctx"] = ctx
    return ctx


def render_camera_panel(scene_name: str, conf: float, cam_idx: int, location: str,
                        fps: float = 5.0, weight_path: str = "") -> bool:
    """摄像头实时面板：开始/暂停控制 + 实时画面 + 实时时间。
    
    Returns:
        True 表示需要继续下一帧（由调用方统一 st.rerun，避免打断标签页渲染）
    """
    running = bool(st.session_state.get("cam_running"))
    scene = get_scene_cfg(scene_name, conf)
    violation_ids = set(int(c) for c in scene["violation_classes"])

    # ---------- 控制条 ----------
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.2, 1.2, 1.5, 2.5])
    if ctrl1.button("⏸️ 暂停" if running else "▶️ 开始",
                    width="stretch", type="primary"):
        if running:
            st.session_state["cam_running"] = False
            auto_camera_report(scene_name, conf, cam_idx, location)
        else:
            st.session_state["cam_running"] = True
            st.session_state["cam_t0"] = time.time()
            st.session_state["cam_processed"] = 0
            st.session_state["cam_records"] = []
            st.session_state["cam_judge"] = None
            st.session_state["cam_alert"] = None
            # 重开摄像头也要清轨迹，避免上一次的框残留。
            # 这里失败不影响开流，但记一条日志便于排查。
            try:
                get_detector(scene_name, conf, weight_path, "gpu0").reset_stabilizer()
            except Exception as e:
                _LOG.warning("重开摄像头时重置轨迹失败（不影响开流）: %s", e)
        return False

    if ctrl2.button("⏹️ 停止并重置", width="stretch"):
        auto_camera_report(scene_name, conf, cam_idx, location)
        st.session_state["cam_running"] = False
        release_camera()
        st.session_state["cam_processed"] = 0
        st.session_state["cam_records"] = []
        st.session_state["cam_judge"] = None
        return False

    state_text = "🟢 运行中" if running else "⚪ 已暂停"
    ctrl3.markdown(f"<div style='padding-top:8px'>{state_text}</div>",
                   unsafe_allow_html=True)

    if not running:
        st.info("点击「▶️ 开始」后摄像头画面会实时显示，随时可暂停。")
        # 仍显示上次结果
        records = st.session_state.get("cam_records") or []
        if records:
            st.markdown(f"上次运行累计告警 **{len(records)}** 条")
            agent = st.session_state.get("cam_alert")
            if agent is not None:
                _s = agent.summary()
                if _s.get("log_file"):
                    st.caption(
                        f"告警已落盘 `{Path(_s['log_file']).name}`"
                        f" · 推送成功 {_s.get('pushed', 0)} / 失败 {_s.get('push_failed', 0)}"
                    )
            if st.session_state.get("cam_report_paths"):
                st.caption(
                    "报告已生成：`"
                    + Path(st.session_state["cam_report_paths"]["markdown"]).name + "`"
                )
        return False

    # ---------- 单帧处理 ----------
    frame_start = time.time()
    # 检测跳帧：每秒检测 fps 次，剩下的帧只刷新画面（复用上一次的检测框）
    detect_every = max(1, int(round(30.0 / max(float(fps or 5.0), 0.5))))
    result = run_camera_frame(scene_name, conf, cam_idx, location,
                              weight_path=weight_path,
                              detect_every=detect_every)
    if result is None:
        st.session_state["cam_running"] = False
        return False

    processed = int(st.session_state.get("cam_processed", 0)) + 1
    st.session_state["cam_processed"] = processed
    records = list(st.session_state.get("cam_records") or [])
    records.extend(result.get("records") or [])
    st.session_state["cam_records"] = records

    frame = result.get("frame")
    dets = result.get("detections") or []

    # ---------- 画面 ----------
    col_frame, col_stat = st.columns([3, 1], gap="medium")
    with col_stat:
        st.markdown("##### 实时状态")
        stat_box = st.empty()
        feed_box = st.empty()

    if frame is not None:
        vis = draw_detections(frame, dets, violation_ids)
        # 右上角叠加实时时间
        now_text = time.strftime("%H:%M:%S")
        (tw, th), _ = cv2.getTextSize(now_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        w = vis.shape[1]
        cv2.rectangle(vis, (w - tw - 16, 6), (w - 6, th + 16), (20, 20, 20), -1)
        cv2.putText(vis, now_text, (w - tw - 11, th + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 255), 2)
        col_frame.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                        width="stretch")
    else:
        col_frame.warning("读取摄像头帧失败，已暂停")
        st.session_state["cam_running"] = False
        return False

    # ---------- 状态 ----------
    elapsed = time.time() - st.session_state.get("cam_t0", time.time())
    bad = sum(1 for d in dets if int(d["class_id"]) in violation_ids)
    lines = [
        f"**实时时间** {time.strftime('%H:%M:%S')}",
        f"**运行时长** {int(elapsed // 60):02d}:{int(elapsed % 60):02d}",
        f"**已处理** {processed} 帧",
        f"**本帧目标** {len(dets)}　违规 {bad}",
        f"**累计告警** {len(records)}",
    ]
    judge = st.session_state.get("cam_judge")
    if judge is not None:
        tracks = judge.track_summary()
        if tracks:
            need = int(scene["rules"].get("min_consecutive_frames", 1))
            lines.append("")
            lines.append(f"**跟踪中** {len(tracks)} 人")
            for tr in tracks[:3]:
                pct = min(5, int(5 * tr["frames"] / max(1, need)))
                mv = tr.get("travel", 0)
                lines.append(f"　#{tr['track_id']} {tr['duration']:.1f}s 走了{mv:.1f}身 "
                             f"({tr['frames']}/{need}) {'▓' * pct}{'░' * (5 - pct)}")
        if scene["task"] == "pose":
            vals = [d.get("head_drop") for d in dets if d.get("head_drop") is not None]
            if vals:
                thr = scene["rules"].get("posture", {}).get("threshold", -0.32)
                lines.append("")
                lines.append(f"**低头指标** {', '.join(f'{v:+.2f}' for v in vals)}")
                lines.append(f"　阈值 {thr}")
    stat_box.markdown("  \n".join(lines))

    s_agent = st.session_state.get("cam_alert")
    if s_agent is not None and getattr(s_agent, "count", 0):
        _s = s_agent.summary()
        col_stat.caption(
            f"告警已落盘 `{Path(_s['log_file']).name}`"
            f" · 推送成功 {_s['pushed']} / 失败 {_s['push_failed']}"
        )

    if result.get("records"):
        latest = result["records"][-1]
        _, icon, _ = LEVEL_STYLE.get(latest.get("level", "high"), ("", "🟠", ""))
        feed_box.warning(f"{icon} **{latest['title']}**　{latest['video_time']}\n\n"
                         f"{latest['class_name']} · {latest['max_score']:.2f}")

    # 控制帧率：按目标 fps 补齐间隔，避免刷太快占满 CPU
    interval = 1.0 / max(1.0, float(fps))
    used = time.time() - frame_start
    if used < interval:
        time.sleep(interval - used)

    # 由外层统一 rerun，使告警列表与报告也每帧刷新
    return True


# ============================================================
# 共用渲染组件
# ============================================================
def render_alert_dashboard(records: list[dict]) -> None:
    """告警统计仪表盘：类别分布、时间趋势、点位分布。"""
    if not records:
        return

    df = pd.DataFrame(records)
    # 视频时间转秒，用于画趋势
    def _to_sec(v) -> float:
        try:
            parts = str(v).split(":")
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        except Exception:
            pass
        return 0.0

    df["秒"] = df["video_time"].map(_to_sec) if "video_time" in df else 0.0
    cat_col = "class_name" if "class_name" in df else None

    with st.expander("📊 统计仪表盘", expanded=True):
        c1, c2 = st.columns(2)

        # 左：告警类别分布
        with c1:
            if cat_col:
                st.caption("告警类别分布")
                vc = df[cat_col].value_counts()
                st.bar_chart(vc, height=180)
            else:
                st.caption("暂无类别数据")

        # 右：告警等级分布
        with c2:
            st.caption("告警等级分布")
            lv = df["level"].value_counts() if "level" in df else pd.Series(dtype=int)
            if not lv.empty:
                st.bar_chart(lv, height=180)

        # 时间趋势：按视频时间排序后画累计告警
        st.caption("告警时间分布（视频时间轴）")
        if len(df) >= 2:
            trend = df.sort_values("秒")[["秒", "video_time"]].copy()
            trend["累计告警"] = range(1, len(trend) + 1)
            trend["时刻(秒)"] = trend["秒"]
            st.line_chart(trend.set_index("时刻(秒)")["累计告警"], height=160)
        else:
            st.info("告警数不足 2 条，暂无趋势图")

        # 点位分布
        if "location" in df and df["location"].nunique() > 1:
            st.caption("点位分布")
            st.bar_chart(df["location"].value_counts(), height=150)


def render_alert_table(records: list[dict]) -> None:
    """渲染告警列表：等级汇总 + 统计图表 + 表格 + 证据截图。"""
    from collections import Counter

    cnt = Counter(r.get("level", "medium") for r in records)
    cols = st.columns(len(LEVEL_STYLE))
    for i, (lvl, (_, icon, label)) in enumerate(LEVEL_STYLE.items()):
        cols[i].metric(f"{icon} {label}", cnt.get(lvl, 0))

    st.markdown("")
    render_alert_dashboard(records)
    st.markdown("")
    rows = [{
        "编号": a.get("alert_id"),
        "视频时间": a.get("video_time"),
        "事件": a.get("title"),
        "等级": a.get("level"),
        "类别": a.get("class_name"),
        "置信度": round(float(a.get("max_score", 0)), 3),
        "目标数": a.get("count", 1),
        "地点": a.get("location"),
    } for a in records]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    snaps = [a["snapshot"] for a in records
             if a.get("snapshot") and Path(a["snapshot"]).exists()]
    if snaps:
        st.markdown("#### 证据截图")
        ncol = 3
        for i in range(0, len(snaps), ncol):
            cs = st.columns(ncol)
            for j in range(ncol):
                if i + j < len(snaps):
                    with cs[j]:
                        st.image(snaps[i + j], caption=Path(snaps[i + j]).name,
                                 width="stretch")


def generate_camera_report(cam_idx: int, scene_name: str, conf: float,
                           location: str, records: list[dict]) -> None:
    """为摄像头会话生成报告并存入 session_state。"""
    from agents.reporter import ReportAgent

    scene = get_scene_cfg(scene_name, conf)
    out_paths = get_output_paths()
    reporter = ReportAgent(Path(out_paths["reports"]), scene)

    processed = int(st.session_state.get("cam_processed", 0))
    t0 = st.session_state.get("cam_t0", time.time())
    elapsed = max(0.1, time.time() - t0)

    info = {
        "path": _display_name(cam_idx),
        "source_type": "camera",
        "resolution": "640x480",
        "duration_sec": round(elapsed, 1),
        "source_fps": 0,
        "sample_fps": 0,
        "frame_step": 1,
        "processed_frames": processed,
    }
    perf = {
        "count": processed,
        "median_ms": 0, "p95_ms": 0,
        "fps": round(processed / elapsed, 1) if processed else 0,
        "elapsed_sec": round(elapsed, 1),
    }
    report = reporter.build(info, records, perf)
    tag = Path(str(cam_idx)).name if not isinstance(cam_idx, int) else f"camera{cam_idx}"
    tag = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in tag)[:40] or "camera"
    paths = reporter.save(report, stem=f"{tag}_{scene_name}")
    st.session_state["cam_report_paths"] = paths


def auto_camera_report(scene_name: str, conf: float, cam_idx,
                       location: str) -> None:
    """暂停/停止摄像头时自动出报告。

    只在「有告警」且「告警数比上次生成时多」时重新写，避免反复点暂停
    刷出一堆同名报告。
    """
    records = st.session_state.get("cam_records") or []
    if not records:
        return
    last_n = int(st.session_state.get("cam_report_n", -1))
    if st.session_state.get("cam_report_paths") and last_n == len(records):
        return
    try:
        generate_camera_report(cam_idx, scene_name, conf, location, records)
        st.session_state["cam_report_n"] = len(records)
    except Exception as e:  # 报告失败不能打断监控
        _LOG.warning("自动生成报告失败: %s", e)


# ============================================================
# 界面
# ============================================================
def render_hero(scene: dict | None, running: bool = False) -> None:
    """顶部横幅：状态灯 + 场景 + 模型信息胶囊。"""
    if scene:
        _TASK_CN = {"detect": "目标检测", "segment": "实例分割", "pose": "姿态估计"}
        chips = (
            f'<span class="chip chip-strong">{scene["display_name"]}</span>'
            f'<span class="chip">{_TASK_CN.get(scene["task"], scene["task"])}</span>'
            f'<span class="chip">模型 {Path(scene["model"]).name}</span>'
            f'<span class="chip">置信度 {float(scene["conf"]):.2f}</span>'
        )
        subtitle = f'当前场景　<b style="color:#cfe9f7">{scene["display_name"]}</b>'
    else:
        chips = ""
        subtitle = "五场景：安全帽 / 火灾烟雾 / 睡岗 / 徘徊 / 值班室综合"

    dot = '<span class="dot dot-on"></span>' if running else '<span class="dot dot-off"></span>'
    state_text = "运行中" if running else "待机"
    state_color = "#1a7f37" if running else "#8c959f"

    st.markdown(
        f'<div class="hero">'
        f'<h1>🛡️ 安防智能体 <span style="color:#0969da">V1.0</span>'
        f'<span style="margin-left:auto;font-size:.82rem;font-weight:600;color:{state_color}">'
        f'{dot}{state_text}</span></h1>'
        f'<p>{subtitle}　·　LangGraph 多智能体</p>'
        f'<div class="chips">{chips}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def badge_html(level: str) -> str:
    cls, icon, label = LEVEL_STYLE.get(level, ("badge-medium", "🟡", level))
    return f'<span class="badge {cls}">{icon} {label}</span>'


def main() -> None:
    cfg = load_scene_configs()
    scenes = list_scenes(cfg)

    # ---------------- 侧边栏 ----------------
    with st.sidebar:
        st.markdown("## ⚙️ 检测设置")
        scene_name = st.selectbox(
            "场景", scenes,
            format_func=lambda s: f"{cfg['scenes'][s]['display_name']}（{s}）",
        )
        scene_preview = get_scene(cfg, scene_name)

        # ---------- 模型权重选择 ----------
        weights = list_weights(scene_name)
        if len(weights) > 1:
            weight_choice = st.selectbox(
                "模型权重", weights, index=0,
                format_func=weight_label,
                key=f"weight_{scene_name}",
                help=".engine 最快但需 TensorRT 环境；.onnx 跨平台；.pt 最通用",
            )
        elif weights:
            weight_choice = weights[0]
            st.caption(f"权重：`{weight_choice.name}`")
        else:
            weight_choice = None
            st.warning("该场景找不到可用权重")
        weight_path = str(weight_choice) if weight_choice else ""

        if scene_name == "helmet":
            from capabilities.helmet import PIPELINE_LABELS
            pipeline = st.selectbox(
                "安全帽检测方式",
                options=list(PIPELINE_LABELS.keys()),
                format_func=lambda k: PIPELINE_LABELS[k],
                index=0,
                key=f"pipeline_{scene_name}",
                help="端到端最快；两步法对远处小人头更准；"
                     "姿态兜底能补「人站着但模型没框到」的漏检",
            )
        else:
            pipeline = "single"

        conf = st.slider("置信度阈值", 0.10, 0.90, float(scene_preview["conf"]), 0.05,
                         key=f"conf_{scene_name}",
                         help="调高减少误报，调低减少漏报")

        scene_fps = float(scene_preview.get("sample_fps") or 1.0)
        # 含跟踪规则的场景（值班室综合值守）帧率不能太低，否则 ID 会断
        has_tracking = "multi_rules" in scene_preview.get("rules", {}) or \
            scene_preview.get("rules", {}).get("type") == "tracking"

        # ---------- 处理模式：抽帧 / 逐帧全速 ----------
        full_frame = st.toggle(
            "⚡ 逐帧全速（不抽帧）",
            value=False,
            key=f"fullframe_{scene_name}",
            help="关闭 = 按下方帧率抽帧，省算力；"
                 "开启 = 每一帧都检测，处理速度最快，适合捕捉瞬时事件",
        )

        if full_frame:
            sample_fps = 0.0      # 0 表示逐帧，不抽帧
            st.caption("逐帧模式：视频每一帧都会被检测")
        else:
            fps_hint = ("该场景含徘徊跟踪，低于 3fps 跟踪会断"
                        if has_tracking else "每秒取多少帧检测")
            sample_fps = st.slider("抽帧帧率", 0.5, 10.0, scene_fps, 0.5,
                                   key=f"fps_{scene_name}", help=fps_hint)

        with st.expander("📍 点位与输出设置", expanded=False):
            location = st.text_input("点位名称", value="施工区 A 号点位",
                                     help="会写进告警记录和报告标题，多路部署时用来区分位置")

        st.markdown("## 📹 视频来源")
        source = st.radio("来源", ["内置测试视频", "上传视频", "实时摄像头"],
                          label_visibility="collapsed")

        video_path: Path | int | None = None
        max_frames = 0

        if source == "内置测试视频":
            vids = list_videos()
            if vids:
                labels = [str(p.relative_to(PROJECT_ROOT / "datasets" / "videos")) for p in vids]
                pick = st.selectbox("测试视频", labels)
                video_path = PROJECT_ROOT / "datasets" / "videos" / pick
            else:
                st.warning("没有测试视频")
        elif source == "上传视频":
            up = st.file_uploader("上传 MP4", type=["mp4", "avi", "mov"])
            if up is not None:
                video_path = save_uploaded_video(up)
        else:
            # 摄像头分两种：本地 USB 索引 / 网络 RTSP 地址
            cam_kind = st.radio(
                "摄像头类型", ["本地 USB 摄像头", "网络摄像头 (RTSP)"],
                horizontal=True, key="cam_kind", label_visibility="collapsed",
            )
            if cam_kind == "本地 USB 摄像头":
                cam_idx = st.selectbox("摄像头索引", [0, 1, 2, 3], index=0,
                                       help="0 一般是默认摄像头；12V 模拟摄像头经采集卡接入后也显示为索引")
                video_path = int(cam_idx)
            else:
                # 现场已登记的摄像头直接选；也可以手填别的地址
                from config_loader import load_all as _load_all
                _cams = (_load_all().get("cameras") or {}).get("cameras") or {}
                _rtsp = {
                    f"{cid}｜{c.get('name', '')}": str(c.get("source") or "")
                    for cid, c in _cams.items()
                    if str(c.get("source") or "").startswith("rtsp://")
                }
                _rtsp["手动输入地址"] = ""
                _pick = st.selectbox("摄像头", list(_rtsp), key="rtsp_pick")
                if _pick == "手动输入地址":
                    rtsp_url = st.text_input(
                        "RTSP 地址",
                        value="",
                        placeholder="rtsp://admin:12345@192.168.1.64:554/Streaming/Channels/101",
                        help="海康 / 大华 / 宇视等网络摄像头的取流地址，含账号密码",
                    )
                else:
                    rtsp_url = _rtsp[_pick]
                    st.caption(f"`{rtsp_url}`")
                video_path = rtsp_url.strip() if rtsp_url.strip() else None
                if not video_path:
                    st.caption("填入摄像头的 RTSP 地址后才能启动")

            cam_fps = st.slider("预览帧率", 1.0, 15.0, 5.0, 1.0,
                                key="cam_fps",
                                help="越高越流畅，显卡占用也越高。5fps 已足够看清动作")
            sample_fps = float(cam_fps)
            max_frames = 0        # 摄像头持续运行，由界面按钮控制停止
            st.caption("进入后可随时用「暂停」按钮控制，不需要设帧数上限")

        st.divider()
        if video_path is not None and not isinstance(video_path, int):
            max_frames = st.number_input("最多处理帧数（0 = 全部）", 0, 5000, 0, 10,
                                         key="max_frames_file")

        # 本地索引（int）和 RTSP 地址（str）都算摄像头模式
        is_camera = isinstance(video_path, int) or (
            isinstance(video_path, str)
            and video_path.lower().startswith(("rtsp://", "http://"))
        )
        run = st.button("🚀 开始检测" if not is_camera else "📷 进入摄像头模式",
                        type="primary", width="stretch",
                        disabled=video_path is None)

        if video_path is not None:
            label = _display_name(video_path) if not is_camera else f"摄像头 {_display_name(video_path)}"
            st.caption(f"已选：`{label}`")

        # ---------- 值班看板入口 ----------
        # 看板是独立的 Streamlit 应用（工单 09 的产出物 cloud/dashboard.py），
        # 单独进程 + 独立端口，避免这里的推理被打断。
        st.divider()
        dashboard_url = os.getenv("DASHBOARD_URL", "http://localhost:8502")
        st.link_button("🚨 打开值班看板（新窗口）", dashboard_url,
                       width="stretch",
                       help="长期值守页面：所有工地的告警、推送判断结果、人工确认/误报。"
                            "需要另开一个终端运行 cloud/dashboard.py")
        st.caption(f"`{dashboard_url}`")

    # ---------------- 主体 ----------------
    render_hero(get_scene_cfg(scene_name, conf),
                running=bool(st.session_state.get('cam_running')))

    # 摄像头模式：保留三标签页结构，告警列表与报告同样可用
    if is_camera:
        # 点「进入摄像头模式」就直接开跑，不用再点一次「开始」
        if run:
            release_camera()
            st.session_state["cam_running"] = True
            st.session_state["cam_t0"] = time.time()
            st.session_state["cam_processed"] = 0
            st.session_state["cam_records"] = []
            st.session_state["cam_judge"] = None
            st.session_state["cam_alert"] = None
            st.session_state["cam_report_paths"] = None

        tab_live, tab_edge, tab_multi, tab_alerts, tab_report = st.tabs(
            ["🎬 实时监控", "⚙️ 规则引擎", "🛰️ 多路监控", "🚨 告警列表", "📄 报告查看"]
        )
        need_next = False

        with tab_edge:
            from edge_panel import render_edge_pipeline_tab
            render_edge_pipeline_tab()

        with tab_multi:
            from multi_panel import render_multi_camera_panel
            render_multi_camera_panel()

        with tab_live:
            need_next = render_camera_panel(scene_name, conf, video_path,
                                            location, fps=sample_fps,
                                            weight_path=weight_path)

        cam_records = st.session_state.get("cam_records") or []

        with tab_alerts:
            if not cam_records:
                st.markdown(
                    '<div class="empty-state"><div class="big">🚨</div>'
                    '<b>摄像头还没有告警</b>'
                    '<div class="empty-steps"><div class="empty-step">'
                    '<span class="n">i</span><b>点「进入摄像头模式」</b>'
                    '<span>确认预览画面出现</span></div>'
                    '<div class="empty-step"><span class="n">ii</span>'
                    '<b>等违规出现</b><span>未戴安全帽会自动进这个列表</span></div>'
                    '</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                render_alert_table(cam_records)

        with tab_report:
            if not cam_records:
                st.markdown(
                    '<div class="empty-state"><div class="big">📄</div>'
                    '<b>暂无报告</b><br><br>产生告警后可生成报告</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"### 摄像头 {_display_name(video_path)} · 累计 {len(cam_records)} 条告警")
                if st.button("📄 生成报告", type="primary"):
                    generate_camera_report(video_path, scene_name, conf,
                                           location, cam_records)
                last_paths = st.session_state.get("cam_report_paths")
                if last_paths:
                    st.success(f"报告已生成：`{Path(last_paths['markdown']).name}`")
                    md = Path(last_paths["markdown"]).read_text(encoding="utf-8")
                    st.download_button("⬇️ 下载 Markdown", data=md,
                                       file_name=Path(last_paths["markdown"]).name,
                                       mime="text/markdown")
                    st.divider()
                    st.markdown(md)

        if need_next:
            st.rerun()
        return

    tab_live, tab_edge, tab_multi, tab_alerts, tab_report = st.tabs(
        ["🎬 实时监控", "⚙️ 规则引擎", "🛰️ 多路监控", "🚨 告警列表", "📄 报告查看"]
    )

    with tab_edge:
        from edge_panel import render_edge_pipeline_tab
        render_edge_pipeline_tab()

    with tab_multi:
        from multi_panel import render_multi_camera_panel
        render_multi_camera_panel()

    if run and video_path is not None:
        with tab_live:
            try:
                result = run_detection(video_path, scene_name, conf, sample_fps,
                                       int(max_frames), location, weight_path,
                                       pipeline)
                st.session_state["last_result"] = result
            except Exception as e:
                st.error(f"处理失败：{type(e).__name__}: {e}")
                st.exception(e)
                return

    result = st.session_state.get("last_result")

    # ---- 实时监控 ----
    with tab_live:
        if result is None:
            st.markdown(
                '<div class="empty-state">'
                '<div class="big">🎬</div>'
                '<b>还没有运行检测</b>'
                '<div class="empty-steps">'
                '<div class="empty-step"><span class="n">1</span>'
                '<b>选场景</b><span>安全帽 / 火灾烟雾 / 值班室综合值守</span></div>'
                '<div class="empty-step"><span class="n">2</span>'
                '<b>选视频</b><span>内置测试视频，或上传自己的监控片段</span></div>'
                '<div class="empty-step"><span class="n">3</span>'
                '<b>开始检测</b><span>左键点击蓝色按钮，结果逐帧显示</span></div>'
                '</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown("")
            c1, c2, c3, c4 = st.columns(4)
            c1.info("**① 选场景**\n\n安全帽 / 火灾 / 睡岗 / 徘徊")
            c2.info("**② 选来源**\n\n测试视频 / 上传 / 摄像头")
            c3.info("**③ 开始检测**\n\n逐帧显示检测结果")
            c4.info("**④ 看结果**\n\n告警列表 + 报告")
        else:
            render_result_panel(result)

    # ---- 告警列表 ----
    with tab_alerts:
        if not result or not result["records"]:
            st.markdown(
                '<div class="empty-state">'
                '<div class="big">🚨</div>'
                '<b>本次没有告警</b><br><br>'
                '换个视频或调低置信度阈值再试'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            records = result["records"]
            # 等级汇总
            from collections import Counter
            cnt = Counter(r.get("level", "medium") for r in records)
            cols = st.columns(len(LEVEL_STYLE))
            for i, (lvl, (_, icon, label)) in enumerate(LEVEL_STYLE.items()):
                cols[i].metric(f"{icon} {label}", cnt.get(lvl, 0))

            st.markdown("")
            rows = [{
                "编号": a.get("alert_id"),
                "视频时间": a.get("video_time"),
                "事件": a.get("title"),
                "等级": a.get("level"),
                "类别": a.get("class_name"),
                "置信度": round(float(a.get("max_score", 0)), 3),
                "目标数": a.get("count", 1),
                "地点": a.get("location"),
            } for a in records]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

            snaps = [a["snapshot"] for a in records if a.get("snapshot") and Path(a["snapshot"]).exists()]
            if snaps:
                st.markdown("#### 证据截图")
                ncol = 3
                for i in range(0, len(snaps), ncol):
                    cols = st.columns(ncol)
                    for j in range(ncol):
                        if i + j < len(snaps):
                            with cols[j]:
                                st.image(snaps[i + j], caption=Path(snaps[i + j]).name,
                                         width="stretch")

    # ---- 报告查看 ----
    with tab_report:
        if not result:
            st.markdown(
                '<div class="empty-state">'
                '<div class="big">📄</div>'
                '<b>检测完成后这里会显示报告</b>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            paths = result.get("paths") or {}
            md_path = paths.get("markdown")
            json_path = paths.get("json")
            if not md_path or not Path(md_path).exists():
                st.warning("报告文件丢失，重新跑一次检测即可")
                return
            md_text = Path(md_path).read_text(encoding="utf-8")

            c1, c2 = st.columns(2)
            with c1:
                st.download_button("⬇️ 下载 Markdown", data=md_text,
                                   file_name=Path(md_path).name,
                                   mime="text/markdown", width="stretch")
            with c2:
                if json_path and Path(json_path).exists():
                    st.download_button("⬇️ 下载 JSON",
                                       data=Path(json_path).read_text(encoding="utf-8"),
                                       file_name=Path(json_path).name,
                                       mime="application/json", width="stretch")
            st.divider()
            st.markdown(md_text)


if __name__ == "__main__":
    main()
