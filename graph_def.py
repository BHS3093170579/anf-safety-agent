# -*- coding: utf-8 -*-
"""
LangGraph 状态图定义 —— 工单编号：CV专高5-10（工单 02 - 任务 1/2/7）

本文件对应工单要求：
    任务 1：LangGraph 跑通 StateGraph（Node / Edge / ConditionalEdge）
    任务 2：状态 Schema（TypedDict），alerts 用 Annotated[list, operator.add] 累积
    任务 7：用 add_conditional_edges 实现动态路由

流程：
    START → video_capture → detection → judgment
                │
                ├─ has_alert → alert_agent → report_agent → END
                └─ no_alert  ─────────────────────────→ END

设计说明（为什么 Graph 只处理「单帧」）：
    工单的任务 3 描述 VideoCaptureAgent「输出帧图像 + 时间戳」，
    说明这一版是单帧处理、外层循环驱动视频。这样 Graph 结构清晰、
    易测试，也便于后续扩展成多路摄像头并行。
    跨帧的告警累积由 main.py 负责汇总，报告由 ReportAgent 统一生成。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph


# ============================================================
# 状态 Schema（工单 02 任务 2）
# ============================================================
class SecurityVideoState(TypedDict, total=False):
    """安防视频处理状态。

    工单要求的基础字段全部保留，并补充了场景与时间戳等实用字段。
    alerts 用 Annotated[list, operator.add]：多个节点先后写入时会「追加」而不是覆盖，
    这是多 Agent 场景最容易被忽略的坑。
    """

    # ---- 工单要求字段 ----
    video_path: str
    frame: np.ndarray | None
    detections: list[dict]
    alerts: Annotated[list[dict], operator.add]
    report: str

    # ---- 扩展字段 ----
    scene: str
    frame_index: int
    timestamp: float
    has_alert: bool
    alert_records: Annotated[list[dict], operator.add]


# ============================================================
# 节点工厂：把 Agent 实例注入成 LangGraph 节点
# ============================================================
def make_capture_node(video_agent, frames_iter_ref: dict):
    """VideoCaptureAgent 节点：从迭代器取下一帧放进状态。

    视频流是「有状态」的，不能把整个视频塞进 State，
    因此这里通过一个持有迭代器的引用对象逐帧取值。
    """

    def capture_node(state: SecurityVideoState) -> dict:
        try:
            packet = next(frames_iter_ref["iter"])
        except StopIteration:
            return {"frame": None, "frame_index": -1, "timestamp": 0.0}

        return {
            "frame": packet.frame,
            "frame_index": packet.index,
            "timestamp": packet.timestamp,
        }

    return capture_node


def make_detection_node(detector):
    """DetectionAgent 节点：帧 → 检测结果。"""

    def detection_node(state: SecurityVideoState) -> dict:
        frame = state.get("frame")
        if frame is None:
            return {"detections": []}
        detections = detector.detect(frame)
        return {"detections": detections}

    return detection_node


def make_judgment_node(judgment):
    """JudgmentAgent 节点：检测结果 → 是否告警。"""

    def judgment_node(state: SecurityVideoState) -> dict:
        detections = state.get("detections") or []
        timestamp = float(state.get("timestamp", 0.0))
        alerts = judgment.judge(detections, timestamp)
        return {
            "alerts": alerts,            # Annotated[operator.add] 会自动累加
            "has_alert": bool(alerts),
        }

    return judgment_node


def make_alert_node(alert_agent, video_name: str):
    """AlertAgent 节点：推送告警 + 存证据截图。"""

    def alert_node(state: SecurityVideoState) -> dict:
        new_alerts = state.get("alerts") or []
        if not new_alerts:
            return {"alert_records": []}

        frame = state.get("frame")
        records = [alert_agent.push(a, frame=frame, video_name=video_name)
                   for a in new_alerts]
        return {"alert_records": records}

    return alert_node


def make_report_node(reporter, state_ref: dict):
    """ReportAgent 节点：把本次处理的告警汇总成报告。"""

    def report_node(state: SecurityVideoState) -> dict:
        state_ref["report"] = reporter
        records = state.get("alert_records") or []
        return {"report": f"本轮新增告警 {len(records)} 条"}

    return report_node


# ============================================================
# 条件路由（工单 02 任务 7）
# ============================================================
def route_after_judgment(state: SecurityVideoState) -> str:
    """有告警走 alert → report，无告警直接 END。

    注意：判断依据用 has_alert（本帧是否新触发），
    而不是 len(state["alerts"])——因为 alerts 是累加的，
    用长度判断会在第二帧之后永远为 True。
    """
    if state.get("has_alert"):
        return "has_alert"
    return "no_alert"


def route_after_capture(state: SecurityVideoState) -> str:
    """视频读完则直接结束，避免空帧进入检测。"""
    if state.get("frame") is None:
        return "done"
    return "continue"


# ============================================================
# 构图
# ============================================================
def build_graph(scene_cfg: dict, detector, judgment, alert_agent,
                reporter, video_agent, video_name: str = ""):
    """构建并编译安防视频处理状态图。

    Returns:
        (compiled_graph, frames_iter_ref) —— 调用方靠 frames_iter_ref 重置迭代器
    """
    frames_iter_ref: dict[str, Any] = {}
    state_ref: dict[str, Any] = {}

    builder = StateGraph(SecurityVideoState)

    builder.add_node("video_capture", make_capture_node(video_agent, frames_iter_ref))
    builder.add_node("detection", make_detection_node(detector))
    builder.add_node("judgment", make_judgment_node(judgment))
    builder.add_node("alert_agent", make_alert_node(alert_agent, video_name))
    builder.add_node("report_agent", make_report_node(reporter, state_ref))

    # 基础边
    builder.add_edge(START, "video_capture")

    # 读完视频 → END；否则继续检测
    builder.add_conditional_edges(
        "video_capture",
        route_after_capture,
        {"continue": "detection", "done": END},
    )

    builder.add_edge("detection", "judgment")

    # 动态路由（条件边）
    builder.add_conditional_edges(
        "judgment",
        route_after_judgment,
        {"has_alert": "alert_agent", "no_alert": END},
    )

    builder.add_edge("alert_agent", "report_agent")
    builder.add_edge("report_agent", END)

    graph = builder.compile()
    return graph, frames_iter_ref


def new_iter_ref(video_agent) -> dict:
    """创建一个持有新迭代器的引用对象。"""
    return {"iter": video_agent.iter_frames()}
