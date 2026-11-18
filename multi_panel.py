"""多路监控页签：2~4 路相机持续跑，每路可单独选检测能力。"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import streamlit as st

from edge_panel import CAP_SCENES, _draw, _get_pipeline, _load_yaml

ROOT = Path(__file__).resolve().parent
CAMS = ("Camera_01", "Camera_02", "Camera_03", "Camera_04")

# 场景 → 视频目录
SCENE_DIR = {
    "helmet": "helmet",
    "fire": "fire",
    "sleep": "sleep",
    "away": "sleep",
    "loiter": "loiter",
}

# 每格默认放对应老项目自己的演示素材
PREFERRED = {
    "Camera_01": "demo_gongdi.mp4",            # qc\未佩戴安全帽检测
    "Camera_02": "fire_scene_test.mp4",        # anf_Agent 火灾
    "Camera_03": "tired_man_sleeping_desk.mp4",  # qc\sleep + qc\离岗检测
    "Camera_04": "loiter_user_17s.mp4",        # qc\LoiterDetection
}

# 每格对应哪个老项目
ORIGIN = {
    "Camera_01": "安全帽检测 · qc\\未佩戴安全帽检测",
    "Camera_02": "火灾分割 · anf_Agent fire",
    "Camera_03": "睡岗 + 离岗 · qc\\sleep、qc\\离岗检测",
    "Camera_04": "徘徊 · qc\\LoiterDetection",
}

# 一批跑多少帧再交出控制权，让「暂停」能点得动
BATCH = 6

# 检测跳帧：每 N 帧推理一次，其余帧复用上一次的检测框
DETECT_EVERY = 2

def _all_rtsp(cams: dict) -> list[str]:
    """现场所有可用的 RTSP 地址（去重保序）。"""
    out: list[str] = []
    for c in (cams.get("cameras") or {}).values():
        src = str(c.get("source") or "")
        if src.startswith("rtsp://") and src not in out:
            out.append(src)
    return out

def _options_for(cam_cfg: dict, fallback: dict, cid: str,
                 extra_rtsp: list[str] | None = None) -> list[str]:
    """该相机可选的源：现场所有 RTSP 排前面，其后是场景目录下的 mp4。

    `extra_rtsp` 让睡岗 / 徘徊这些格子也能直接选现场的摄像头，
    而不是只能用自己的 source（这两条规则本来就该跑在值班室、门口的实时流上）。
    """
    out: list[str] = []
    own = str(cam_cfg.get("source") or "")
    if own.startswith("rtsp://"):
        out.append(own)
    for src in extra_rtsp or []:
        if src not in out:
            out.append(src)

    videos: list[Path] = []
    for scene in cam_cfg.get("scene") or []:
        sub = SCENE_DIR.get(scene)
        if not sub:
            continue
        d = ROOT / "datasets" / "videos" / sub
        if d.is_dir():
            videos.extend(sorted(d.glob("*.mp4")))
    fb = fallback.get(cid)
    if fb:
        p = ROOT / fb
        if p not in videos:
            videos.insert(0, p)
    # 去重保序
    seen = {Path(p).name for p in out if not p.startswith("rtsp://")}
    for p in videos:
        if p.name in seen:
            continue
        seen.add(p.name)
        out.append(str(p.relative_to(ROOT)))
    return out

def _resolve(src: str) -> str:
    """RTSP 原样返回，本地文件拼成绝对路径。"""
    return src if src.startswith(("rtsp://", "http://")) else str(ROOT / src)

def _read_frame(cap: cv2.VideoCapture, src: str):
    """读一帧；本地视频读到头就回到第一帧，RTSP 断了返回 None。"""
    ok, frame = cap.read()
    if ok:
        return frame
    if src.startswith(("rtsp://", "http://")):
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = cap.read()
    return frame if ok else None

def _stop(reason: str = "") -> None:
    """释放所有流，写报告（只落文件，展示交给下一轮脚本）。"""
    st_ = st.session_state.get("multi") or {}
    for cap in (st_.get("caps") or {}).values():
        try:
            cap.release()
        except Exception:
            pass
    st_["caps"] = {}
    st_["running"] = False
    if st_.get("i", 0) > 0:
        info = _write_multi_report(
            st_.get("sources") or {}, st_.get("i", 0),
            st_.get("alerts") or {}, st_.get("states") or {},
            tuple(st_.get("active") or CAMS), reason,
        )
        st_["last_report"] = info

def render_multi_camera_panel() -> None:
    cams, _ = _load_yaml()
    fallback = cams.get("fallback") or {}
    st_ = st.session_state.setdefault("multi", {})
    running = bool(st_.get("running"))

    st.markdown("#### 🛰️ 多路并行监控")
    st.caption("持续运行；每路可单独选检测能力。本地视频播完自动循环，RTSP 断流只停那一路。")

    n = st.slider("同时开几路", 2, 4, int(st_.get("n", 4)), step=1,
                  key="multi_n", disabled=running)
    active = CAMS[:n]
    st_["n"] = n
    st_["active"] = list(active)

    cols = st.columns(len(active))
    sources: dict[str, str] = {}
    caps_choice: dict[str, str] = {}
    placeholders: dict = {}
    status: dict = {}
    rtsp_all = _all_rtsp(cams)

    for col, cid in zip(cols, active):
        cfg = cams["cameras"][cid]
        with col:
            st.markdown(f"**{cid}**")
            st.caption(ORIGIN.get(cid, ""))
            opts = _options_for(cfg, fallback, cid, extra_rtsp=rtsp_all)
            if not opts:
                st.warning("没有可用视频")
                continue
            want = PREFERRED.get(cid)
            idx = next((i for i, p in enumerate(opts) if Path(p).name == want), 0)
            sources[cid] = st.selectbox(
                "视频源", opts, index=idx, key=f"multi_src_{cid}",
                label_visibility="collapsed", disabled=running,
            )
            caps_choice[cid] = st.selectbox(
                "检测能力", list(CAP_SCENES), index=0, key=f"multi_cap_{cid}",
                help="选「安全帽检测」可以让两路摄像头都只跑安全帽。",
                disabled=running,
            )
            placeholders[cid] = st.empty()
            status[cid] = st.empty()
            placeholders[cid].info("待启动")

    b1, b2, b3 = st.columns([1, 1, 2])
    start = b1.button("▶ 开始监控", type="primary", width="stretch",
                      disabled=running)
    stop = b2.button("⏸ 暂停", width="stretch", disabled=not running)
    b3.caption(
        f"已运行 {st_.get('i', 0)} 帧 · 累计告警 "
        f"{sum((st_.get('alerts') or {}).values())} 条"
        if running else "点「开始监控」后持续运行，随时可暂停。"
    )

    if stop:
        _stop("手动暂停")
        st.rerun()

    if start:
        pipes, caps, cfgs = {}, {}, {}
        try:
            for cid in active:
                src = sources[cid]
                pipes[cid], cfgs[cid] = _get_pipeline(
                    cid, src, True, caps_choice[cid],
                )
                caps[cid] = cv2.VideoCapture(_resolve(src), cv2.CAP_FFMPEG)
                if not caps[cid].isOpened():
                    st.error(f"{cid} 打不开 {src}")
                    for c in caps.values():
                        c.release()
                    return
        except Exception as e:  # 模型加载失败不要留半截句柄
            for c in caps.values():
                c.release()
            st.error(f"初始化失败：{e}")
            return
        st_.update({
            "running": True, "caps": caps, "pipes": pipes, "cfgs": cfgs,
            "sources": {cid: sources[cid] for cid in active},
            "alerts": {cid: 0 for cid in active},
            "states": {cid: {} for cid in active},
            "i": 0, "t0": time.time(),
        })
        st.rerun()

    if not running:
        # 暂停态：显示上一次的统计与报告
        if st_.get("i", 0) > 0:
            for cid in (st_.get("active") or []):
                st_cid = (st_.get("states") or {}).get(cid) or {}
                bad = [k for k, v in st_cid.items() if v != "normal"]
                status[cid].success(
                    f"告警 {(st_.get('alerts') or {}).get(cid, 0)} 条 · "
                    + ("、".join(f"{k}={st_cid[k]}" for k in bad)
                       if bad else "全部正常")
                )
        info = st_.get("last_report")
        if info:
            _show_report(info)
        return

    # ---------- 持续跑（fragment 局部刷新，避免整页重绘闪烁） ----------
    @st.fragment(run_every=0.2)
    def _run_loop() -> None:
        st_ = st.session_state.get("multi") or {}
        if not st_.get("running"):
            st.rerun(scope="app")
            return
        pipes = st_.get("pipes") or {}
        caps = st_.get("caps") or {}
        cfgs = st_.get("cfgs") or {}
        alerts = st_.get("alerts") or {}
        states = st_.get("states") or {}

        for _ in range(BATCH):
            if not st_.get("running"):
                break
            i = int(st_.get("i", 0))
            for cid in active:
                cap = caps.get(cid)
                if cap is None:
                    continue
                frame = _read_frame(cap, (st_.get("sources") or {}).get(cid, ""))
                if frame is None:
                    cap.release()
                    caps.pop(cid, None)
                    continue
                vt = i / 5.0
                # 检测跳帧：每 2 帧推理一次，中间帧复用上一次的框，
                # 画面刷新更快、GPU 负载减半。
                cached = st_.setdefault("last_rep", {})
                if i % DETECT_EVERY == 0 or cid not in cached:
                    try:
                        rep = pipes[cid].process(cid, frame, now=vt,
                                                 video_time=vt, demo=True)
                    except Exception as e:
                        status[cid].error(f"推理失败：{e}")
                        caps.pop(cid, None)
                        continue
                    cached[cid] = rep
                    alerts[cid] = alerts.get(cid, 0) + len(rep.alerts)
                    for a in rep.alerts:
                        recs = st_.setdefault("records", [])
                        recs.append({
                            "cam": cid,
                            "type": a.get("violation_type", ""),
                            "track": a.get("track_id", 0),
                            "snap": a.get("snapshot_path") or "",
                            "msg": a.get("message", ""),
                            "t": time.strftime("%H:%M:%S"),
                        })
                        if len(recs) > 50:
                            del recs[:-50]
                    for r in rep.rule_results:
                        states.setdefault(cid, {})[r.rule] = r.state
                else:
                    rep = cached[cid]
                vis = _draw(frame, rep, cfgs[cid])
                placeholders[cid].image(vis, channels="BGR", width="stretch")
            st_["i"] = i + 1

        st_.update({"alerts": alerts, "states": states})
        for cid in active:
            bad = [k for k, v in (states.get(cid) or {}).items() if v != "normal"]
            status[cid].success(
                f"告警 {alerts.get(cid, 0)} 条 · "
                + ("、".join(f"{k}={v}" for k, v in (states.get(cid) or {}).items())
                   if bad else "全部正常")
            )

        # 最近告警：带抓拍图，能直接看清是谁、什么违规
        recs = (st_.get("records") or [])[-8:]
        st.markdown("##### 🚨 最近告警")
        if not recs:
            st.caption("还没有告警。规则触发时会在这里列出抓拍图。")
            return
        for row in reversed(recs):
            c1, c2 = st.columns([1, 4])
            snap = row.get("snap") or ""
            with c1:
                if snap and Path(snap).exists():
                    st.image(snap, width="stretch")
                else:
                    st.caption("无截图")
            with c2:
                st.markdown(
                    f"**{row['t']} · {row['cam']} · {row['type']}**  "
                    f"`ID{row['track']}`"
                )
                if row.get("msg"):
                    st.caption(row["msg"])
                if snap:
                    st.caption(f"`{snap}`")

    _run_loop()

def _write_multi_report(sources: dict, frames: int, alerts: dict,
                        states: dict, cams: tuple[str, ...] = CAMS,
                        reason: str = "") -> dict:
    """多路监控暂停时写一份 Markdown 汇总，返回 {"text", "name"}。"""
    from datetime import datetime

    out_dir = ROOT / "outputs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    path = out_dir / f"multi_{now:%Y%m%d_%H%M%S}.md"

    rows = []
    for cid in cams:
        st_ = states.get(cid) or {}
        bad = "、".join(f"{k}={st_[k]}" for k in st_ if st_[k] != "normal") or "全部正常"
        rows.append(f"| {cid} | `{sources.get(cid, '')}` | {alerts.get(cid, 0)} | {bad} |")

    text = (
        f"# 多路并行监控报告\n\n"
        f"- 生成时间：{now:%Y-%m-%d %H:%M:%S}\n"
        + (f"- 结束原因：{reason}\n" if reason else "")
        + f"- 路数：{len(cams)}\n"
        f"- 已处理帧数：{frames}\n"
        f"- 总告警：{sum(alerts.values())} 条\n\n"
        f"| 相机 | 视频源 | 告警数 | 触发规则 |\n|---|---|---|---|\n"
        + "\n".join(rows) + "\n"
    )
    path.write_text(text, encoding="utf-8")
    return {"text": text, "name": path.name}

def _show_report(info: dict) -> None:
    st.success(f"报告已生成：`{info['name']}`（outputs/reports）")
    st.caption("告警已实时落盘 `outputs/events/edge_alerts.jsonl`；配了 webhook 会同步推送到钉钉/企业微信。")
    with st.expander("📄 查看报告内容", expanded=True):
        st.markdown(info["text"])
    st.download_button(
        "⬇️ 下载报告", data=info["text"], file_name=info["name"],
        mime="text/markdown",
    )
