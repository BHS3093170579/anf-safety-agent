"""四路混合冒烟：Camera_01/02 走现场 RTSP，Camera_03/04 走本地视频。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 10


def main() -> int:
    from config_loader import load_all
    from edge_panel import _get_pipeline

    cams = load_all()["cameras"]
    fallback = cams.get("fallback") or {}
    sources = {}
    for cid in ("Camera_01", "Camera_02", "Camera_03", "Camera_04"):
        cfg = cams["cameras"][cid]
        src = str(cfg.get("source") or "")
        if not src:
            src = fallback.get(cid, "")
        sources[cid] = src

    pipes, caps, states, alerts = {}, {}, {}, {}
    for cid, src in sources.items():
        pipes[cid], _ = _get_pipeline(cid, src, True)
        path = src if src.startswith(("rtsp://", "http://")) else str(ROOT / src)
        caps[cid] = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        print(f"{cid}: open={caps[cid].isOpened()} src={src}")
        states[cid], alerts[cid] = {}, 0
        if caps[cid].isOpened():
            for _ in range(5):
                caps[cid].read()

    t_all, n = 0.0, 0
    for i in range(FRAMES):
        for cid in sources:
            cap = caps.get(cid)
            if cap is None:
                continue
            ok, frame = cap.read()
            if not ok:
                continue
            t0 = time.perf_counter()
            rep = pipes[cid].process(cid, frame, now=i / 5.0, video_time=i / 5.0, demo=True)
            t_all += (time.perf_counter() - t0) * 1000
            n += 1
            alerts[cid] += len(rep.alerts)
            for r in rep.rule_results:
                states[cid][r.rule] = r.state

    for cid in sources:
        if caps.get(cid) is not None:
            caps[cid].release()
    print(f"\n{n} 次推理，均摊 {t_all / max(1, n):.1f} ms/帧")
    for cid in sources:
        print(f"  {cid}: 告警 {alerts[cid]} 条  {states[cid]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
