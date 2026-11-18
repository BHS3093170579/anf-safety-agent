"""RTSP 实拍冒烟：两路网络摄像头各跑 N 帧，测端到端耗时。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 30


def main() -> int:
    from config_loader import load_all
    from edge_panel import _get_pipeline

    cams = load_all()["cameras"]
    for cid in ("Camera_01", "Camera_02"):
        src = cams["cameras"][cid]["source"]
        if not src.startswith("rtsp://"):
            print(f"{cid} 没有 RTSP 地址：{src!r}")
            continue

        pipe, cam_cfg = _get_pipeline(cid, src, True)
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"{cid} 打不开 {src}")
            continue

        print(f"\n=== {cid} {src} ===")
        print(f"  分辨率 {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
              f"@ {cap.get(cv2.CAP_PROP_FPS):.0f}fps")

        # 丢前 5 帧，等解码器稳定
        for _ in range(5):
            cap.read()

        t_all, n, ok_n = 0.0, 0, 0
        alerts = 0
        last = {}
        for i in range(FRAMES):
            ok, frame = cap.read()
            if not ok:
                print("  读帧失败，停止")
                break
            t0 = time.perf_counter()
            rep = pipe.process(cid, frame, now=i / 5.0, video_time=i / 5.0, demo=True)
            dt = (time.perf_counter() - t0) * 1000
            t_all += dt
            n += 1
            ok_n += 1
            alerts += len(rep.alerts)
            for r in rep.rule_results:
                last[r.rule] = r.state
            if i % 10 == 0:
                print(f"  帧 {i:3d}: {dt:5.1f} ms  框 {len(rep.tracks)}  {last}")
        cap.release()
        if n:
            print(f"  合计 {n} 帧，均值 {t_all / n:.1f} ms/帧（{1000 / (t_all / n):.1f} FPS），告警 {alerts} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
