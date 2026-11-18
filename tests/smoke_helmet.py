"""三路安全帽检测器的冒烟测试：同一帧分别走 single / two_stage / pose_rule。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.detector import DetectionAgent  # noqa: E402
from capabilities.helmet import HelmetCascadeDetector  # noqa: E402
from config_loader import get_scene, load_config  # noqa: E402

VIDEO = ROOT / "datasets" / "videos" / "helmet" / "demo.mp4"

if not VIDEO.exists():
    print(f"[SKIP] 缺少演示视频: {VIDEO}（见 datasets/videos/README.md）")
    raise SystemExit(0)

def grab(seconds: float):
    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"读不到视频帧：{VIDEO}")
    return frame

def main() -> int:
    scene = get_scene(load_config(), "helmet")
    scene["conf"] = 0.08          # 与 get_detector 一致：推理下限压低，后处理再卡
    scene["user_conf"] = 0.40

    frame = grab(5.0)
    print(f"frame={frame.shape}")

    for pipeline in ("single", "two_stage", "pose_rule"):
        det = HelmetCascadeDetector(scene, pipeline=pipeline, device="0")
        det.detect(frame)                     # 预热
        t0 = time.perf_counter()
        out = det.detect(frame)
        ms = (time.perf_counter() - t0) * 1000
        brief = [(d["class"], round(float(d["score"]), 3)) for d in out[:8]]
        print(f"{pipeline:10s} boxes={len(out):2d}  {ms:6.1f} ms  {brief}")

    # 回归：默认的 YOLO 检测器仍然可用
    base = DetectionAgent(scene)
    t0 = time.perf_counter()
    out = base.detect(frame)
    print(f"{'yolo(engine)':10s} boxes={len(out):2d}  "
          f"{(time.perf_counter() - t0) * 1000:6.1f} ms")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
