"""三路安全帽检测的视频冒烟：逐帧跑 demo，统计耗时与违规框。

只验证「能跑、不崩、结果合理」，不做精度评估（精度用 test 集指标）。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from capabilities.helmet import HelmetCascadeDetector  # noqa: E402
from config_loader import get_scene, load_config  # noqa: E402

VIDEO = ROOT / "datasets" / "videos" / "helmet" / "demo.mp4"

if not VIDEO.exists():
    print(f"[SKIP] 缺少演示视频: {VIDEO}（见 datasets/videos/README.md）")
    raise SystemExit(0)


def main() -> int:
    scene = get_scene(load_config(), "helmet")
    scene["conf"] = 0.08
    scene["user_conf"] = 0.40

    cap = cv2.VideoCapture(str(VIDEO))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    print(f"视频帧数：{len(frames)}")

    for pipeline in ("single", "two_stage", "pose_rule"):
        det = HelmetCascadeDetector(scene, pipeline=pipeline, device="0")
        det.detect(frames[0])                    # 预热
        total = 0.0
        violations = 0
        frames_with = 0
        for frame in frames:
            t0 = time.perf_counter()
            out = det.detect(frame)
            total += (time.perf_counter() - t0) * 1000
            n = sum(1 for d in out if d["class"] == "no-hardhat")
            violations += n
            frames_with += 1 if n else 0
        print(f"{pipeline:10s} 平均 {total / len(frames):6.1f} ms/帧  "
              f"违规框合计 {violations:3d}  有违规的帧 {frames_with}/{len(frames)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
