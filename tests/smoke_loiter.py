"""徘徊冒烟：Camera_04 走 EdgePipeline，看 loiter 规则是否推进。"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge.model_service import ModelService  # noqa: E402
from edge.pipeline import EdgePipeline  # noqa: E402

VIDEO = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    ROOT / "datasets" / "videos" / "loiter" / "demo.mp4")

if not VIDEO.exists():
    print(f"[SKIP] 缺少演示视频: {VIDEO}（见 datasets/videos/README.md）")
    raise SystemExit(0)

def main() -> int:
    with open(ROOT / "config" / "cameras.yaml", encoding="utf-8") as f:
        cams = yaml.safe_load(f)
    with open(ROOT / "config" / "rules.yaml", encoding="utf-8") as f:
        rules = yaml.safe_load(f)

    cam_cfg = cams["cameras"]["Camera_04"]
    svc = ModelService(conf=0.35, kpt_conf=0.25, imgsz=640)
    pipe = EdgePipeline(cams={"Camera_04": cam_cfg}, rules=rules,
                        model_service=svc, save_snapshot=False, verbose=False)

    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / 5.0)))
    idx = 0
    last = None
    alerts = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            vt = idx / fps
            rep = pipe.process("Camera_04", frame, now=vt, video_time=vt, demo=True)
            for r in rep.rule_results:
                if r.rule != "loiter":
                    continue
                if r.state != last:
                    last = r.state
                    print(f"  t={vt:5.1f}s loiter -> {r.state}  {r.reason}  {r.detail}")
                if r.state == "abnormal":
                    alerts += 1
        idx += 1
    cap.release()
    print("loiter 最终状态:", last)
    print("abnormal 帧数:", alerts)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
