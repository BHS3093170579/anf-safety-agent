"""睡岗 + 离岗冒烟：Camera_03 走 EdgePipeline，看 sleep / away 规则是否推进。"""
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
    ROOT / "datasets" / "videos" / "sleep" / "demo.mp4")

if not VIDEO.exists():
    print(f"[SKIP] 缺少演示视频: {VIDEO}（见 datasets/videos/README.md）")
    raise SystemExit(0)

def load_cfg() -> tuple[dict, dict]:
    with open(ROOT / "config" / "cameras.yaml", encoding="utf-8") as f:
        cams = yaml.safe_load(f)
    with open(ROOT / "config" / "rules.yaml", encoding="utf-8") as f:
        rules = yaml.safe_load(f)
    return cams, rules

def main() -> int:
    cams, rules = load_cfg()
    cam_cfg = cams["cameras"]["Camera_03"]

    svc = ModelService(conf=0.35, kpt_conf=0.25, imgsz=640, lying_ok=True)
    pipe = EdgePipeline(cams={"Camera_03": cam_cfg}, rules=rules,
                        model_service=svc, save_snapshot=False, verbose=False)

    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / 5.0)))          # 抽到 5fps
    idx = 0
    seen: dict[str, str] = {}
    alerts = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            video_time = idx / fps
            rep = pipe.process("Camera_03", frame, video_time=video_time, demo=True)
            for r in rep.rule_results:
                key = f"{r.rule}"
                if r.rule == "sleep" and r.detail:
                    print(f"  t={video_time:5.1f}s sleep frames={r.detail.get('frames')} "
                          f"drop={r.detail.get('head_drop')} state={r.state}")
                if seen.get(key) != r.state:
                    seen[key] = r.state
                    if r.state != "normal":
                        print(f"  t={video_time:5.1f}s {r.rule:8s} -> {r.state}  {r.reason}")
                if r.state == "abnormal" and r.rule in ("sleep", "away"):
                    alerts += 1
        idx += 1
    cap.release()

    print("规则最终状态:", seen)
    print("sleep/away 告警帧数:", alerts)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
