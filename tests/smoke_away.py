"""离岗冒烟：把 ROI 挪到无人区 + 缩短阈值，验证 away 规则的 suspected/abnormal 推进。"""
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


def main() -> int:
    with open(ROOT / "config" / "cameras.yaml", encoding="utf-8") as f:
        cams = yaml.safe_load(f)
    with open(ROOT / "config" / "rules.yaml", encoding="utf-8") as f:
        rules = yaml.safe_load(f)

    cam_cfg = cams["cameras"]["Camera_03"]
    # 左上角一块空区域，保证画面里没人落进去
    cam_cfg = dict(cam_cfg)
    cam_cfg["roi"] = [{"name": "测试空区", "purpose": "away",
                       "polygon": [[0.0, 0.0], [0.15, 0.0], [0.15, 0.15], [0.0, 0.15]]}]
    # 缩短阈值：3s 疑似、6s 确认
    rules = dict(rules)
    rules["away"] = dict(rules["away"], suspect_after_sec=3, accumulate_sec=6)

    svc = ModelService(conf=0.35, kpt_conf=0.25, imgsz=640, lying_ok=True)
    pipe = EdgePipeline(cams={"Camera_03": cam_cfg}, rules=rules,
                        model_service=svc, save_snapshot=False, verbose=False)

    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / 5.0)))
    idx = 0
    last = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            vt = idx / fps
            rep = pipe.process("Camera_03", frame, video_time=vt)
            for r in rep.rule_results:
                if r.rule == "away" and r.state != last:
                    last = r.state
                    print(f"  t={vt:5.1f}s away -> {r.state}  {r.reason}")
        idx += 1
    cap.release()
    print("away 最终状态:", last)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
