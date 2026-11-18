"""五能力并行联调：四路相机各跑 30 帧，共享模型池，检查互不干扰。"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge.model_service import ModelService  # noqa: E402
from edge.pipeline import EdgePipeline  # noqa: E402

FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 30


def main() -> int:
    with open(ROOT / "config" / "cameras.yaml", encoding="utf-8") as f:
        cams = yaml.safe_load(f)
    with open(ROOT / "config" / "rules.yaml", encoding="utf-8") as f:
        rules = yaml.safe_load(f)
    with open(ROOT / "config" / "scenes.yaml", encoding="utf-8") as f:
        scenes = yaml.safe_load(f)["scenes"]

    cams_cfg = cams["cameras"]
    fallback = cams.get("fallback") or {}

    svc = ModelService(conf=0.35, kpt_conf=0.25, imgsz=640, lying_ok=True)

    from agents.detector import DetectionAgent

    detectors = {}
    for name in ("helmet", "fire"):
        if name in scenes:
            cfg = dict(scenes[name])
            cfg.setdefault("name", name)
            detectors[name] = DetectionAgent(cfg)

    pipe = EdgePipeline(
        cams={cid: cfg for cid, cfg in cams_cfg.items() if cid in ("Camera_01", "Camera_02", "Camera_03", "Camera_04")},
        rules=rules,
        model_service=svc,
        detectors=detectors,
        save_snapshot=False,
        verbose=False,
    )

    caps, idxs = {}, {}
    for cid, cfg in cams_cfg.items():
        if cid not in pipe.engines:
            continue
        src = fallback.get(cid) or cfg.get("source") or ""
        src = Path(src) if not str(src).startswith(("rtsp", "http")) else src
        p = (ROOT / src) if isinstance(src, Path) else src
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            print(f"[skip] {cid} 打不开 {p}")
            continue
        caps[cid] = cap
        idxs[cid] = 0

    triggered: dict[str, dict[str, str]] = {}
    alerts: dict[str, int] = {}
    errors = 0

    for _ in range(FRAMES):
        for cid, cap in list(caps.items()):
            ok, frame = cap.read()
            if not ok:
                caps.pop(cid, None)
                continue
            vt = idxs[cid] / 5.0
            try:
                rep = pipe.process(cid, frame, now=vt, video_time=vt, demo=True)
            except Exception as exc:  # noqa: BLE001
                errors += 1
                print(f"[ERR] {cid} t={vt:.1f} {type(exc).__name__}: {exc}")
                continue
            for r in rep.rule_results:
                st = triggered.setdefault(cid, {})
                if st.get(r.rule) != r.state:
                    st[r.rule] = r.state
                    if r.state != "normal":
                        print(f"  {cid} t={vt:5.1f}s {r.rule:9s} -> {r.state}  {r.reason}")
            alerts[cid] = alerts.get(cid, 0) + len(rep.alerts)
            idxs[cid] += 1

    for cap in caps.values():
        cap.release()

    print("\n各相机最终规则状态：")
    for cid in sorted(triggered):
        print(f"  {cid}: {triggered[cid]}")
    print("各相机上报告警数：", alerts)
    print("异常次数：", errors)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
