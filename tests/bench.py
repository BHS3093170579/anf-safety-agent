"""性能基准：单模型 vs 多能力叠加 vs 四路并行。"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.detector import DetectionAgent  # noqa: E402
from edge.model_service import ModelService  # noqa: E402
from edge.pipeline import EdgePipeline  # noqa: E402

VIDEOS = ROOT / "datasets" / "videos"


def load_yaml(name: str) -> dict:
    with open(ROOT / "config" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def grab_frames(path: Path, n: int) -> list:
    cap = cv2.VideoCapture(str(path))
    out = []
    while len(out) < n:
        ok, frame = cap.read()
        if not ok:
            break
        out.append(frame)
    cap.release()
    return out


def bench(fn, frames: list, warmup: int = 5) -> tuple[float, float]:
    for f in frames[:warmup]:
        fn(f)
    ts = []
    for f in frames:
        t0 = time.perf_counter()
        fn(f)
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts), sum(ts) / len(ts)


def main() -> int:
    scenes = load_yaml("scenes.yaml")["scenes"]
    cams = load_yaml("cameras.yaml")

    print("加载模型...", flush=True)
    svc = ModelService(conf=0.35, kpt_conf=0.25, imgsz=640, lying_ok=True)
    detectors = {}
    for name in ("helmet", "fire"):
        cfg = dict(scenes[name])
        cfg.setdefault("name", name)
        detectors[name] = DetectionAgent(cfg)

    f_helmet = grab_frames(VIDEOS / "helmet" / "demo.mp4", 60)
    f_sleep = grab_frames(VIDEOS / "sleep" / "demo.mp4", 60)
    f_fire = grab_frames(VIDEOS / "fire" / "demo.mp4", 60)
    f_loiter = grab_frames(VIDEOS / "loiter" / "demo.mp4", 60)
    if not (f_helmet and f_sleep and f_fire and f_loiter):
        print("[SKIP] 缺少演示视频，先按 datasets/videos/README.md 准备素材")
        return 0

    cam_cfg = cams["cameras"]

    def mk(cid: str) -> EdgePipeline:
        return EdgePipeline(
            cams={cid: cam_cfg[cid]},
            rules=load_yaml("rules.yaml"),
            model_service=svc,
            detectors=detectors,
            save_snapshot=False,
            verbose=False,
        )

    p1, p3, p4 = mk("Camera_01"), mk("Camera_03"), mk("Camera_04")

    rows = []

    med, avg = bench(lambda f: svc.process("Camera_03", f), f_sleep)
    rows.append(("pose 单模型（睡岗视频）", med, avg))

    med, avg = bench(lambda f: detectors["helmet"].detect(f), f_helmet)
    rows.append(("helmet engine 单模型", med, avg))

    med, avg = bench(lambda f: detectors["fire"].detect(f), f_fire)
    rows.append(("fire engine 单模型", med, avg))

    t = 0.0
    for i, f in enumerate(f_sleep):
        t0 = time.perf_counter()
        p3.process("Camera_03", f, now=i / 5.0, video_time=i / 5.0, demo=True)
        t += time.perf_counter() - t0
    rows.append(("Camera_03 全链路（pose+睡岗/离岗）", t / len(f_sleep) * 1000, t / len(f_sleep) * 1000))

    t = 0.0
    for i, f in enumerate(f_loiter):
        t0 = time.perf_counter()
        p4.process("Camera_04", f, now=i / 5.0, video_time=i / 5.0, demo=True)
        t += time.perf_counter() - t0
    rows.append(("Camera_04 全链路（pose+徘徊）", t / len(f_loiter) * 1000, t / len(f_loiter) * 1000))

    t = 0.0
    for i, f in enumerate(f_helmet):
        t0 = time.perf_counter()
        p1.process("Camera_01", f, now=i / 5.0, video_time=i / 5.0, demo=True)
        t += time.perf_counter() - t0
    rows.append(("Camera_01 全链路（pose+helmet+fire 三模型）", t / len(f_helmet) * 1000, t / len(f_helmet) * 1000))

    # 四路并行
    p_all = EdgePipeline(
        cams={cid: cam_cfg[cid] for cid in ("Camera_01", "Camera_02", "Camera_03", "Camera_04")},
        rules=load_yaml("rules.yaml"),
        model_service=svc,
        detectors=detectors,
        save_snapshot=False,
        verbose=False,
    )
    caps = {}
    for cid in ("Camera_01", "Camera_02", "Camera_03", "Camera_04"):
        src = cams["fallback"][cid]
        caps[cid] = cv2.VideoCapture(str(ROOT / src))
    idx = {cid: 0 for cid in caps}
    t = 0.0
    n = 0
    for _ in range(40):
        for cid, cap in list(caps.items()):
            ok, frame = cap.read()
            if not ok:
                caps.pop(cid, None)
                continue
            t0 = time.perf_counter()
            p_all.process(cid, frame, now=idx[cid] / 5.0, video_time=idx[cid] / 5.0, demo=True)
            t += time.perf_counter() - t0
            idx[cid] += 1
            n += 1
    for cap in caps.values():
        cap.release()
    rows.append((f"四路并行（{n} 帧总耗时/{n}）", t / n * 1000, t / n * 1000))

    print(f"\n{'测试项':46s} {'中位 ms':>9s} {'均值 ms':>9s} {'等效 FPS':>9s}")
    for name, med, avg in rows:
        print(f"{name:46s} {med:9.1f} {avg:9.1f} {1000.0 / avg:9.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
