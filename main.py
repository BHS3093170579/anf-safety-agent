# -*- coding: utf-8 -*-
"""
安防智能体 V1.0 · 主程序 ——

用法：
    # 用默认场景（helmet）跑测试视频
    python main.py --video datasets/videos/helmet/construction_site_test.mp4

    # 指定场景
    python main.py --scene fire --video datasets/videos/fire/Arson012_x264.mp4

    # 只跑前 30 个采样帧（快速验证）
    python main.py --video xxx.mp4 --max-frames 30

    # 覆盖置信度阈值
    python main.py --video xxx.mp4 --conf 0.45

环境要求：
    必须在 smoke_trt 环境运行（TensorRT 10.16）
    E:\\ANACONDA\\envs\\smoke_trt\\python.exe main.py --video ...
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config_loader import get_output_dirs, get_scene, load_config  # noqa: E402
from graph_def import build_graph  # noqa: E402

from agents.alert import AlertAgent  # noqa: E402
from agents.detector import DetectionAgent  # noqa: E402
from agents.judgment import JudgmentAgent  # noqa: E402
from agents.reporter import ReportAgent  # noqa: E402
from agents.video_capture import VideoCaptureAgent  # noqa: E402

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="安防智能体 V1.0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--scene", default="helmet", help="场景名（见 config/scenes.yaml）")
    p.add_argument("--video", default="", help="输入视频路径")
    p.add_argument("--camera", type=int, default=None,
                   help="摄像头索引（0 = 默认摄像头），与 --video 二选一")
    p.add_argument("--rtsp", default="",
                   help="RTSP 网络摄像头地址，如 rtsp://user:pass@ip:554/stream")
    p.add_argument("--config", default="", help="配置文件路径，默认 config/scenes.yaml")
    p.add_argument("--location", default="", help="摄像头点位名（写入告警地点）")
    p.add_argument("--conf", type=float, default=None, help="覆盖配置里的置信度阈值")
    p.add_argument("--fps", type=float, default=None, help="抽帧帧率，默认取配置")
    p.add_argument("--max-frames", type=int, default=0,
                   help="最多处理多少采样帧，0 不限制（摄像头建议设 0 由 Ctrl+C 停止）")
    p.add_argument("--cam-width", type=int, default=1280, help="摄像头期望宽度")
    p.add_argument("--cam-height", type=int, default=720, help="摄像头期望高度")
    p.add_argument("--no-snapshot", action="store_true", help="不保存告警截图")
    return p.parse_args()

def banner() -> None:
    print("=" * 68)
    print("  安防智能体 V1.0 · 多智能体视频分析")
    print("  LangGraph + Ultralytics + TensorRT")
    print("=" * 68)

def main() -> int:
    args = parse_args()
    banner()

    # ---------- 加载配置 ----------
    cfg = load_config(args.config or None)
    scene = get_scene(cfg, args.scene)
    if args.conf is not None:
        scene["conf"] = float(args.conf)

    out_dirs = get_output_dirs(cfg)
    sampling = cfg.get("sampling", {})

    # ---------- 确定输入源：摄像头 / RTSP / 视频文件 ----------
    if args.camera is not None:
        source = args.camera
    elif args.rtsp:
        source = args.rtsp
    elif args.video:
        source = args.video
    else:
        print("[错误] 必须指定输入源：--video <路径> 或 --camera <索引> 或 --rtsp <地址>")
        return 2

    # 抽帧率优先级：命令行 > 场景单独配置 > 全局配置
    scene_fps = scene.get("sample_fps")
    # 摄像头是无限流，max_frames 不设就是不限制（靠 Ctrl+C 停止）
    max_frames = args.max_frames or (sampling.get("max_frames", 0) if args.camera is None and not args.rtsp else 0)

    # args.fps = 0 是有效值（表示逐帧不抽帧），所以不能用 or 判断
    if args.fps is not None:
        fps_arg = float(args.fps)
    else:
        fps_arg = scene_fps or sampling.get("fps", 1.0)

    video_agent = VideoCaptureAgent(
        source,
        sample_fps=fps_arg,
        max_frames=max_frames,
        camera_width=args.cam_width,
        camera_height=args.cam_height,
    )
    info = video_agent.info()

    print(f"\n场景     : {scene['display_name']}  ({scene['name']})")
    print(f"模型     : {Path(scene['model']).name}  task={scene['task']}  "
          f"imgsz={scene['imgsz']}  conf={scene['conf']}")
    if info["source_type"] == "camera":
        print(f"输入源   : 摄像头 {source}")
        print(f"信息     : {info['resolution']}  实时流")
        print(f"抽帧     : {info['sample_fps']} fps"
              + (f"，最多 {max_frames} 帧" if max_frames else "，按 Ctrl+C 停止"))
    else:
        print(f"视频     : {Path(info['path']).name}")
        print(f"信息     : {info['resolution']}  {info['duration_sec']}s  "
              f"原始 {info['source_fps']} fps")
        if info.get("full_frame_mode"):
            print(f"抽帧     : 逐帧全速（不抽帧），共 {info['expected_samples']} 帧")
        else:
            print(f"抽帧     : {info['sample_fps']} fps（每 {info['frame_step']} 帧取 1 帧），"
                  f"预计 {info['expected_samples']} 帧")
    print("-" * 68)

    t_build = time.time()
    detector = DetectionAgent(scene)
    judgment = JudgmentAgent(scene)
    alert_agent = AlertAgent(
        scene,
        snapshot_dir=out_dirs["snapshots"],
        location=args.location,
    )
    reporter = ReportAgent(out_dirs["reports"], scene)
    print(f"[初始化] 模型加载 + 预热完成，耗时 {time.time() - t_build:.1f}s\n")

    # ---------- 构建 Graph ----------
    graph, iter_ref = build_graph(
        scene, detector, judgment, alert_agent, reporter, video_agent,
        video_name=Path(args.video).name,
    )

    # ---------- 逐帧驱动 ----------
    all_records: list[dict] = []
    processed = 0
    t0 = time.time()

    # 每次运行都从视频头开始（build_graph 里的迭代器可能已被消费）
    iter_ref["iter"] = video_agent.iter_frames()

    while True:
        result = graph.invoke({
            "video_path": str(args.video),
            "scene": scene["name"],
            "alerts": [],           # 每轮清零，跨帧累积由 main 负责
            "alert_records": [],
        })

        idx = result.get("frame_index", -1)
        if idx is None or idx < 0:
            break

        processed += 1
        records = result.get("alert_records") or []
        all_records.extend(records)

        if processed % 20 == 0:
            print(f"  … 已处理 {processed} 帧，累计告警 {len(all_records)} 条")

    elapsed = time.time() - t0

    # ---------- 生成报告 ----------
    perf = detector.stats()
    perf["elapsed_sec"] = round(elapsed, 2)
    info_out = dict(info)
    info_out["processed_frames"] = processed

    report = reporter.build(info_out, all_records, perf)
    paths = reporter.save(report, stem=f"{scene['name']}_{Path(args.video).stem}")

    # ---------- 终端汇总 ----------
    print("\n" + "=" * 68)
    print("  处理完成")
    print("=" * 68)
    print(f"  处理帧数   : {processed}")
    print(f"  端到端耗时 : {elapsed:.1f} 秒")
    print(f"  检测耗时   : 中位 {perf.get('median_ms', 0):.1f} ms / "
          f"P95 {perf.get('p95_ms', 0):.1f} ms")
    print(f"  等效 FPS   : {perf.get('fps', 0):.1f}")
    print(f"  告警总数   : {len(all_records)}")
    if all_records:
        by_type: dict[str, int] = {}
        for a in all_records:
            by_type[a.get("event_type", "?")] = by_type.get(a.get("event_type", "?"), 0) + 1
        print(f"  告警分布   : {by_type}")
    print(f"  报告       : {paths['markdown']}")
    print(f"  截图目录   : {out_dirs['snapshots']}")
    print("=" * 68)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
