"""从 PyTorch 权重导出 TensorRT engine / ONNX。

TensorRT engine 与 GPU 型号、驱动、TRT 版本绑定，换机器后必须重新导出。

用法：
    python scripts/export_engines.py --weights weights/helmet/best.pt --imgsz 960
    python scripts/export_engines.py --weights weights/fire/best.pt --imgsz 960 --format onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="best.pt 路径")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--format", default="engine", choices=["engine", "onnx"])
    ap.add_argument("--half", action="store_true", default=True, help="FP16（仅 engine 有效）")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    w = Path(args.weights)
    if not w.exists():
        raise SystemExit(f"找不到权重文件: {w}")

    model = YOLO(str(w))
    out = model.export(
        format=args.format,
        imgsz=args.imgsz,
        half=args.half and args.format == "engine",
        device=args.device,
        opset=18,
        batch=1,
    )
    print("导出完成:", out)


if __name__ == "__main__":
    main()
