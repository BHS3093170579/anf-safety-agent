# 模型权重

权重不进仓库，按下面的目录放置后程序才能跑。

```
weights/
├── helmet/
│   ├── best.pt          # 必需：YOLO11s 安全帽检测，960
│   ├── best.engine      # 可选：TensorRT FP16，按本机 GPU 导出
│   └── best.onnx
├── fire/
│   ├── best.pt          # 必需：YOLO 实例分割，960（火焰 / 烟雾两类）
│   ├── best.engine
│   └── best.onnx
├── pose/
│   └── yolov8n-pose.pt  # 必需：睡岗、离岗、徘徊共用
└── helmet_qc/           # 可选：两步法与姿态兜底
    ├── best.pt
    ├── head_det.pt
    ├── head_cls.pt
    └── yolov8n-pose.pt
```

## 获取方式

权重不随仓库发布，按自己的情况准备：

- **自己训练**：安全帽用 YOLO11s（960，`hardhat` / `no-hardhat` 两类），火灾烟雾用 YOLO 实例分割（960，`fire` / `smoke` 两类）。训练脚本与数据集不在本仓库。
- **已有权重**：把 `.pt` 放到对应目录即可，没有 `.engine` 时自动回退到 PyTorch 推理。

## 导出 engine

```bash
python scripts/export_engines.py --weights weights/helmet/best.pt --imgsz 960
python scripts/export_engines.py --weights weights/fire/best.pt --imgsz 960
```

导出成功后把生成的 `best.engine` 放在同目录，`config/scenes.yaml` 里的 `model` 指向 `.engine` 即可（没有 engine 时指向 `.pt`，速度慢一些但能跑）。
