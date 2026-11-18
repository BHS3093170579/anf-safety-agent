# 安防智能体 · 工地安全五合一检测

面向工地安全的实时视频分析系统，把**安全帽佩戴、火焰烟雾、值班睡岗、岗位离岗、禁区徘徊**五类隐患合并到一条边缘推理链路，共用一套模型池、规则引擎、事件总线和 Web 界面。

> 一套模型池同时带多路摄像头；一次姿态推理同时服务睡岗、离岗、徘徊三条规则。

| 安全帽检测（未戴帽标红） | 火焰 / 烟雾分割 |
|---|---|
| ![安全帽检测](docs/images/helmet.png) | ![火焰分割](docs/images/fire.png) |

## 目录

- [能力](#能力)
- [架构](#架构)
- [快速开始](#快速开始)
- [配置](#配置)
- [性能](#性能)
- [目录结构](#目录结构)
- [已知限制](#已知限制)
- [License](#license)

---

## 能力

| 能力 | 模型 | 判定方式 | 默认相机 |
|---|---|---|---|
| 未戴安全帽 | YOLO11s 检测（TensorRT FP16，960） | 人体框 × 头框配对，`hardhat` / `no-hardhat` 两类 | Camera_01 |
| 火灾烟雾 | YOLO 实例分割（TensorRT FP16，960） | 掩码面积占比 + 连续帧确认 | Camera_02 |
| 睡岗 | YOLOv8n-pose | 头部下沉比 `(nose.y − 肩中点y) / 肩宽`，阈值 −0.32 | Camera_03 |
| 离岗 | YOLOv8n-pose | ROI 内无人累计 60 s 疑似、180 s 确认 | Camera_03 |
| 徘徊 | YOLOv8n-pose + ByteTrack | ROI 内停留 8 s 疑似、20 s 确认，或折返 2 次 | Camera_04 |

安全帽另有两种可选检测方式：**两步法**（头检测 + 头分类，小目标更准）和**姿态兜底**（端到端结果再用人头姿态补齐）。

界面共五个页签：实时监控、规则引擎、多路监控（2–4 路并行）、告警列表、报告查看。

---

## 架构

```mermaid
flowchart LR
    subgraph 输入
        A[RTSP 网络摄像头]
        B[本地视频 / USB 摄像头]
    end

    subgraph 取流层
        C[StreamReader<br/>单路单线程 · 按时间戳降采样 · 指数退避重连]
    end

    subgraph 推理层
        D[ModelService<br/>共享模型池 · 按权重路径缓存]
        D1[YOLO11s 安全帽]
        D2[YOLO-Seg 火焰烟雾]
        D3[YOLOv8n-pose 姿态]
        D --> D1 & D2 & D3
    end

    subgraph 规则层
        E[RulesEngine]
        E1[helmet / fire]
        E2[away / sleep]
        E3[loiter]
        E --> E1 & E2 & E3
    end

    subgraph 输出层
        F[EventDispatcher<br/>有界队列 · 异步上报]
        G[抓拍存证 / 事件日志]
        H[钉钉 / 企业微信通知]
        I[报告生成]
    end

    A & B --> C --> D
    D --> E --> F
    F --> G & H & I
```

决策流（LangGraph）串起「取流 → 检测 → 判断 → 告警 → 报告」，条件路由决定是否进入告警节点。

关键设计：

- **一次推理多规则复用**：姿态模型只跑一次，睡岗、离岗、徘徊三条规则共用同一份关键点。
- **按场景分派规则**：每路相机在配置里声明自己的 `scene`，规则引擎只跑声明过的规则，避免误触发。
- **去重与冷却**：同一目标 30 秒内不重复报，且同一相机同一规则下，所有历史告警框都参与 IoU 比对（只比对最后一个框会被别人覆盖）。
- **检测跳帧**：摄像头模式按目标帧率跳帧推理，画面仍逐帧刷新。

---

## 快速开始

### 1. 环境

- Python 3.10+
- 可选：NVIDIA GPU + TensorRT 10.x（用 `.engine` 加速）；没有 GPU 时用 `.pt` 走 PyTorch 推理

```bash
git clone https://github.com/BHS3093170579/anf-safety-agent.git
cd anf-safety-agent
pip install -r requirements.txt
```

### 2. 准备模型权重

权重（约 200 MB）不随仓库发布，需要自己准备；程序启动时若缺权重，会提示具体缺哪个文件。

按下面结构放置：

```
weights/
├── helmet/
│   ├── best.engine      # TensorRT FP16，960（可选，没有就用 best.pt）
│   ├── best.pt          # PyTorch 权重
│   └── best.onnx
├── fire/
│   ├── best.engine
│   ├── best.pt
│   └── best.onnx
├── pose/
│   └── yolov8n-pose.pt
└── helmet_qc/           # 两步法与姿态兜底用
    ├── head_det.pt
    ├── head_cls.pt
    └── yolov8n-pose.pt
```

三个来源，任选一种：

1. **自己训练**：安全帽用 YOLO11s 在 960 输入下训练（两类别 `hardhat` / `no-hardhat`），火灾烟雾用 YOLO 实例分割（960，`fire` / `smoke` 两类）。训练脚本与数据集不在本仓库。
2. **已有权重**：把 `.pt` 按上面目录放好就能跑，没有 `.engine` 时自动回退到 PyTorch 推理。
3. **导出 engine**：有 NVIDIA GPU 时用 `best.pt` 导出 TensorRT FP16，本机实测推理快约 2 倍：

   ```bash
   python scripts/export_engines.py --weights weights/helmet/best.pt --imgsz 960
   python scripts/export_engines.py --weights weights/fire/best.pt --imgsz 960
   ```

> TensorRT engine 绑定具体 GPU 与 TRT 版本，换机器必须用 `best.pt` 重新导出。

### 3. 准备测试视频

```bash
mkdir -p datasets/videos/{helmet,fire,sleep,loiter}
# 放入任意 mp4，例如 datasets/videos/helmet/demo.mp4
```

没有现场摄像头时，`config/cameras.yaml` 里的 `fallback` 会让四个相机自动回退到本地视频。

### 4. 启动

```bash
# Web 界面
streamlit run app.py --server.port 8501

# 命令行单视频
python main.py --scene helmet --video datasets/videos/helmet/demo.mp4
```

浏览器打开 <http://127.0.0.1:8501>。

---

## 配置

四份 YAML，改阈值不需要改代码：

| 文件 | 内容 |
|---|---|
| `config/scenes.yaml` | 场景与模型：权重路径、输入尺寸、置信度、类别、后处理 |
| `config/cameras.yaml` | 相机与场景绑定、ROI 多边形、RTSP 地址、回退视频 |
| `config/rules.yaml` | 五条规则的时间阈值、冷却时长、告警级别 |
| `config/system.yaml` | 全局参数：取流帧率、重连策略、告警 webhook |

接入现场摄像头，用环境变量覆盖：

```bash
export CAM01_URL="rtsp://user:pass@192.168.1.101:554/av0_0"
export CAM02_URL="rtsp://user:pass@192.168.1.102:554/av0_0"
```

配置里写成 `${CAM01_URL:默认值}`，变量不存在时用冒号后的默认值。

告警推送（可选）：

```bash
export DINGDING_WEBHOOK="https://oapi.dingtalk.com/robot/send?access_token=xxx"
```

不配置时只落盘到 `outputs/events/edge_alerts.jsonl`，不影响运行。

---

## 性能

单卡 RTX 4060 Laptop 8 GB，TensorRT FP16 实测：

| 场景 | 单帧耗时 | 等效帧率 |
|---|---|---|
| 安全帽单模型（单独加载） | 7 ms | ~140 FPS |
| 安全帽（三模型同驻） | 11 ms | ~90 FPS |
| 火灾分割 | 13.6 ms | ~73 FPS |
| 睡岗 / 离岗全链路 | 12.2 ms | ~82 FPS |
| 徘徊全链路 | 11.2 ms | ~89 FPS |
| 四路并行（均摊） | 24.3 ms | 41 FPS |

模型精度（安全帽，独立测试集 335 张）：

| 指标 | 数值 |
|---|---|
| mAP@0.5 | 0.929 |
| mAP@0.5:0.95 | 0.587 |
| 未戴帽召回 | 0.924 |
| 戴帽召回 | 0.873 |

三模型同驻显存约 156 MB。

> 想把这个仓库发布到 GitHub？推送步骤、仓库设置与自检清单见 [docs/PUBLISH.md](docs/PUBLISH.md)。

---

## 目录结构

```
anf-safety-agent/
├── app.py                  # Streamlit 前端（五页签）
├── multi_panel.py          # 多路监控页（2–4 路、每路可选能力）
├── edge_panel.py           # 规则引擎页（边缘流水线、画框、状态标签）
├── main.py                 # 命令行入口
├── graph_def.py            # LangGraph 决策流
├── config_loader.py        # 配置加载与校验
├── stream_reader.py        # 异步取流（退避重连、按时间戳降采样）
├── agents/                 # 取流、检测、研判、告警、报告、稳定器
├── capabilities/helmet/    # 安全帽三路：端到端 / 两步法 / 姿态兜底
├── edge/                   # 流水线、规则引擎、事件派发、模型服务、落盘
├── tracking/               # ByteTrack 封装、关键点转人体框
├── utils/                  # 模型池、Re-ID、轨迹恢复
├── config/                 # scenes / cameras / rules / system
├── tests/                  # 冒烟与性能基准（入口 run_all.py）
├── scripts/                # 权重导出等工具脚本
├── docs/                   # 架构说明与截图
├── weights/                # 模型权重（不进仓库）
└── datasets/videos/        # 演示与测试视频（不进仓库）
```

---

## 测试

```bash
python tests/run_all.py        # 全部回归，跳过需要摄像头的用例
python tests/run_all.py --with-cam
python tests/bench.py          # 单模型 / 单路 / 四路性能基准
```

---

## 已知限制

- TensorRT engine 与 GPU、TRT 版本绑定，换机器必须重新导出。
- 近景画面里两人头部紧挨时，端到端模型可能给出合并大框；目前用人体框约束 + 显示层抑制嵌套框缓解，彻底解决需引入人体检测裁头。
- 睡岗判定依赖头部下沉比，侧躺、趴桌以外的异常姿势识别有限。
- 演示模式会自动缩短时间阈值（配置里另有正式值）。

---

## License

[MIT](LICENSE)
