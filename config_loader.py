# -*- coding: utf-8 -*-
"""
配置加载模块 —— 工单编号：CV项目-安防智能体-02-项目初始化、配置体系与多路视频异步流读取

职责：
    1. 读取 config/ 下的四份 YAML，并提供校验与查询
         system.yaml    全局参数（路径、拉流、告警、云端）    ← 工单 02 新增
         rules.yaml     五大业务场景阈值                      ← 工单 02 新增
         cameras.yaml   摄像头地址与 ROI                      ← 工单 02 新增
         scenes.yaml    模型与类别定义（V1.0 沿用，不破坏）
    2. 支持环境变量热替换，写法 ${VAR} 或 ${VAR:默认值}
       这样摄像头地址、机器人 webhook、云端密钥都不用写进仓库。

为什么配置要拆成四份：
    system 改一次全场生效，rules 改了只影响判定，cameras 改了只影响拉流，
    scenes 改了只影响模型。四者变更频率完全不同，混在一个文件里改一处要
    回归全部。拆开之后，现场调 ROI 不用碰模型配置，换摄像头不用碰规则。

向后兼容：
    V1.0 的 load_config / get_scene / list_scenes / get_output_dirs 签名不变，
    app.py 与 agents/* 无需改动。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = PROJECT_ROOT / "config"

CONFIG_PATH = CONFIG_DIR / "scenes.yaml"       # V1.0 主配置
SYSTEM_PATH = CONFIG_DIR / "system.yaml"
RULES_PATH = CONFIG_DIR / "rules.yaml"
CAMERAS_PATH = CONFIG_DIR / "cameras.yaml"

# 允许的模型任务类型
VALID_TASKS = {"detect", "segment", "classify", "pose", "obb"}

# ${VAR} 或 ${VAR:default}
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")

# 记录哪些环境变量没配到，便于启动时提示
_missing_env: set[str] = set()


# ============================================================
# 环境变量替换
# ============================================================
def expand_env(value: Any) -> Any:
    """递归把配置里的 ${VAR} 换成环境变量值。

    规则：
        ${VAR}          → 环境变量，不存在则替换为空字符串，并记入 _missing_env
        ${VAR:默认值}    → 环境变量，不存在则用默认值（不算缺失）

    dict 的 key 和 value 都替换，list 逐项替换，其它类型原样返回。
    """
    if isinstance(value, str):
        def _sub(m: re.Match) -> str:
            name, default = m.group(1), m.group(2)
            env = os.environ.get(name)
            if env is not None and env != "":
                return env
            if default is not None:
                return default
            _missing_env.add(name)
            return ""
        return _ENV_PATTERN.sub(_sub, value)

    if isinstance(value, dict):
        return {expand_env(k): expand_env(v) for k, v in value.items()}

    if isinstance(value, list):
        return [expand_env(v) for v in value]

    return value


def missing_env_vars() -> list[str]:
    """返回本次进程中没配到的环境变量名（去重排序）。

    给启动日志用：缺 ${DINGDING_WEBHOOK} 不影响跑检测，但推送会静默失败，
    所以启动时应该打一行提示，而不是等到报警推不出去才发现。
    """
    return sorted(_missing_env)


# ============================================================
# 通用读取
# ============================================================
def _read_yaml(path: Path) -> dict[str, Any]:
    """读 YAML 并做环境变量替换，返回 dict。"""
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层必须是字典：{path}")
    return expand_env(data)


def _resolve(path_str: str) -> str:
    """把配置里的相对路径按项目根目录解析成绝对路径。

    绝对路径（比如 C:/models/best.pt）原样返回。
    """
    if not path_str:
        return ""
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str(PROJECT_ROOT / p)


# ============================================================
# 三份新配置（工单 02）
# ============================================================
def load_system(path: str | Path | None = None) -> dict[str, Any]:
    """加载 system.yaml 并补齐默认值。"""
    cfg = _read_yaml(Path(path) if path else SYSTEM_PATH)

    # 路径段：全部转成绝对路径，方便别处直接用
    paths = cfg.get("paths") or {}
    for key, rel in list(paths.items()):
        if isinstance(rel, str):
            paths[key] = _resolve(rel)
    cfg["paths"] = paths

    # 拉流段默认值
    stream = cfg.setdefault("stream", {})
    stream.setdefault("target_fps", 5)
    stream.setdefault("max_streams", 8)
    stream.setdefault("buffer_size", 1)
    stream.setdefault("open_timeout_ms", 5000)
    stream.setdefault("read_timeout_ms", 3000)
    stream.setdefault("stall_timeout_sec", 10)
    stream.setdefault("reconnect_delays", [1, 2, 4, 8, 16, 32, 60])
    stream.setdefault("offline_after_retries", 3)
    stream.setdefault("recover_frames", 10)

    if stream["target_fps"] <= 0:
        raise ValueError("system.yaml: stream.target_fps 必须大于 0")
    if not stream["reconnect_delays"]:
        raise ValueError("system.yaml: stream.reconnect_delays 不能为空")

    # 告警段默认值
    alert = cfg.setdefault("alert", {})
    alert.setdefault("webhook_url", "")
    alert.setdefault("webhook_timeout_sec", 5)
    alert.setdefault("dedup_window_sec", 5)

    return cfg


def load_rules(path: str | Path | None = None) -> dict[str, Any]:
    """加载 rules.yaml 并做基本校验。"""
    cfg = _read_yaml(Path(path) if path else RULES_PATH)

    metrics = ("helmet", "fire", "intrusion", "fall", "away", "sleep")
    missing = [m for m in metrics if m not in cfg]
    if missing:
        raise KeyError(f"rules.yaml 缺少场景规则：{missing}")

    # min_consecutive_frames 是可选字段：
    #   安全帽/火灾/睡岗靠帧数去抖，必须写
    #   入侵/跌倒/离岗靠秒数累计（dwell_sec / critical_after_sec / accumulate_sec），
    #   可以只写秒数，由业务层按 fps 换算，所以这里存在才校验
    for name in metrics:
        if "min_consecutive_frames" not in cfg[name]:
            continue
        n = cfg[name]["min_consecutive_frames"]
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"rules.yaml: {name}.min_consecutive_frames 必须是 ≥1 的整数")

    return cfg


def load_cameras(path: str | Path | None = None) -> dict[str, Any]:
    """加载 cameras.yaml，校验摄像头与 ROI。"""
    cfg = _read_yaml(Path(path) if path else CAMERAS_PATH)

    cams = cfg.get("cameras")
    if not isinstance(cams, dict) or not cams:
        raise ValueError("cameras.yaml 缺少 cameras 段")

    for cam_id, cam in cams.items():
        if "source" not in cam:
            raise KeyError(f"cameras.yaml: [{cam_id}] 缺少 source")
        fps = cam.get("target_fps", 5)
        if not isinstance(fps, (int, float)) or fps <= 0:
            raise ValueError(f"cameras.yaml: [{cam_id}] target_fps 必须大于 0")
        # ROI 校验：坐标必须归一化到 [0,1]，否则换了分辨率就错位
        for roi in cam.get("roi") or []:
            poly = roi.get("polygon") or []
            if len(poly) < 3:
                raise ValueError(
                    f"cameras.yaml: [{cam_id}] ROI [{roi.get('name')}] 至少要 3 个点"
                )
            for pt in poly:
                if len(pt) != 2 or not all(0.0 <= float(v) <= 1.0 for v in pt):
                    raise ValueError(
                        f"cameras.yaml: [{cam_id}] ROI [{roi.get('name')}] "
                        f"坐标必须归一化到 0~1，实际：{pt}"
                    )

    return cfg


def load_all() -> dict[str, Any]:
    """一次性加载全部配置，返回一个总字典。

    返回结构：
        {
          "system":  {...},   # system.yaml
          "rules":   {...},   # rules.yaml
          "cameras": {...},   # cameras.yaml
          "scenes":  {...},   # scenes.yaml（V1.0，含 output 段）
        }
    """
    return {
        "system": load_system(),
        "rules": load_rules(),
        "cameras": load_cameras(),
        "scenes": load_config(),
    }


def resolve_stream_source(cameras_cfg: dict, cam_id: str) -> str:
    """取某一路摄像头的流地址，必要时回退到本地视频。

    现场没接 RTSP 时，cameras.yaml 里 source 展开后会变成空串。
    这时如果 fallback.enabled 为真，就用 fallback 段里的本地文件顶上，
    保证整条链路（拉流→推理→告警→报告）能跑通。
    """
    cam = (cameras_cfg.get("cameras") or {}).get(cam_id)
    if not cam:
        raise KeyError(f"cameras.yaml 里没有 [{cam_id}]")

    src = (cam.get("source") or "").strip()
    if src:
        if not src.isdigit() and "://" not in src:
            return _resolve(src)
        return src

    fb = cameras_cfg.get("fallback") or {}
    if fb.get("enabled") and fb.get(cam_id):
        return _resolve(fb[cam_id])

    raise ValueError(
        f"[{cam_id}] 的 source 为空，且 fallback 里也没有可用地址。"
        f"请设置环境变量或修改 config/cameras.yaml"
    )


# ============================================================
# V1.0 主配置（模型与场景），签名保持不变
# ============================================================
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """加载并校验场景配置（scenes.yaml）。"""
    cfg = _read_yaml(Path(path) if path else CONFIG_PATH)

    if "scenes" not in cfg:
        raise ValueError("配置文件缺少 scenes 段")

    for name, scene in cfg["scenes"].items():
        _validate_scene(name, scene)

    return cfg


def _validate_scene(name: str, scene: dict) -> None:
    """校验单个场景配置，缺字段立刻报错，不要等跑到一半才发现。

    支持两种规则组织方式：
        单规则：顶层 alert + rules（安全帽 / 火灾 / 睡岗 / 徘徊）
        多规则：rules.multi_rules[*].alert（值班室综合场景）
    所以顶层 alert 只在单规则场景才是必填。
    """
    rules_cfg = scene.get("rules") or {}
    multi = rules_cfg.get("multi_rules")

    required = ["model", "task", "classes", "violation_classes"]
    if not multi:
        required.append("alert")
    for key in required:
        if key not in scene:
            raise KeyError(f"场景 [{name}] 缺少必填字段：{key}")

    task = scene["task"]
    if task not in VALID_TASKS:
        raise ValueError(f"场景 [{name}] 的 task={task} 不支持，可选：{VALID_TASKS}")

    classes = scene["classes"]
    if not isinstance(classes, dict) or not classes:
        raise ValueError(f"场景 [{name}] 的 classes 必须是非空字典")

    for cid in scene["violation_classes"]:
        if cid not in classes:
            raise ValueError(
                f"场景 [{name}] 的 violation_classes 含有未定义类别 {cid}"
                f"（已定义：{list(classes.keys())}）"
            )

    if multi:
        if not isinstance(multi, list) or not multi:
            raise ValueError(f"场景 [{name}] 的 multi_rules 必须是非空列表")
        for i, r in enumerate(multi):
            if not r.get("name"):
                raise KeyError(f"场景 [{name}] 第 {i + 1} 条规则缺少 name")
            alert = r.get("alert")
            if not isinstance(alert, dict):
                raise KeyError(f"场景 [{name}] 规则 [{r.get('name')}] 缺少 alert")
            for k in ("event_type", "level", "title", "message"):
                if k not in alert:
                    raise KeyError(
                        f"场景 [{name}] 规则 [{r['name']}] 的 alert 缺少 {k}"
                    )

    model_path = _resolve(scene["model"])
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"场景 [{name}] 的模型文件不存在：{model_path}\n"
            f"请先把模型权重放到 weights/ 下（见 weights/README.md），"
            f"或修正 config/scenes.yaml 的 model 字段"
        )


def get_scene(cfg: dict, scene_name: str) -> dict:
    """取出指定场景配置，并把 model 转成绝对路径。"""
    if scene_name not in cfg["scenes"]:
        raise KeyError(
            f"未知场景 [{scene_name}]，可选：{list(cfg['scenes'].keys())}"
        )
    scene = dict(cfg["scenes"][scene_name])
    scene["name"] = scene_name
    scene["model"] = _resolve(scene["model"])
    scene.setdefault("imgsz", 640)
    scene.setdefault("conf", 0.25)
    scene.setdefault("iou", 0.45)
    scene.setdefault("rules", {})
    scene["rules"].setdefault("min_consecutive_frames", 1)
    return scene


def list_scenes(cfg: dict) -> list[str]:
    """列出所有可用场景名。"""
    return list(cfg["scenes"].keys())


def get_output_dirs(cfg: dict) -> dict[str, Path]:
    """返回输出目录（已创建）。

    优先读 system.yaml 的 paths 段，没有就退回 scenes.yaml 的 output 段。
    """
    out = cfg.get("output", {}) or {}
    paths = cfg.get("paths", {}) or {}
    dirs = {
        "snapshots": Path(
            paths.get("snapshots_dir")
            or PROJECT_ROOT / out.get("snapshots_dir", "outputs/events")
        ),
        "reports": Path(
            paths.get("reports_dir")
            or PROJECT_ROOT / out.get("reports_dir", "outputs/reports")
        ),
        "videos": Path(
            paths.get("videos_out_dir")
            or PROJECT_ROOT / out.get("videos_dir", "outputs/videos")
        ),
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs
