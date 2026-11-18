"""线程安全的推理模型池。

工单编号：CV项目-安防智能体-03-边缘端—ModelPool+显存控制与人体 BBox 重构追踪

设计目标（对应工单第 2 节「极限显存瘦身」）：
    显存中只允许常驻**单份** pose 模型，严禁再加载通用目标检测模型。
    人体外接框由 17 个骨骼关键点反向计算（见 tracking/keypoint_bbox.py），
    因此不需要任何额外的 detect 模型。

为什么需要这一层：
    ultralytics 的 YOLO 对象持有一份 CUDA 上的权重与激活缓存。
    如果每路视频各自 `YOLO("yolo11s-pose.pt")`，8 路就是 8 份，
    在 Jetson Nano（4GB）或低端卡上直接 OOM。
    这里用「进程内单例 + 推理串行锁」保证：
        · 权重只解析一次，显存只涨一次
        · 多线程并发到达时排队推理，显存峰值不随线程数增长

关于推理为什么串行：
    同一个 CUDA 模型实例被多线程同时 forward 是不安全的（cuDNN workspace、
    显存池分配都会冲突）。与其每线程克隆一份（显存翻倍），不如串行排队——
    单帧推理 6~12ms，8 路按 5FPS 只需要 40 帧/秒，串行完全够用。
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

POSE_DIR = PROJECT_ROOT / "weights" / "pose"
# 工单原文写的是 yolov8s-pose，本项目统一沿用现有权重。
# 优先用 TensorRT FP16 引擎（快一倍多），引擎不可用时自动回退到 .pt，
# 两者都是 COCO-17 关键点、ultralytics 同接口，下游代码不用改。
#
# he 项目只迁移了 yolov8n-pose.pt（6.5MB，与独立项目的睡岗/徘徊共用），
# 原 yolo11s-pose 的 TensorRT 引擎没有随项目带过来；需要更快时按
# `yolo export model=weights/pose/yolov8n-pose.pt format=engine half=True`
# 自己生成，再把文件名改成下面的 POSE_ENGINE 即可。
POSE_ENGINE = POSE_DIR / "yolov8n-pose.engine"
POSE_PT = POSE_DIR / "yolov8n-pose.pt"
DEFAULT_POSE_WEIGHT = POSE_ENGINE if POSE_ENGINE.exists() else POSE_PT


def is_engine(path: str | Path) -> bool:
    return Path(path).suffix.lower() == ".engine"


def validate_trt_engine(path: str | Path) -> tuple[bool, str]:
    """校验文件是不是**真的** TensorRT 引擎。

    为什么必须做这一步（实测踩到的坑）：
        目录里这个 yolo11s-pose.engine 的前 16 字节是
        `6a02 0000 7b22 6465 7363 7269 7074 696f 6e22`
        也就是 protobuf + `{"description": "Ultralytics YOLO11s-pose ..."`——
        它其实是个**改了扩展名的 ONNX 文件**，不是 TRT 序列化引擎。

        危害比报错更大：ultralytics 按 engine 读取时会走进 TensorRT 的
        反序列化路径并**挂住不返回**（实测 CPU 归零、内存 1.1GB 卡死），
        连 try/except 都接不住。所以必须在交给 YOLO() 之前先验一遍。

    Returns:
        (是否可用, 说明)
    """
    p = Path(path)
    if not p.exists():
        return False, "文件不存在"

    # 先花 1ms 看一眼文件头，能挡掉「ONNX 改名」这种最常见的冒牌货
    try:
        with open(p, "rb") as f:
            head = f.read(8)
        if head[:2] == b"\x6a\x02" or head[:1] in (b"{", b"\x08"):
            return False, f"文件头 {head.hex()} 是 protobuf/ONNX，不是 TRT 引擎"
    except Exception as exc:
        return False, f"读文件头失败：{exc}"

    try:
        import tensorrt as trt
    except ImportError:
        return False, "当前环境没装 tensorrt，无法加载 engine"

    try:
        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        with open(p, "rb") as f:
            data = f.read()
        engine = runtime.deserialize_cuda_engine(data)
        if engine is None:
            return False, "TRT 反序列化返回 None（版本或 GPU 架构不匹配）"
        return True, "OK"
    except Exception as exc:
        return False, f"TRT 反序列化异常：{exc}"


def resolve_pose_weight(prefer_engine: bool = True) -> Path:
    """挑一个**确认能用**的 pose 权重。

    engine 快（FP16 + 算子融合），但不可移植：它绑定导出时的
    TensorRT 版本、CUDA 版本和 GPU 架构。这里对 engine 做真实反序列化校验，
    不通过就静默回退到 .pt，避免程序在加载阶段卡死。
    """
    if prefer_engine and POSE_ENGINE.exists():
        ok, why = validate_trt_engine(POSE_ENGINE)
        if ok:
            return POSE_ENGINE
        print(f"[ModelPool] engine 不可用（{why}），回退 .pt")
    if POSE_PT.exists():
        return POSE_PT
    return POSE_ENGINE


class ModelPool:
    """进程内唯一的 Pose 模型持有者。

    用法：
        pool = ModelPool.get()                       # 拿单例
        res = pool.predict(frame, conf=0.35)         # 推理
        print(pool.vram_used_mb())                   # 看显存

    注意 `ModelPool.get()` 而不是 `ModelPool()`：直接构造会创建新实例，
    绕过单例保护。这里把 __init__ 也锁住，双保险。
    """

    _instance: "ModelPool | None" = None
    # 必须是 RLock（可重入锁），不能用 Lock。
    # 因为 get() 先拿锁再构造实例，而 __init__ 里为了防「绕过 get 直接构造」
    # 又拿了一次同一把锁。普通 Lock 不重入 → 自己等自己 → 永久死锁
    # （现象：CPU 归零、无报错、19 分钟无响应）。
    _singleton_lock = threading.RLock()

    def __init__(self, weight: str | Path | None = None, device: str = "0") -> None:
        # 用 __init__ 也加锁，防止 get() 之外被直接构造两次
        with self._singleton_lock:
            if getattr(self, "_initialized", False):
                return
            self._initialized = True
            # 注意：这里必须走 resolve_pose_weight() 而不能直接用
            # DEFAULT_POSE_WEIGHT。后者只判断文件是否存在，而目录里的
            # yolo11s-pose.engine 其实是改了扩展名的 ONNX 冒牌货——
            # 一旦交给 YOLO() 就会在 TensorRT 反序列化里挂死（实测 19 分钟无响应）。
            self.weight = str(weight) if weight else str(resolve_pose_weight())
            self.device = str(device)
            self._model: Any = None
            self._load_lock = threading.Lock()
            self._infer_lock = threading.Lock()
            self._load_count = 0
            self._infer_count = 0
            self._load_ts: float | None = None
            self._warm: list[float] = []

    # ---------- 单例入口 ----------
    @classmethod
    def get(cls, weight: str | Path | None = None, device: str = "0") -> "ModelPool":
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = cls(weight=weight, device=device)
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """仅测试用：把单例清空（会释放 CUDA 缓存）。"""
        with cls._singleton_lock:
            if cls._instance is not None:
                cls._instance.release()
            cls._instance = None

    # ---------- 模型加载 ----------
    def load(self) -> Any:
        """懒加载。整个进程只执行一次真正的 YOLO(...) 构造。"""
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:      # 双重检查：并发时只有一个进去
                return self._model

            t0 = time.perf_counter()
            # 必须先用离线模式：ultralytics 导入/构造时会做联网检查
            # （探测更新、校验依赖、拉字体）。本次实测环境下这些请求会长时间
            # 阻塞——同样的代码加了这个开关是 4.6 秒，不加则 19 分钟无响应。
            os.environ.setdefault("YOLO_OFFLINE", "true")
            os.environ.setdefault("YOLO_VERBOSE", "false")
            from ultralytics import YOLO

            # engine 绑定了导出时的 TRT/CUDA/GPU，换环境就会加载失败。
            # 这里按「指定权重 → 另一个格式」的顺序试，全失败才抛错。
            #
            # 注意：不能在循环之前就先判断 self.weight 是否存在并直接抛错——
            # 那样「指定了 .engine 但机器上只有 .pt」这种最常见的场景会直接失败，
            # 回退逻辑形同虚设。存在性检查交给 candidates 过滤。
            candidates: list[Path] = [Path(self.weight)]
            for alt in (POSE_ENGINE, POSE_PT):
                if alt.resolve() != Path(self.weight).resolve():
                    candidates.append(alt)
            candidates = [c for c in candidates if c.exists()]

            if not candidates:
                raise FileNotFoundError(
                    f"Pose 权重不存在：{self.weight}\n"
                    f"已尝试：{[str(POSE_ENGINE), str(POSE_PT)]}\n"
                    f"工单 03 要求只加载 pose 模型，请先把权重放到 weights/pose/ 目录"
                )

            model = None
            last_err: Exception | None = None
            for cand in candidates:
                try:
                    model = YOLO(str(cand))
                    # 预热：把反序列化/显存池/workspace 的第一次开销吃掉，
                    # 否则第一帧会被误判成显存爬升或首帧超时。
                    model.predict(
                        np.zeros((640, 640, 3), dtype=np.uint8),
                        device=self.device, verbose=False, conf=0.9,
                    )
                    self.weight = str(cand)
                    break
                except Exception as exc:      # noqa: BLE001 - 需要在多格式间回退
                    last_err = exc
                    model = None
                    print(f"[ModelPool] 加载 {Path(cand).name} 失败：{exc}")

            if model is None:
                raise RuntimeError(
                    f"pose 权重全部加载失败（最后错误：{last_err}）\n"
                    f"试过：{[Path(c).name for c in candidates]}"
                )

            self._model = model
            self._load_count += 1
            self._load_ts = time.time()
            self._warm.append((time.perf_counter() - t0) * 1000)

            print(
                f"[ModelPool] 模型已加载（第 {self._load_count} 次，进程内唯一实例）\n"
                f"            权重={Path(self.weight).name}  device={self.device}  "
                f"耗时={(time.perf_counter()-t0)*1000:.0f}ms  "
                f"显存={self.vram_used_mb():.1f}MB"
            )
            return self._model

    # ---------- 推理 ----------
    def predict(self, frame: np.ndarray, **kwargs) -> Any:
        """对外唯一推理入口。多线程并发调用时自动排队。"""
        model = self.load()
        default = dict(conf=0.35, iou=0.5, device=self.device, verbose=False)
        # engine 的输入分辨率在导出时就定死了，传 imgsz 只会触发警告
        if not is_engine(self.weight):
            default["imgsz"] = 640
        default.update(kwargs)
        with self._infer_lock:
            self._infer_count += 1
            return model.predict(frame, **default)

    def track(self, frame: np.ndarray, **kwargs) -> Any:
        """预留：如果要走 ultralytics 内置 track（本工单不用，因为要自己重构 BBox）。"""
        raise NotImplementedError(
            "工单 03 要求用「关键点重构 BBox + 自建 ByteTrack」，"
            "请走 tracking/bytetrack_wrapper.py，不要用内置 track()"
        )

    # ---------- 显存观测 ----------
    @staticmethod
    def vram_used_mb() -> float:
        """当前进程已分配的显存（MB）。没有 CUDA 时返回 0。"""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024 ** 2
        except Exception:
            pass
        return 0.0

    @staticmethod
    def vram_reserved_mb() -> float:
        """当前进程向驱动申请的显存（MB）。比 allocated 大，含缓存池。"""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_reserved() / 1024 ** 2
        except Exception:
            pass
        return 0.0

    def vram_json(self) -> dict:
        return {
            "allocated_mb": round(self.vram_used_mb(), 2),
            "reserved_mb": round(self.vram_reserved_mb(), 2),
            "load_count": self._load_count,
            "infer_count": self._infer_count,
        }

    # ---------- 统计与释放 ----------
    def stats(self) -> dict:
        return {
            "weight": Path(self.weight).name,
            "device": self.device,
            "loaded": self._model is not None,
            "load_count": self._load_count,
            "infer_count": self._infer_count,
            "load_ms": round(self._warm[0], 1) if self._warm else 0.0,
            **self.vram_json(),
        }

    def release(self) -> None:
        self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def ensure_dirs() -> None:
    """建目录（首次运行时用）。"""
    for d in ("utils", "tracking", "edge"):
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    pool = ModelPool.get()
    print("单例校验：", ModelPool.get() is pool)

    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    r = pool.predict(img)
    print("推理输出：", type(r).__name__, len(r))
    print("统计：", pool.stats())
