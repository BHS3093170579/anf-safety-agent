# -*- coding: utf-8 -*-
"""线程安全的事件异步分发与上报器。

设计要点
--------
1. **非阻塞入队**：`push()` 永远不等待网络。队列满（默认 50）时，按设计
   要求优先丢弃最老的 `suspected` 记录，保住 `abnormal` 红色高危告警。

2. **后台守护线程**：`DaemonUploader` 用 `while True` 从队列取事件，
   调 `requests.post` 上报。网络慢/断开只影响这个线程，主流水线照常跑。

3. **Pydantic 校验**：事件载荷在入队前用 `AlertEvent` 校验，保证上游
   构造的 JSON 一定符合规范（验收标准第 2 条）。

4. **可注入上传函数**：测试时把 `post_fn` 换成本地桩，不需要真网络。
"""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------- #
# 事件数据模型（Pydantic 校验规范）
# ---------------------------------------------------------------------- #
FEATURE_DIM = 512

class AlertEvent(BaseModel):
    """一条待上报的告警事件。

    字段与要求一一对应：事件 UUID、摄像头 ID、违规类型、特征向量、抓拍图。
    """

    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex, description="事件唯一 ID")
    camera_id: str = Field(..., min_length=1, description="摄像头编号，如 Camera_01")
    track_id: int = Field(0, ge=0, description="目标追踪 ID；区域级规则（离岗/入侵）固定 0")
    violation_type: str = Field(..., min_length=1, description="违规类型：intrusion/fall/away/sleep/helmet")
    severity: Literal["suspected", "abnormal"] = Field("abnormal", description="告警级别")
    timestamp: float = Field(..., gt=0, description="事件发生的秒级时间戳")
    video_time: float = Field(0.0, ge=0, description="事件在视频内的秒数")
    reid_feature: list[float] = Field(default_factory=list, description="512 维 Re-ID 特征")
    snapshot_path: str | None = Field(None, description="本地抓拍图片路径")
    message: str = Field("", description="人类可读描述")
    extra: dict[str, Any] = Field(default_factory=dict, description="附加信息")
    created_at: float = Field(default_factory=time.time, description="入队时刻")

    @field_validator("reid_feature")
    @classmethod
    def _check_feature(cls, v: list[float]) -> list[float]:
        if not v:
            return v
        if len(v) != FEATURE_DIM:
            raise ValueError(f"Re-ID 特征必须是 {FEATURE_DIM} 维，收到 {len(v)} 维")
        return [round(float(x), 6) for x in v]

    def to_payload(self) -> dict:
        """转成待 POST 的 JSON 字典（特征保留完整精度）。"""
        d = self.model_dump()
        d.pop("created_at", None)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_payload(), ensure_ascii=False)

# ---------------------------------------------------------------------- #
# 分发器
# ---------------------------------------------------------------------- #
class EventDispatcher:
    """线程安全队列 + 后台守护上报线程。"""

    def __init__(
        self,
        maxsize: int = 50,
        endpoint: str = "http://127.0.0.1:8000/api/events",
        timeout: float = 2.0,
        post_fn: Callable[[str, dict, float], bool] | None = None,
        auto_start: bool = False,
        verbose: bool = False,
    ) -> None:
        # 要求全局线程安全队列 queue.Queue(maxsize=50)
        self.event_queue: queue.Queue[AlertEvent] = queue.Queue(maxsize=maxsize)
        self.maxsize = maxsize
        self.endpoint = endpoint
        self.timeout = timeout
        self._post_fn = post_fn or self._default_post
        self.verbose = verbose

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._worker: DaemonUploader | None = None

        # 统计
        self.pushed = 0
        self.dropped_suspected = 0
        self.dropped_abnormal = 0
        self.sent = 0
        self.failed = 0
        self.last_error = ""

        if auto_start:
            self.start()

    # ------------------------------------------------------------------ #
    # 入队
    # ------------------------------------------------------------------ #
    def push(self, event: AlertEvent | dict) -> bool:
        """非阻塞入队。返回 True 表示成功进入队列。"""
        evt = event if isinstance(event, AlertEvent) else AlertEvent(**event)

        try:
            self.event_queue.put_nowait(evt)
            with self._lock:
                self.pushed += 1
            return True
        except queue.Full:
            pass

        # 队列满：优先丢最老的 suspected，保住 abnormal
        if not self._drop_oldest_suspected():
            if evt.severity == "suspected":
                with self._lock:
                    self.dropped_suspected += 1
                self._log("队列满且无 suspected 可丢，丢弃本条 suspected")
                return False
            # 整队都是 abnormal：丢最老的一条 abnormal（保留最新高危）
            try:
                self.event_queue.get_nowait()
                with self._lock:
                    self.dropped_abnormal += 1
            except queue.Empty:
                pass
            self._log("队列满且全为 abnormal，丢弃最老的一条 abnormal")

        try:
            self.event_queue.put_nowait(evt)
            with self._lock:
                self.pushed += 1
            return True
        except queue.Full:
            with self._lock:
                self.dropped_abnormal += 1
            return False

    def _drop_oldest_suspected(self) -> bool:
        """把队列倒出来，删掉最老的一条 suspected，再按原顺序放回。

        队列上限只有 50，且只在满的时候才触发，开销可忽略。
        """
        with self._lock:
            if self.event_queue.empty():
                return False
            items: list[AlertEvent] = []
            while True:
                try:
                    items.append(self.event_queue.get_nowait())
                except queue.Empty:
                    break

            idx = next((i for i, e in enumerate(items) if e.severity == "suspected"), -1)
            if idx < 0:
                for e in items:              # 原样放回
                    self.event_queue.put_nowait(e)
                return False

            items.pop(idx)
            self.dropped_suspected += 1
            for e in items:
                self.event_queue.put_nowait(e)
            return True

    # ------------------------------------------------------------------ #
    # 后台线程
    # ------------------------------------------------------------------ #
    def _default_post(self, url: str, payload: dict, timeout: float) -> bool:
        import requests  # noqa: PLC0415

        r = requests.post(url, json=payload, timeout=timeout)
        return 200 <= r.status_code < 300

    def start(self) -> "EventDispatcher":
        if self._worker is not None and self._worker.is_alive():
            return self
        self._stop.clear()
        self._worker = DaemonUploader(self)
        self._worker.start()
        self._log("DaemonUploader 已启动")
        return self

    def stop(self, drain_timeout: float = 2.0) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=drain_timeout)
            self._worker = None

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def qsize(self) -> int:
        return self.event_queue.qsize()

    def stats(self) -> dict:
        with self._lock:
            return {
                "qsize": self.event_queue.qsize(),
                "maxsize": self.maxsize,
                "pushed": self.pushed,
                "sent": self.sent,
                "failed": self.failed,
                "dropped_suspected": self.dropped_suspected,
                "dropped_abnormal": self.dropped_abnormal,
                "uploader_alive": bool(self._worker and self._worker.is_alive()),
                "last_error": self.last_error,
            }

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[EventDispatcher] {msg}")

# ---------------------------------------------------------------------- #
# 守护上报线程
# ---------------------------------------------------------------------- #
class DaemonUploader(threading.Thread):
    """后台守护线程：while True 从队列取事件并 POST 上报。"""

    def __init__(self, dispatcher: EventDispatcher) -> None:
        super().__init__(name="DaemonUploader", daemon=True)
        self.d = dispatcher

    def run(self) -> None:  # noqa: D401
        d = self.d
        while not d.stopped:
            try:
                evt = d.event_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                ok = d._post_fn(d.endpoint, evt.to_payload(), d.timeout)
                with d._lock:
                    if ok:
                        d.sent += 1
                    else:
                        d.failed += 1
                        d.last_error = "POST 返回非 2xx"
            except Exception as exc:  # 网络断开、超时、DNS 失败等都不允许冒泡
                with d._lock:
                    d.failed += 1
                    d.last_error = f"{type(exc).__name__}: {exc}"
            finally:
                d.event_queue.task_done()
