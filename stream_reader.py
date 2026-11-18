# -*- coding: utf-8 -*-
"""
多路视频异步流读取器 ——

要解决的三个现场问题：
    1. 一路 RTSP 断线（比如挖掘机把网线铲了），主进程在 cv2.VideoCapture.read()
       处永久阻塞，导致另外两路也跟着卡死。
       → 每路一个独立线程，谁卡谁自己卡，主进程不受影响。
    2. 原始流 25/30 FPS，算法只需要 5 FPS，多余的解码和推理全是浪费。
       → 拉流线程按**时间戳**降采样，不按帧号（RTSP 的帧号不可靠）。
    3. 消费端（推理）比生产端（拉流）慢时，缓冲会积压出延迟，
       画面越跑越落后。
       → 双缓冲 + 缓冲满时丢弃最老帧，消费端永远拿最新画面。

对外接口：
    mgr = StreamManager(system_cfg, cameras_cfg)
    mgr.start()
    while True:
        pkt = mgr.get_frame("Camera_01", timeout=1.0)
        if pkt: 用 pkt.frame 去推理
    mgr.stop()

设计取舍：
    - 用 threading 而不是 multiprocessing：OpenCV 的 GIL 在 FFMPEG 解码时
      大部分时间会释放，实测 3 路 5FPS 的线程方案 CPU 占用远低于多进程开销。
    - 双缓冲而不是队列：队列再长也只是延后，安防要的是「现在」。
      保留 2 块是为了让读写指针永不同时指向同一块内存，锁只保护指针交换。
"""

from __future__ import annotations

import socket
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

# ============================================================
# 状态与数据结构
# ============================================================
class StreamState(str, Enum):
    """单路流的状态机。"""
    INIT = "init"                # 刚创建，还没开始
    ONLINE = "online"            # 正常出帧
    RECONNECTING = "reconnecting"  # 断流，正在退避重连
    OFFLINE = "offline"          # 重试超过阈值，已注册 stream_offline
    STOPPED = "stopped"          # 主动停止

@dataclass
class FramePacket:
    """一帧画面 + 元信息。

    带 cam_id 和 ts 是为了让下游知道「这帧是哪路、什么时刻的」。
    告警去重、ROI 判定、报告时间轴都依赖 ts，不能省。
    """
    cam_id: str
    frame: np.ndarray
    ts: float                    # time.monotonic()
    wall_ts: float               # time.time()，写进报告用
    seq: int                     # 该路第几帧
    source_fps: float = 0.0

@dataclass
class ReaderStats:
    """单路运行统计，给控制台和界面看。"""
    frames_emitted: int = 0      # 已吐给算法的帧数
    frames_decoded: int = 0      # 从流里解出来的帧数（含被降采样丢掉的）
    reconnects: int = 0          # 重连次数
    offline_events: int = 0      # 注册过几次 stream_offline
    last_frame_ts: float = 0.0
    state: StreamState = StreamState.INIT
    last_error: str = ""

    @property
    def decode_ratio(self) -> float:
        """解出来的帧里，有多少比例真的送到了算法。"""
        if self.frames_decoded <= 0:
            return 0.0
        return self.frames_emitted / self.frames_decoded

# ============================================================
# 双缓冲
# ============================================================
class DoubleBuffer:
    """双缓冲：写线程只写，读线程只取最新，锁只保护指针交换。

    为什么不用 queue.Queue：
        队列是「先来先得」，消费慢的时候积压的是延迟。
        双缓冲是「后来的覆盖先来的」，消费端拿到的永远是最新一帧，
        代价是中间帧被丢掉——安防场景丢中间帧完全可以接受，
        延迟不可接受。

    线程安全：
        _write(buf) 和 read_latest() 操作的是不同的 slot（交换指针保证），
        所以真正的内容读写不需要锁，只有指针交换需要。
    """

    def __init__(self) -> None:
        self._slots: list[FramePacket | None] = [None, None]
        self._write_idx = 0
        self._read_idx = 1
        self._lock = threading.Lock()
        self._new_frame = threading.Event()

    def write(self, pkt: FramePacket) -> None:
        """写入一帧（覆盖旧帧）。"""
        with self._lock:
            self._slots[self._write_idx] = pkt
            # 交换读写指针：刚写满的这块变成可读的
            self._write_idx, self._read_idx = self._read_idx, self._write_idx
            self._new_frame.set()

    def read_latest(self, timeout: float | None = None) -> FramePacket | None:
        """取最新一帧。超时返回 None。"""
        if not self._new_frame.wait(timeout):
            return None
        with self._lock:
            self._new_frame.clear()
            return self._slots[self._read_idx]

    def peek(self) -> FramePacket | None:
        """不消费，只看当前最新帧（给状态面板用）。"""
        with self._lock:
            return self._slots[self._read_idx]

    def clear(self) -> None:
        with self._lock:
            self._slots = [None, None]
            self._new_frame.clear()

# ============================================================
# 单路读取线程
# ============================================================
class StreamReader(threading.Thread):
    """一路视频流一个读取线程。

    生命周期：
        构造 → start() → [打开流 → 读帧 → 降采样 → 写缓冲] 循环 → stop()

    断流处理（要求的指数退避）：
        读失败 → state=RECONNECTING → 按 reconnect_delays 退避
        第 1 次等 1s，第 2 次等 2s，第 3 次等 4s …… 最长 60s
        重试次数 ≥ offline_after_retries(默认3) → 注册 stream_offline 事件
        但**不停止线程**，继续按 60s 间隔尝试恢复——
        因为现场网线接回来之后，程序应该自己好起来，不需要人重启服务。
    """

    def __init__(
        self,
        cam_id: str,
        source: str | int,
        target_fps: float = 5.0,
        system_cfg: dict | None = None,
        on_event: Callable[[str, str, dict], None] | None = None,
        loop_file: bool = True,
    ) -> None:
        super().__init__(name=f"StreamReader-{cam_id}", daemon=True)
        self.cam_id = cam_id
        self.source = source
        self.target_fps = float(target_fps) if target_fps > 0 else 5.0
        self.interval = 1.0 / self.target_fps
        self.loop_file = loop_file

        cfg = (system_cfg or {}).get("stream", {}) or {}
        self.open_timeout_ms = int(cfg.get("open_timeout_ms", 5000))
        self.read_timeout_ms = int(cfg.get("read_timeout_ms", 3000))
        self.stall_timeout = float(cfg.get("stall_timeout_sec", 10))
        self.reconnect_delays = list(cfg.get("reconnect_delays", [1, 2, 4, 8, 16, 32, 60]))
        self.offline_after = int(cfg.get("offline_after_retries", 3))
        self.recover_frames = int(cfg.get("recover_frames", 10))

        self.buffer = DoubleBuffer()
        self.stats = ReaderStats()

        self._on_event = on_event
        self._running = False
        self._cap: cv2.VideoCapture | None = None
        self._is_file = False
        self._seq = 0
        self.source_fps = 0.0     # _open() 成功后才赋值，此处先占位避免属性缺失
        self._ok_streak = 0       # 断流后连续成功帧数，达到 recover_frames 才判定恢复

        # https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html
        # 判断源类型：纯数字是摄像头序号，含 :// 是网络流，其余是本地文件
        s = str(source)
        self._is_file = (not s.isdigit()) and ("://" not in s)

    # ---------- 事件上报 ----------
    def _emit(self, event_type: str, detail: dict) -> None:
        if self._on_event:
            try:
                self._on_event(self.cam_id, event_type, detail)
            except Exception as e:  # 回调本身绝不能把拉流线程带崩
                print(f"[{self.cam_id}] 事件回调异常：{e}", file=sys.stderr)

    def _log(self, msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] [{self.cam_id}] {msg}", flush=True)

    # ---------- 打开流 ----------
    @staticmethod
    def _probe_net(url: str, timeout: float = 2.0) -> tuple[bool, str]:
        """打开前先探测 TCP 端口通不通。

        为什么必须做这一步：
            cv2.VideoCapture(url, CAP_FFMPEG) 打开一个不可达的 RTSP 地址时，
            FFMPEG 会在 C 层重试几十秒，而且 OpenCV videoio 内部有全局锁，
            这期间**同进程其他路的 cap.read() 也会被拖慢**（实测三路里两路
            正常流从 24fps 掉到 7fps）。
            先用 socket 探一下端口：不通就直接判失败，代价 2 秒而不是几十秒，
            也不会污染其他路。
        """
        from urllib.parse import urlparse

        try:
            u = urlparse(url)
            host = u.hostname
            if not host:
                return False, "地址无法解析出主机名"
            port = u.port or (554 if u.scheme.startswith("rtsp") else 80)
            with socket.create_connection((host, port), timeout=timeout):
                return True, ""
        except socket.timeout:
            return False, f"端口探测超时（>{timeout}s）"
        except OSError as e:
            return False, f"端口不可达：{e}"

    def _open(self) -> bool:
        """打开一路流。成功返回 True。"""
        src = self.source

        if self._is_file:
            path = Path(str(src))
            if not path.exists():
                self.stats.last_error = f"文件不存在：{path}"
                return False
            # 本地文件不需要超时属性，FFMPEG 后端会忽略
            cap = cv2.VideoCapture(str(path))
        elif str(src).isdigit():
            cap = cv2.VideoCapture(int(src))
        else:
            # 网络流：先做 RTSP 协议探测（不碰 cv2，无全局锁竞争）
            ok, why = self._probe_rtsp(
                str(src), timeout=min(2.0, self.open_timeout_ms / 1000.0)
            )
            if not ok:
                self.stats.last_error = why
                return False

            # 探测通过才进 cv2，此时地址可达，open 通常几十毫秒
            cap = self._open_with_timeout(
                url=str(src),
                timeout=min(3.0, self.open_timeout_ms / 1000.0),
            )
            if cap is None:
                return False

        if not cap.isOpened():
            cap.release()
            self.stats.last_error = "打开失败（地址不可达或编码不支持）"
            return False

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.stats.last_error = ""
        self._cap = cap
        self.source_fps = float(fps)
        return True

    # ---------- 打开流 ----------
    @staticmethod
    def _probe_rtsp(url: str, timeout: float = 2.0) -> tuple[bool, str]:
        """用 RTSP 协议本身探测地址是否可拉流，全程不碰 cv2。

        为什么不能用 TCP connect 探测：
            本机若开了 TUN 模式的代理（SakuraCat / Clash 之类），任何
            socket.connect() 都会被代理抢先接受，探测恒为 True，等于没探。

        为什么必须绕开 cv2 探测：
            cv2.VideoCapture.open() 打开不可达地址会阻塞 19 秒，而且它持有
            OpenCV videoio 的全局锁——这 19 秒内**同进程其他路的 VideoCapture
            构造也全被挡住**，三路里两路正常流会延迟十几秒才启动。
            直接发一个 RTSP OPTIONS 请求，用回包首行判断：
              RTSP/1.0 200 OK / 401 Unauthorized  → 服务在，能连
              空回包 / 连接被关 / 超时              → 服务不在
            这既没有锁竞争，也不依赖 OpenCV 的超时属性（它在 FFMPEG
            后端上 set() 返回 False，等于不存在）。
        """
        from urllib.parse import urlparse

        try:
            u = urlparse(url)
            host = u.hostname
            if not host:
                return False, "地址无法解析出主机名"
            port = u.port or (554 if u.scheme.startswith("rtsp") else 80)
        except Exception as e:
            return False, f"地址解析失败：{e}"

        # 非 RTSP 的 http(s) 流没有 OPTIONS 握手，只能退回 TCP 探测
        if not url.lower().startswith("rtsp://"):
            try:
                with socket.create_connection((host, port), timeout=timeout):
                    return True, ""
            except OSError as e:
                return False, f"端口不可达：{e}"

        try:
            with socket.create_connection((host, port), timeout=timeout) as s:
                s.settimeout(timeout)
                req = (
                    f"OPTIONS {url} RTSP/1.0\r\n"
                    f"CSeq: 1\r\n"
                    f"User-Agent: security-agent-probe\r\n\r\n"
                ).encode("ascii", "ignore")
                s.sendall(req)
                data = s.recv(256)
        except socket.timeout:
            return False, f"RTSP 握手超时（>{timeout}s）"
        except OSError as e:
            return False, f"RTSP 连接失败：{e}"

        if not data:
            return False, "RTSP 服务无响应（连接被对端关闭）"
        head = data[:8].upper()
        if head.startswith(b"RTSP/"):
            return True, ""
        return False, f"对端不是 RTSP 服务（首包：{data[:20]!r}）"

    def _open_with_timeout(self, url: str, timeout: float = 2.0) -> cv2.VideoCapture | None:
        """带硬超时的 open，作为探测之后的第二道保险。

        探测已经挡掉了绝大多数不可达地址，这里兜底两种情况：
          - 探测通过但 cv2 打不开（编码不支持、鉴权失败）
          - 探测与 open 之间网络恰好断开
        超时后那个线程会继续卡在 C 层成为「僵尸 open」，但它不写共享状态，
        只占一个线程栈。因为探测已经把「必然卡死」的情况挡在外面，
        这里真正超时的概率很低，不会有僵尸堆积。
        """
        result: dict[str, Any] = {}

        def _worker() -> None:
            try:
                c = cv2.VideoCapture()
                c.open(url, cv2.CAP_FFMPEG)
                result["cap"] = c
            except Exception as e:      # pragma: no cover - 极少走到
                result["err"] = e

        t = threading.Thread(target=_worker, daemon=True,
                             name=f"open-{self.cam_id}")
        t.start()
        t.join(timeout)

        if t.is_alive():
            with StreamReader._zombie_lock:
                StreamReader._zombie_opens += 1
            self.stats.last_error = f"打开超时（>{timeout:.1f}s）"
            return None

        if "err" in result:
            self.stats.last_error = f"打开异常：{result['err']}"
            return None

        return result.get("cap")

    # 僵尸 open 线程计数，给日志看
    _zombie_opens = 0
    _zombie_lock = threading.Lock()

    def _close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    # ---------- 主循环 ----------
    def run(self) -> None:
        self._running = True
        retry = 0
        last_emit = 0.0
        last_ok_ts = time.monotonic()

        while self._running:
            # ---- 打开（或重开）流 ----
            if self._cap is None:
                prev_state = self.stats.state
                self.stats.state = (
                    StreamState.RECONNECTING if retry > 0 else StreamState.INIT
                )
                if self._open():
                    retry = 0
                    last_ok_ts = time.monotonic()
                    self._log(f"已连接  fps={self.source_fps:.1f}  目标={self.target_fps:.0f}FPS")
                    if prev_state is StreamState.INIT:
                        # 首次连上：直接算在线
                        self.stats.state = StreamState.ONLINE
                        self._emit("stream_online", {
                            "cam_id": self.cam_id,
                            "ts": time.time(),
                        })
                    else:
                        # 断线重连：先停在 RECONNECTING，由连续帧数确认后才宣告恢复。
                        # 否则「能打开」就报恢复，接上又立刻断的抖动流会让状态反复横跳。
                        self.stats.state = StreamState.RECONNECTING
                        self._ok_streak = 0
                else:
                    # 打不开：按退避序列等一会再试
                    delay = self._next_delay(retry)
                    retry += 1
                    self.stats.reconnects += 1
                    self._log(
                        f"连接失败（{self.stats.last_error}），"
                        f"第 {retry} 次重试，{delay}s 后"
                    )
                    if retry >= self.offline_after and self.stats.state is not StreamState.OFFLINE:
                        self.stats.state = StreamState.OFFLINE
                        self.stats.offline_events += 1
                        self._emit("stream_offline", {
                            "cam_id": self.cam_id,
                            "retries": retry,
                            "error": self.stats.last_error,
                            "ts": time.time(),
                        })
                        self._log(f"⚠ 已注册 stream_offline（重试 {retry} 次均失败）")
                    self._sleep(delay)
                    continue

            # ---- 读一帧 ----
            ok, frame = self._cap.read()
            now = time.monotonic()

            if not ok or frame is None:
                # 本地文件读到头：循环播放，否则测 Case2 时 15 秒后就没帧了。
                # set() 可能失败（返回 False），失败还 continue 就是死循环，必须检查。
                if self._is_file and self.loop_file:
                    if self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                        self._ok_streak = 0
                        continue
                    # 回绕失败：当成断流，走下面的重连分支

                # ---- 读帧失败 = 一次连接故障，走与打开失败相同的退避 ----
                self._close()
                self._ok_streak = 0
                stall_sec = now - last_ok_ts
                retry += 1
                self.stats.reconnects += 1
                self.stats.state = StreamState.RECONNECTING

                # stream_stalled 只在真的"卡住"时上报一次，避免退避期间刷屏
                if retry == 1:
                    self._log(f"读帧失败，判定断流（已 {stall_sec:.1f}s 无新帧），准备重连")
                    self._emit("stream_stalled", {
                        "cam_id": self.cam_id,
                        "stall_sec": round(stall_sec, 2),
                        "ts": time.time(),
                    })

                # 两个条件任一满足即标记离线：
                #   ① 连续重试达到 offline_after（要求 3 次）
                #   ② 无新帧时长超过 stall_timeout（要求 10 秒）
                if retry >= self.offline_after or stall_sec >= self.stall_timeout:
                    if self.stats.state is not StreamState.OFFLINE:
                        self.stats.state = StreamState.OFFLINE
                        self.stats.offline_events += 1
                        self._emit("stream_offline", {
                            "cam_id": self.cam_id,
                            "retries": retry,
                            "stall_sec": round(stall_sec, 2),
                            "error": self.stats.last_error,
                            "ts": time.time(),
                        })
                        self._log(
                            f"⚠ 已注册 stream_offline"
                            f"（重试 {retry} 次，{stall_sec:.1f}s 无新帧）"
                        )

                self._sleep(self._next_delay(retry - 1))
                continue

            self.stats.frames_decoded += 1
            last_ok_ts = now

            # 从 OFFLINE/RECONNECTING 恢复：要连续成功 recover_frames 帧才算真的好了。
            # 只看一帧就切 ONLINE 会被抖动流骗（接上一帧又断），导致状态反复横跳。
            if self.stats.state in (StreamState.OFFLINE, StreamState.RECONNECTING):
                self._ok_streak += 1
                if self._ok_streak >= self.recover_frames:
                    self.stats.state = StreamState.ONLINE
                    self._ok_streak = 0
                    retry = 0
                    self._log(f"流已恢复（连续 {self.recover_frames} 帧）")
                    self._emit("stream_online", {
                        "cam_id": self.cam_id,
                        "ts": time.time(),
                    })
            else:
                self._ok_streak = 0

            # ---- 按时间戳降采样到 target_fps ----
            # 不按帧号 step：RTSP 掉帧后帧号会跳，按帧号抽会越抽越乱。
            if now - last_emit < self.interval:
                continue
            last_emit = now

            self._seq += 1
            self.stats.frames_emitted += 1
            self.stats.last_frame_ts = now

            self.buffer.write(FramePacket(
                cam_id=self.cam_id,
                frame=frame,
                ts=now,
                wall_ts=time.time(),
                seq=self._seq,
                source_fps=self.source_fps,
            ))

        self._close()
        self.stats.state = StreamState.STOPPED

    # ---------- 工具 ----------
    def _next_delay(self, retry: int) -> float:
        """取第 retry 次重试的等待秒数，超出序列长度就固定用最后一个。"""
        if not self.reconnect_delays:
            return 1.0
        return float(self.reconnect_delays[min(retry, len(self.reconnect_delays) - 1)])

    def _sleep(self, seconds: float) -> None:
        """可被打断的等待，stop() 时不用等满退避时间。"""
        end = time.monotonic() + max(0.0, seconds)
        while self._running:
            remain = end - time.monotonic()
            if remain <= 0:
                break
            # time.sleep 传负数会抛 ValueError，这里夹到 0.1 上限内
            time.sleep(min(0.1, remain))

    def stop(self) -> None:
        self._running = False

    def status(self) -> dict[str, Any]:
        """给控制台/界面用的状态快照。"""
        stale = (
            time.monotonic() - self.stats.last_frame_ts
            if self.stats.last_frame_ts else float("inf")
        )
        return {
            "cam_id": self.cam_id,
            "source": str(self.source),
            "state": self.stats.state.value,
            "frames_emitted": self.stats.frames_emitted,
            "frames_decoded": self.stats.frames_decoded,
            "decode_ratio": round(self.stats.decode_ratio, 3),
            "reconnects": self.stats.reconnects,
            "offline_events": self.stats.offline_events,
            "idle_sec": round(stale, 2) if stale != float("inf") else None,
            "last_error": self.stats.last_error,
        }

# ============================================================
# 多路管理器
# ============================================================
class StreamManager:
    """管理所有摄像头读取线程。

    用法：
        mgr = StreamManager(system_cfg, cameras_cfg)
        mgr.start(only=["Camera_01", "Camera_02"])
        pkt = mgr.get_frame("Camera_01", timeout=0.5)
        mgr.stop()

    为什么单独包一层：
        上层（app.py / pipeline）不应该知道线程、缓冲、重连这些细节，
        它只想问「Camera_01 现在最新一帧是什么」。
    """

    def __init__(
        self,
        system_cfg: dict,
        cameras_cfg: dict,
        on_event: Callable[[str, str, dict], None] | None = None,
    ) -> None:
        self.system_cfg = system_cfg or {}
        self.cameras_cfg = cameras_cfg or {}
        self._readers: dict[str, StreamReader] = {}
        self._on_event = on_event
        self._events: list[dict] = []
        self._events_lock = threading.Lock()

    # ---------- 事件收集 ----------
    def _handle_event(self, cam_id: str, event_type: str, detail: dict) -> None:
        """收集 stream_* 事件，同时转给外部回调。"""
        with self._events_lock:
            self._events.append({"cam_id": cam_id, "type": event_type, **detail})
        if self._on_event:
            self._on_event(cam_id, event_type, detail)

    def drain_events(self) -> list[dict]:
        """取走累计的流事件（给告警模块消费）。"""
        with self._events_lock:
            out = self._events[:]
            self._events.clear()
        return out

    # ---------- 启动 / 停止 ----------
    def start(self, only: list[str] | None = None) -> list[str]:
        """启动指定摄像头（默认全部 enabled 的），返回实际启动的 id 列表。

        source 解析优先级：
            cameras.yaml 的 source（环境变量已替换）
            → fallback 里的本地视频
        """
        from config_loader import resolve_stream_source

        default_fps = float(self.system_cfg.get("stream", {}).get("target_fps", 5))
        max_streams = int(self.system_cfg.get("stream", {}).get("max_streams", 8))
        started: list[str] = []

        cams = self.cameras_cfg.get("cameras") or {}
        for cam_id, cam in cams.items():
            if only and cam_id not in only:
                continue
            if not cam.get("enabled", True):
                continue
            if cam_id in self._readers:
                continue
            if len(self._readers) >= max_streams:
                print(f"⚠ 已达 max_streams={max_streams}，跳过 {cam_id}", flush=True)
                break

            try:
                src = resolve_stream_source(self.cameras_cfg, cam_id)
            except Exception as e:
                print(f"⚠ [{cam_id}] 无法解析视频源：{e}", flush=True)
                continue

            fps = float(cam.get("target_fps", default_fps))
            fps = min(fps, default_fps)   # 与全局取小，防止现场配错

            reader = StreamReader(
                cam_id=cam_id,
                source=src,
                target_fps=fps,
                system_cfg=self.system_cfg,
                on_event=self._handle_event,
                loop_file=True,
            )
            reader.start()
            self._readers[cam_id] = reader
            started.append(cam_id)

        return started

    def stop(self, timeout: float = 3.0) -> None:
        for r in self._readers.values():
            r.stop()
        for r in self._readers.values():
            r.join(timeout=timeout)
        self._readers.clear()

    # ---------- 取帧 ----------
    def get_frame(self, cam_id: str, timeout: float | None = None) -> FramePacket | None:
        """取某一路的最新帧。超时返回 None。

        timeout=None 表示不等待，只取当前已有的；给 0 也当成立即返回。
        """
        r = self._readers.get(cam_id)
        if r is None:
            return None
        if not timeout:
            return r.buffer.peek()
        return r.buffer.read_latest(timeout)

    def get_all_latest(self) -> dict[str, FramePacket | None]:
        """一次取所有路的最新帧（非阻塞），给多路并行推理用。"""
        return {cid: r.buffer.peek() for cid, r in self._readers.items()}

    # ---------- 状态 ----------
    def status(self) -> list[dict]:
        return [r.status() for r in self._readers.values()]

    def states(self) -> dict[str, str]:
        return {cid: r.stats.state.value for cid, r in self._readers.items()}

    def cam_ids(self) -> list[str]:
        return list(self._readers.keys())

# ============================================================
# 便捷入口：直接跑起来看控制台输出（ Case1 / Case2 手工验证用）
# ============================================================
def _demo() -> None:
    """命令行冒烟测试：python stream_reader.py [--case1|--case2]"""
    import argparse
    import config_loader as C

    ap = argparse.ArgumentParser(description="多路视频异步流读取器冒烟测试")
    ap.add_argument("--case", choices=["1", "2"], default="2",
                    help="1=断流重连, 2=三路稳定性")
    ap.add_argument("--seconds", type=float, default=20.0, help="运行时长")
    args = ap.parse_args()

    system_cfg = C.load_system()
    cameras_cfg = C.load_cameras()

    events: list[str] = []

    def on_event(cam_id: str, etype: str, detail: dict) -> None:
        line = f"  ★ 事件 {cam_id} {etype} {detail}"
        events.append(line)
        print(line, flush=True)

    mgr = StreamManager(system_cfg, cameras_cfg, on_event=on_event)

    if args.case == "1":
        # 指向一个不存在的 RTSP，观察退避日志
        cameras_cfg["cameras"]["Camera_01"]["source"] = "rtsp://10.255.255.1:554/none"
    mgr.start()

    t0 = time.monotonic()
    # 直接用每一路的 emitted 计数差值算 FPS，不要在主循环里 read_latest：
    # 3 路串行阻塞等待会把轮询周期拉长，统计出来偏低（实测 3.7 而实际 4.95）
    base = {cid: r.stats.frames_emitted for cid, r in mgr._readers.items()}
    try:
        while time.monotonic() - t0 < args.seconds:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass

    elapsed = time.monotonic() - t0
    counts = {
        cid: r.stats.frames_emitted - base[cid]
        for cid, r in mgr._readers.items()
    }
    print("\n" + "=" * 60)
    print(f"运行 {elapsed:.1f}s，各路收到帧数：")
    for cid, n in counts.items():
        fps = n / elapsed if elapsed > 0 else 0
        print(f"  {cid}: {n} 帧  ≈ {fps:.2f} FPS")
    print("\n各路状态：")
    for s in mgr.status():
        print(f"  {s['cam_id']}: {s['state']:12} emitted={s['frames_emitted']:4} "
              f"decoded={s['frames_decoded']:5} 重连={s['reconnects']}")
    mgr.stop()

if __name__ == "__main__":
    _demo()
