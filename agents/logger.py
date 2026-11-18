# -*- coding: utf-8 -*-
"""
统一日志入口

为什么需要：
    代码里有 10 处 `except Exception: pass`，其中多数是「合理降级」
    （摄像头不支持某属性、ffmpeg 找不到就换方案）。
    这些地方**不能改变容错行为**，否则会让原本能跑的环境跑不起来。
    但完全静默会导致出问题时无处可查——所以改成「行为不变，但留一条记录」。

用法：
    from agents.logger import get_logger
    log = get_logger(__name__)
    log.debug("摄像头不支持 BUFFERSIZE，已忽略: %s", e)

级别控制：
    环境变量 DSH_LOG_LEVEL=DEBUG 可打开详细日志，默认 INFO。
"""
from __future__ import annotations

import logging
import os

_CONFIGURED = False


def _configure_once() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.environ.get("DSH_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))

    root = logging.getLogger("anf")
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)
    root.propagate = False

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """取一个挂在 anf 命名空间下的 logger。"""
    _configure_once()
    # 去掉 agents. 前缀，日志里更短
    short = name.replace("agents.", "").replace("__main__", "main")
    return logging.getLogger(f"anf.{short}")
