"""边缘告警推送落点：本地 JSONL 落盘 + 可选 Webhook。

`EventDispatcher` 原本是往后端 HTTP 接口 POST 的，独立部署时后端不一定在，
所以这里给它换一个 `post_fn`：

1. 所有事件无条件追加到 `outputs/events/edge_alerts.jsonl`，保证告警不丢；
2. 配了 `alert.webhook_url` 就顺手推一条钉钉/企业微信文本消息；
3. 任何一步失败都不抛异常，只返回 False 让调度器自己计数。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

_LEVEL_CN = {
    "critical": "严重",
    "high": "高",
    "medium": "中",
    "low": "低",
}


def _event_text(payload: dict[str, Any]) -> str:
    """把边缘事件拼成一句人能读的推送文案。"""
    level = _LEVEL_CN.get(str(payload.get("severity", "")), str(payload.get("severity", "")))
    cam = payload.get("camera_id", "")
    rule = payload.get("violation_type", "")
    msg = payload.get("message") or f"{rule} 告警"
    vt = payload.get("video_time")
    extra = f"，视频内 {vt}s" if vt is not None else ""
    snap = payload.get("snapshot_path")
    tail = f"\n证据：{Path(snap).name}" if snap else ""
    return f"安防告警 · {level}\n点位：{cam}\n事件：{msg}{extra}\n告警号：{payload.get('event_id', '')}{tail}"


def make_post_fn(jsonl_path: str | Path, webhook_url: str = "",
                 timeout: float = 3.0) -> Callable[[str, dict, float], bool]:
    """返回一个可直接喂给 `EventDispatcher` 的上报函数。"""
    path = Path(jsonl_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hook = (webhook_url or "").strip()

    def _post(url: str, payload: dict, t: float) -> bool:  # noqa: ARG001
        ok = True
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            ok = False

        if not hook:
            return ok

        try:
            import requests

            resp = requests.post(
                hook,
                json={"msgtype": "text", "text": {"content": _event_text(payload)}},
                timeout=timeout,
            )
            ok = ok and 200 <= resp.status_code < 300
        except Exception:
            ok = False
        return ok

    return _post
