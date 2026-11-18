# -*- coding: utf-8 -*-
"""
AlertAgent ——

职责：
    把告警推送出去，并保存证据截图。
    告警内容必须包含要求的三项：时间、地点、事件类型。

推送通道：
    console —— 控制台彩色输出（默认，零依赖）
    file    —— 追加写入 JSONL 日志，便于事后追溯

告警同时会画框截图存到 outputs/events/，作为「留证」。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

LEVEL_COLOR = {
    "critical": (0, 0, 255),
    "high": (0, 165, 255),
    "medium": (0, 255, 255),
    "low": (0, 255, 0),
}

# 按类别配色：同一张图里各类目标一眼可区分
CLASS_COLOR = {
    "fire": (0, 0, 255),          # 红 —— 明火
    "smoke": (0, 140, 255),       # 橙 —— 烟雾
    "hardhat": (0, 200, 0),       # 绿 —— 合规
    "no-hardhat": (0, 0, 255),  # 品红 —— 违规
    "person": (255, 255, 0),      # 青 —— 人员（睡岗 / 徘徊）
}
DEFAULT_COLOR = (0, 165, 255)

def color_for(class_name: str, level: str = "high"):
    """优先按类别取色，未登记的类别退回按等级取色。"""
    if class_name in CLASS_COLOR:
        return CLASS_COLOR[class_name]
    return LEVEL_COLOR.get(level, DEFAULT_COLOR)

class AlertAgent:
    """告警 Agent：告警内容 → 控制台/文件 + 证据截图。"""

    def __init__(self, scene_cfg: dict, snapshot_dir: Path,
                 channels: tuple[str, ...] = ("console", "file"),
                 location: str = "", webhook_url: str = "",
                 webhook_timeout: float = 3.0) -> None:
        """
        Args:
            scene_cfg:     场景配置
            snapshot_dir:  截图保存目录
            channels:      推送通道，可选 console / file / webhook
            location:      摄像头位置/点位名，写入告警的「地点」字段
            webhook_url:   钉钉/企业微信/通用 HTTP 机器人地址；空串表示不推
            webhook_timeout: 单次 POST 超时（秒），失败不影响主链路
        """
        self.scene = scene_cfg
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.channels = channels
        self.location = location or f"{scene_cfg['display_name']}点位"
        self.webhook_url = (webhook_url or "").strip()
        self.webhook_timeout = float(webhook_timeout)

        self.alert_log = self.snapshot_dir / "alerts.jsonl"
        self.count = 0
        self.pushed = 0
        self.push_failed = 0
        self.snapshots: list[str] = []

    def push(self, alert: dict, frame: np.ndarray | None = None,
             video_name: str = "") -> dict:
        """推送一条告警，返回补全了地点/截图路径的告警对象。"""
        self.count += 1

        record = dict(alert)
        record["alert_id"] = f"A{self.count:04d}"
        record["location"] = self.location
        record["video"] = video_name
        record["wall_clock"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 视频内时间格式化，便于定位回放位置
        ts = float(record.get("timestamp", 0.0))
        record["video_time"] = f"{int(ts // 60):02d}:{ts % 60:05.2f}"

        if frame is not None:
            path = self._save_snapshot(frame, record)
            if path:
                record["snapshot"] = path
                self.snapshots.append(path)

        if "console" in self.channels:
            self._to_console(record)
        if "file" in self.channels:
            self._to_file(record)
        if "webhook" in self.channels and self.webhook_url:
            self._to_webhook(record)

        return record

    def _save_snapshot(self, frame: np.ndarray, record: dict) -> str:
        """把告警帧画上标注后存盘，作为证据。

        - 画出本帧**全部**检测结果：合规目标（绿）+ 违规目标（高亮）
          只画违规会让用户误以为「戴帽的人没被检出」
        - 按类别配色：fire 红 / smoke 橙 / person 青 / no-hardhat 品红 / hardhat 绿
        - segment 任务对违规目标画半透明掩码 + 轮廓
        """
        vis = frame.copy()
        violation_ids = {int(record.get("class_id", -1))}

        # 优先用 all_targets（本帧该类全部违规目标）
        targets = record.get("all_targets") or [{
            "class_name": record.get("class_name", ""),
            "class_id": record.get("class_id", -1),
            "bbox": record.get("bbox"),
            "score": record.get("max_score", 0),
            "mask_polygons": record.get("mask_polygons"),
            "area_ratio": record.get("area_ratio"),
        }]

        # ---------- 先画合规目标（细框、绿色）----------
        frame_dets = record.get("frame_detections") or []
        for d in frame_dets:
            cid = int(d.get("class_id", -1))
            if cid in violation_ids:
                continue                      # 违规目标稍后高亮画
            bbox = d.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = (int(v) for v in bbox)
            c = color_for(str(d.get("class_name", "")), "low")
            cv2.rectangle(vis, (x1, y1), (x2, y2), c, 1)

        # ---------- 再画违规目标（粗框 + 掩码）----------
        for t in targets:
            cls_name = str(t.get("class_name", ""))
            color = color_for(cls_name, record.get("level", "high"))

            polygons = t.get("mask_polygons") or []
            if polygons:
                overlay = vis.copy()
                for poly in polygons:
                    pts = np.array(poly, dtype=np.int32)
                    if len(pts) >= 3:
                        cv2.fillPoly(overlay, [pts], color)
                vis = cv2.addWeighted(overlay, 0.15, vis, 0.85, 0)
                for poly in polygons:
                    pts = np.array(poly, dtype=np.int32)
                    if len(pts) >= 3:
                        cv2.polylines(vis, [pts], True, color, 3)

            bbox = t.get("bbox")
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = (int(v) for v in bbox)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # ---------- 左上角信息面板 ----------
        # 不把标签贴在每个框上：框贴边时会被裁掉，且同色背景上看不清。
        # 违规目标在前（高亮），合规目标在后（浅色）。
        lines = []
        for t in targets:
            s = f"{t.get('class_name', '')} {t.get('score', 0):.2f}"
            if t.get("area_ratio"):
                s += f"  面积{t['area_ratio'] * 100:.1f}%"
            lines.append((s, color_for(str(t.get("class_name", "")), record.get("level", "high"))))

        ok_targets = [d for d in frame_dets if int(d.get("class_id", -1)) not in violation_ids]
        for d in ok_targets:
            s = f"{d.get('class_name', '')} {d.get('score', 0):.2f}"
            lines.append((s, (150, 190, 150)))     # 合规目标用暗绿，不抢视觉焦点

        if lines:
            pad, lh = 8, 26
            box_w = max(cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)[0][0] for s, _ in lines) + pad * 2
            box_h = lh * len(lines) + pad
            top = 40                                   # 避开顶部横幅
            cv2.rectangle(vis, (6, top), (6 + box_w, top + box_h), (20, 20, 20), -1)
            cv2.rectangle(vis, (6, top), (6 + box_w, top + box_h), (90, 90, 90), 1)
            for i, (s, c) in enumerate(lines):
                cv2.putText(vis, s, (6 + pad, top + pad + lh * i + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, c, 2)

        banner = f"{record['video_time']}  {record.get('title', '')}  @{record['location']}"
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(vis, banner, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        fname = f"{record['alert_id']}_{record['event_type']}_{record['video_time'].replace(':', '-').replace('.', '_')}.jpg"
        path = self.snapshot_dir / fname
        ok = cv2.imwrite(str(path), vis)
        return str(path) if ok else ""

    def _to_console(self, record: dict) -> None:
        color = LEVEL_COLOR.get(record.get("level", "high"), (0, 165, 255))
        try:
            from colorama import Fore, Style
            cmap = {
                "critical": Fore.RED,
                "high": Fore.YELLOW,
                "medium": Fore.CYAN,
                "low": Fore.GREEN,
            }
            c = cmap.get(record.get("level", "high"), Fore.YELLOW)
            head = f"{c}[{record['level'].upper()}]{Style.RESET_ALL}"
        except Exception:
            head = f"[{record.get('level', 'high').upper()}]"

        print(
            f"{head} {record['alert_id']} | {record['wall_clock']} | "
            f"视频时间 {record['video_time']} | {record['location']} | "
            f"{record['title']} | {record.get('class_name', '')} "
            f"score={record.get('max_score', 0):.2f} 连续{record.get('consecutive_frames', 1)}帧"
        )
        if record.get("snapshot"):
            print(f"        证据截图: {record['snapshot']}")

    def _to_file(self, record: dict) -> None:
        """追加一行 JSON，便于日志系统采集。"""
        with open(self.alert_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _to_webhook(self, record: dict) -> None:
        """POST 到钉钉/企业微信群机器人。

        钉钉的 text 消息要求内容里带机器人的自定义关键词，这里统一用
        「安防告警」开头；字段格式对钉钉和企业微信都兼容。失败只计数，
        绝不因为网络问题打断检测主链路。
        """
        text = (
            f"安防告警 · {record.get('level', 'high').upper()}\n"
            f"点位：{record.get('location', '')}\n"
            f"事件：{record.get('title', '')}"
            f"（{record.get('class_name', '')} 置信度 {record.get('max_score', 0):.2f}）\n"
            f"时间：{record.get('wall_clock', '')}（视频内 {record.get('video_time', '')}）\n"
            f"告警号：{record.get('alert_id', '')}"
        )
        try:
            import requests

            resp = requests.post(
                self.webhook_url,
                json={"msgtype": "text", "text": {"content": text}},
                timeout=self.webhook_timeout,
            )
            if 200 <= resp.status_code < 300:
                self.pushed += 1
            else:
                self.push_failed += 1
        except Exception:
            self.push_failed += 1

    def summary(self) -> dict:
        """告警汇总。"""
        return {
            "total_alerts": self.count,
            "snapshots": len(self.snapshots),
            "pushed": self.pushed,
            "push_failed": self.push_failed,
            "log_file": str(self.alert_log) if self.count else "",
        }
