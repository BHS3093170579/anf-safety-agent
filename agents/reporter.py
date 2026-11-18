# -*- coding: utf-8 -*-
"""
ReportAgent —— 工单编号：CV专高5-10（工单 02）

职责：
    汇总一次视频处理的所有告警，生成结构化报告（JSON + Markdown）。

报告内容对应工单要求：
    - 事件总数、按类型/等级分组
    - 每条告警的时间点（视频内时间 + 墙钟时间）
    - 证据截图路径
    - 处理过程的性能统计（帧数、推理耗时、FPS）
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class ReportAgent:
    """报告 Agent：告警列表 + 运行统计 → 报告文件。"""

    def __init__(self, report_dir: Path, scene_cfg: dict) -> None:
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.scene = scene_cfg

    def build(self, video_info: dict, alerts: list[dict],
              perf: dict | None = None) -> dict[str, Any]:
        """组装报告数据对象。"""
        by_level: dict[str, int] = {}
        by_type: dict[str, int] = {}
        for a in alerts:
            by_level[a.get("level", "unknown")] = by_level.get(a.get("level", "unknown"), 0) + 1
            by_type[a.get("event_type", "unknown")] = by_type.get(a.get("event_type", "unknown"), 0) + 1

        return {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scene": {
                "name": self.scene.get("name"),
                "display_name": self.scene.get("display_name"),
                "model": Path(self.scene.get("model", "")).name,
                "task": self.scene.get("task"),
                "imgsz": self.scene.get("imgsz"),
                "conf": self.scene.get("conf"),
            },
            "video": video_info,
            "summary": {
                "total_alerts": len(alerts),
                "by_level": by_level,
                "by_type": by_type,
                "verdict": "发现异常" if alerts else "未发现异常",
            },
            "performance": perf or {},
            "alerts": alerts,
        }

    def save(self, report: dict, stem: str = "") -> dict[str, str]:
        """把报告写成 JSON 与 Markdown 两份，返回路径。"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{stem or self.scene.get('name', 'report')}_{ts}"

        json_path = self.report_dir / f"{base}.json"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        md_path = self.report_dir / f"{base}.md"
        md_path.write_text(self._to_markdown(report), encoding="utf-8")

        return {"json": str(json_path), "markdown": str(md_path)}

    def _to_markdown(self, report: dict) -> str:
        """把报告渲染成 Markdown。"""
        s = report["summary"]
        v = report["video"]
        p = report.get("performance", {})
        scene = report["scene"]

        lines = [
            f"# 安防智能体告警报告 · {scene.get('display_name', '')}",
            "",
            f"> 生成时间：{report['generated_at']}  ",
            f"> 场景：`{scene.get('name')}`　模型：`{scene.get('model')}`　"
            f"task={scene.get('task')}　imgsz={scene.get('imgsz')}　conf={scene.get('conf')}",
            "",
            "## 一、处理概况",
            "",
            "| 项目 | 值 |",
            "|---|---|",
            f"| 视频文件 | `{Path(v.get('path', '')).name}` |",
            f"| 分辨率 | {v.get('resolution', '-')} |",
            f"| 时长 | {v.get('duration_sec', 0)} 秒 |",
            f"| 原始帧率 | {v.get('source_fps', '-')} fps |",
            f"| 抽帧策略 | {v.get('sample_fps', 1)} fps（每 {v.get('frame_step', 1)} 帧取 1 帧） |",
            f"| 实际处理帧数 | {v.get('processed_frames', 0)} |",
            f"| 告警总数 | **{s['total_alerts']}** |",
            f"| 结论 | **{s['verdict']}** |",
            "",
        ]

        if p:
            lines += [
                "## 二、性能统计",
                "",
                "| 指标 | 值 |",
                "|---|---|",
                f"| 检测耗时（中位） | {p.get('median_ms', 0):.1f} ms |",
                f"| 检测耗时（P95） | {p.get('p95_ms', 0):.1f} ms |",
                f"| 等效 FPS | {p.get('fps', 0):.1f} |",
                f"| 端到端耗时 | {p.get('elapsed_sec', 0):.1f} 秒 |",
                "",
            ]

        if s["total_alerts"]:
            lines += ["## 三、告警明细", "", "| # | 视频时间 | 事件 | 类别 | 置信度 | 地点 | 截图 |",
                      "|---|---|---|---|---|---|---|"]
            for a in report["alerts"]:
                snap = Path(a.get("snapshot", "")).name if a.get("snapshot") else "-"
                lines.append(
                    f"| {a.get('alert_id', '-')} | {a.get('video_time', '-')} | "
                    f"{a.get('title', '-')} | {a.get('class_name', '-')} | "
                    f"{a.get('max_score', 0):.2f} | {a.get('location', '-')} | {snap} |"
                )
            lines.append("")
            lines += ["### 告警事件说明", ""]
            for a in report["alerts"][:20]:
                lines.append(
                    f"- **{a.get('alert_id')}**（{a.get('level')}）：{a.get('message')}，"
                    f"视频 {a.get('video_time')}，共 {a.get('count', 1)} 个目标，"
                    f"连续 {a.get('consecutive_frames', 1)} 帧确认"
                )
        else:
            lines += ["## 三、告警明细", "", "本次处理未发现违规事件。"]

        lines += ["", "---", ""]
        return "\n".join(lines)
