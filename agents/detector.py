# -*- coding: utf-8 -*-
"""
DetectionAgent —— 工单编号：CV专高5-10（工单 02 - 任务 4）

职责：
    对单帧图像做目标检测，输出统一格式的结果列表。

设计要点：
    1. 同一个 Agent 同时支持 .pt / .onnx / .engine 三种权重
    2. 同时支持 detect 和 segment 两种任务（火灾模型是 segment），
       统一取其 boxes 输出，segment 的 masks 作为可选字段
    3. 类别名从模型自身读取，再与场景配置校验，避免「类别对不上」导致漏报

输出格式（工单 01 任务 3 规定的接口）：
    [
        {"class": "no-hardhat", "class_id": 1, "bbox": [x1, y1, x2, y2], "score": 0.87},
        ...
    ]
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from agents.geometry import bbox_area, bbox_ios, bbox_iou
from agents.stabilizer import DetectionStabilizer


class DetectionAgent:
    """检测 Agent：一帧图像 → 检测结果列表。"""

    def __init__(self, scene_cfg: dict, warmup: bool = True,
                 load_model: bool = True) -> None:
        """
        Args:
            scene_cfg:  来自 config_loader.get_scene() 的场景配置
            warmup:     是否预热一次推理（TensorRT 首次推理较慢，预热后稳定）
            load_model: 是否加载 YOLO 权重；安全帽三路这类自定义后端传 False，
                        只借用这里的阈值、稳定器与尺寸过滤配置
        """
        from ultralytics import YOLO

        self.scene = scene_cfg
        self.name = scene_cfg["name"]
        self.model_path = scene_cfg["model"]
        self.imgsz = int(scene_cfg["imgsz"])
        self.conf = float(scene_cfg["conf"])
        self.iou = float(scene_cfg["iou"])
        self.task = scene_cfg["task"]
        self.class_map = {int(k): v for k, v in scene_cfg["classes"].items()}

        # 用户在界面上设定的阈值（可能高于 self.conf）
        # self.conf 是推理时的下限，user_conf 是最终过滤阈值
        self.user_conf = float(scene_cfg.get("user_conf", self.conf))
        self.violation_ids = set(int(c) for c in scene_cfg.get("violation_classes", []))

        # ------------------------------------------------------------
        # 后处理参数：从 scenes.yaml 的 postprocess 块读取
        # 以前这些值是硬编码的（use_stabilizer = task=="detect" and name=="helmet"），
        # 加新场景必须改代码。现在新场景只要在 YAML 里打开开关即可复用，
        # 不写这个块就全套关闭，行为与改动前一致。
        # ------------------------------------------------------------
        pp = scene_cfg.get("postprocess") or {}
        self.pp = pp
        self.use_stabilizer = bool(pp.get("enable_stabilizer", False))
        self.hardhat_floor = float(pp.get("hardhat_floor", 0.0))
        self.violation_floor = float(pp.get("violation_floor", self.conf))
        self.min_show = float(pp.get("min_show", 0.0))
        self.max_box_ratio = float(pp.get("max_box_ratio", 1.0))
        self.min_box_px = int(pp.get("min_box_px", 0))

        stab_cfg = pp.get("stabilize") or {}
        self.stabilizer = DetectionStabilizer(
            iou_thres=float(stab_cfg.get("iou_thres", 0.30)),
            dist_ratio=float(stab_cfg.get("dist_ratio", 0.8)),
            max_lost=int(stab_cfg.get("max_lost", 2)),
            min_hits=int(stab_cfg.get("min_hits", 2)),
            track_min_score=float(stab_cfg.get("track_min_score", 0.15)),
        )

        self.model = None
        self.backend = ""
        self.infer_ms: list[float] = []
        if not load_model:
            return

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"模型文件不存在：{self.model_path}")

        self.model = YOLO(self.model_path)

        # 校验模型自带类别与配置是否一致，不一致会导致告警逻辑完全错位
        model_names = getattr(self.model, "names", {}) or {}
        self._verify_classes(model_names)

        self.backend = Path(self.model_path).suffix.lower()

        if warmup:
            self._warmup()

    def _verify_classes(self, model_names: dict) -> None:
        """核对模型类别与场景配置，不一致直接报错。"""
        if not model_names:
            return
        mismatch = []
        for cid, cfg_name in self.class_map.items():
            model_name = model_names.get(cid)
            if model_name is None:
                mismatch.append(f"模型缺少类别 id={cid}")
            elif str(model_name).lower() != str(cfg_name).lower():
                mismatch.append(f"id={cid} 配置为 '{cfg_name}'，模型为 '{model_name}'")
        if mismatch:
            raise ValueError(
                f"场景 [{self.name}] 的类别配置与模型不一致：\n  "
                + "\n  ".join(mismatch)
                + f"\n请修正 config/scenes.yaml 中的 classes 段"
            )

    def _warmup(self) -> None:
        """用一张空图预热，避免首帧耗时异常。"""
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        try:
            self.model.predict(dummy, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                               verbose=False, device=0)
        except Exception:
            # 预热失败不阻断流程（例如 CPU 环境），只是首次推理会慢
            pass

    def _infer(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """推理 + 结果标准化。

        默认走 ultralytics YOLO；子类覆写它就能换成别的后端
        （安全帽三路检测就是这么接进来的），而阈值、稳定器、
        尺寸合理性这些后处理完全不用改。
        """
        results = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            device=0,
        )
        if not results:
            return []
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []

        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        out: list[dict[str, Any]] = []
        for bbox, score, cid in zip(boxes, scores, classes):
            out.append({
                "class": self.class_map.get(int(cid), f"class_{cid}"),
                "class_id": int(cid),
                "bbox": [round(float(v), 2) for v in bbox],
                "score": round(float(score), 4),
            })

        # segment 任务：把掩码转成多边形轮廓一起输出
        # 为什么不直接传 mask 数组：960x960 的 bool 数组接近 1MB，
        # 放进 LangGraph State 会拖慢序列化；转成几十个轮廓点后只有几百字节。
        if self.task == "segment":
            self._attach_mask_polygons(r, out, frame.shape[:2])

        # pose 任务：附带关键点与姿态指标（睡岗场景用）
        if self.task == "pose":
            self._attach_keypoints(r, out)
        return out

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """执行检测，返回标准化结果列表。"""
        t0 = time.perf_counter()
        out = self._infer(frame)
        self.infer_ms.append((time.perf_counter() - t0) * 1000)

        if not out:
            return []

        # ---------- 按类别卡阈值 ----------
        # 开启稳定器的场景（当前只有安全帽）做分类阈值：
        #   正常类别抬到 hardhat_floor（砍掉「头发当帽子」的误检）
        #   违规类别压到 violation_floor（模型犹豫，靠稳定器补）
        # 未开启的场景保持用户设定的统一阈值，行为与改动前一致。
        if self.use_stabilizer:
            filtered = []
            for d in out:
                is_violation = int(d["class_id"]) in self.violation_ids
                if is_violation:
                    thr = min(self.violation_floor, self.user_conf)
                else:
                    thr = max(self.user_conf, self.hardhat_floor)
                if float(d["score"]) >= thr:
                    filtered.append(d)
        else:
            # 其他场景：全部用用户阈值，行为与改动前一致
            filtered = [d for d in out if float(d["score"]) >= self.user_conf]

        # ---------- 时间稳定 ----------
        if self.use_stabilizer:
            stable = self.stabilizer.update(filtered)
            stable = self._resolve_class_conflict(stable)
        else:
            stable = filtered

        # ---------- 显示门槛 + 尺寸合理性 ----------
        min_show = self.min_show if self.use_stabilizer else self.user_conf
        h_img, w_img = frame.shape[:2]
        img_area = float(h_img * w_img) or 1.0

        result = []
        for d in stable:
            if float(d.get("score", 0)) < min_show:
                continue
            # 尺寸检查只对开了稳定器的场景（火灾的掩码区域本来就可能很大）
            if self.use_stabilizer:
                x1, y1, x2, y2 = d["bbox"]
                bw = max(0.0, x2 - x1)
                bh = max(0.0, y2 - y1)
                lim = self.min_box_px
                if lim > 0 and (bw < lim or bh < lim):
                    continue
                if (bw * bh) / img_area > self.max_box_ratio:
                    continue
            result.append(d)
        return result

    @staticmethod
    def _resolve_class_conflict(dets: list[dict]) -> list[dict]:
        """同一目标被同时标成 hardhat 和 no-hardhat 时，只留一个。

        用两种重叠度量：
          IOU (intersection / union)      —— 两个框大小接近时有效
          IOS (intersection / smaller)    —— 小框被大框包住时有效

        典型场景：人头上有一个 0.74 的 hardhat 小框（只框了头发尖），
        和一个 0.35 的 no-hardhat 大框（框住整个头）。
        两者 IOU 只有 0.05，但 IoS 接近 1，属于同一目标的矛盾判断。
        此时保留尺寸更合理的那个（大的），删掉小框。

        几何计算走 agents.geometry，与稳定器共用同一份实现。
        """
        hardhats = [d for d in dets if d.get("class") == "hardhat"]
        if not hardhats:
            return dets

        drop_hardhat: set[int] = set()
        drop_nohat: set[int] = set()

        for i, n in enumerate(dets):
            if n.get("class") != "no-hardhat":
                continue
            nbox = n.get("bbox")
            if not nbox or len(nbox) != 4:
                continue        # 防御：脏数据不参与冲突判断
            na = bbox_area(nbox)
            for h in hardhats:
                hbox = h.get("bbox")
                if not hbox or len(hbox) != 4:
                    continue
                ha = bbox_area(hbox)
                overlap = (bbox_iou(nbox, hbox) > 0.25
                           or bbox_ios(nbox, hbox) > 0.60)
                if not overlap:
                    continue
                # 同一目标：保留尺寸更合理的（大的）。
                # 小框通常是「只框到头发尖」的局部误检。
                if ha < na * 0.5:
                    drop_hardhat.add(id(h))
                elif na < ha * 0.5:
                    drop_nohat.add(i)
                else:
                    # 尺寸接近，按置信度保留高的
                    if float(h.get("score", 0)) >= float(n.get("score", 0)):
                        drop_nohat.add(i)
                    else:
                        drop_hardhat.add(id(h))

        out = []
        for i, d in enumerate(dets):
            if i in drop_nohat:
                continue
            if d.get("class") == "hardhat" and id(d) in drop_hardhat:
                continue
            out.append(d)
        return out

    def reset_stabilizer(self) -> None:
        """换视频 / 重开摄像头时清空轨迹，避免上一段的框串到下一段。"""
        self.stabilizer.reset()

    # COCO 17 关键点索引
    KP_NOSE, KP_L_SHOULDER, KP_R_SHOULDER = 0, 5, 6

    def _attach_keypoints(self, result, detections: list[dict]) -> None:
        """从姿态结果中提取关键点，并计算「低头」指标。

        指标定义：
            head_drop = (鼻尖 y - 肩中点 y) / 肩宽
            - 正常坐姿/站姿：鼻尖在肩膀上方，该值为负（约 -0.3 ~ -0.6）
            - 低头打瞌睡：鼻尖下降到接近或低于肩线，该值趋近 0 或转正
        """
        kps = getattr(result, "keypoints", None)
        if kps is None or kps.data is None:
            return

        data = kps.data.cpu().numpy()   # (N, 17, 3)
        for i, item in enumerate(detections):
            if i >= len(data):
                break
            kp = data[i]
            if kp.shape[-1] < 2:
                continue
            points = [
                [round(float(p[0]), 1), round(float(p[1]), 1),
                 round(float(p[2]), 3) if len(p) > 2 else 1.0]
                for p in kp
            ]
            item["keypoints"] = points

            nose = points[self.KP_NOSE]
            ls = points[self.KP_L_SHOULDER]
            rs = points[self.KP_R_SHOULDER]
            if min(nose[2], ls[2], rs[2]) >= 0.3:
                shoulder_y = (ls[1] + rs[1]) / 2.0
                shoulder_w = abs(rs[0] - ls[0])
                if shoulder_w >= 8:
                    item["head_drop"] = round((nose[1] - shoulder_y) / shoulder_w, 3)
                    item["shoulder_width"] = round(shoulder_w, 1)

    def _attach_mask_polygons(self, result, detections: list[dict],
                              frame_hw: tuple[int, int]) -> None:
        """从分割结果提取每个目标的掩码轮廓。

        为什么用 masks.data 而不是 masks.xy：
            masks.xy 只给「外轮廓」，模型把烟雾判成多块时会产生多个独立多边形，
            中间的浅色区域没覆盖，看起来像「掩码有空洞」。
            用 masks.data（像素级掩码）+ 形态学闭运算能把洞里填上。

        为什么必须处理 letterbox：
            masks.data 的形状是模型输入尺寸（如 960x960），原图是 letterbox 进去的
            （上下有灰边）。直接 resize 到原图会把灰边也拉进去，导致掩码垂直压缩、
            上半部分错位——这就是之前「烟雾从中间被切开」的根因。
        """
        masks = getattr(result, "masks", None)
        if masks is None:
            return

        import cv2

        h, w = frame_hw
        frame_area = float(h * w) or 1.0
        min_area = max(30, int(frame_area * 0.0003))

        data = getattr(masks, "data", None)
        if data is None:
            return
        data = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
        if len(data) == 0:
            return

        mh, mw = data.shape[1], data.shape[2]

        # ---------- 计算 letterbox 参数 ----------
        # Ultralytics 保持宽高比缩放后居中填充
        scale = min(mw / w, mh / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        pad_x = (mw - new_w) // 2
        pad_y = (mh - new_h) // 2
        need_crop = (new_w != w or new_h != h)

        for i, item in enumerate(detections):
            if i >= len(data):
                break

            mask = data[i]

            # ---------- 一步直接缩到「处理尺寸」，避免先放大再缩小 ----------
            # 之前是：裁 letterbox → resize 到 1280x720 → resize 到 640x360，
            # 中间那次放大纯属浪费（0.3ms × 12 个 mask）。
            # 现在直接裁 letterbox 后缩到 (w//DS, h//DS)，轮廓坐标再乘回 DS。
            #
            # DS=2 是实测的平衡点：DS=1 要 16ms，DS=2 降到 4.7ms，
            # 但 DS=3/4 只再省 2ms（在测量噪声范围内），
            # 代价是小火苗（30x30）缩完只剩几个像素，轮廓会明显失真。不划算。
            DS = 2
            tw, th = max(1, w // DS), max(1, h // DS)

            if need_crop:
                y1 = max(0, pad_y)
                y2 = min(mh, pad_y + new_h)
                x1 = max(0, pad_x)
                x2 = min(mw, pad_x + new_w)
                src = mask[y1:y2, x1:x2]
            else:
                src = mask

            small = cv2.resize(src, (tw, th), interpolation=cv2.INTER_LINEAR)
            small = (small > 0.5).astype(np.uint8)

            px_small = int(np.count_nonzero(small))
            if px_small == 0:
                continue

            # 形态学：只对大面积 mask 做（小火苗没有内部空洞，闭运算纯属浪费）
            # px_small 换算回原图面积做判断
            area_ratio_raw = (px_small * DS * DS) / frame_area
            if area_ratio_raw > 0.02:
                k = max(3, int(min(tw, th) * 0.015))
                if k % 2 == 0:
                    k += 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(small, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:6]
            polygons = []
            kept_area = 0.0
            min_area_small = max(5, min_area // (DS * DS))
            for cnt in contours:
                a = cv2.contourArea(cnt)
                if a < min_area_small:
                    continue
                eps = 0.002 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
                if len(approx) >= 3:
                    polygons.append([[int(x) * DS, int(y) * DS]
                                     for x, y in approx])
                    kept_area += a * DS * DS

            if polygons:
                item["mask_polygons"] = polygons
                item["mask_area"] = int(kept_area)
                item["frame_area"] = int(frame_area)
                item["area_ratio"] = round(float(kept_area) / frame_area, 4)

    def stats(self) -> dict[str, float]:
        """返回推理耗时统计。"""
        if not self.infer_ms:
            return {"count": 0}
        arr = np.array(self.infer_ms)
        return {
            "count": len(arr),
            "mean_ms": float(arr.mean()),
            "median_ms": float(np.median(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "fps": float(1000.0 / np.median(arr)) if np.median(arr) > 0 else 0.0,
        }
