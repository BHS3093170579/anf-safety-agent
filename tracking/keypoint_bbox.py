"""从 17 个骨骼关键点反推人体外接检测框（BBox）。

工单编号：CV项目-安防智能体-03-边缘端—ModelPool+显存控制与人体 BBox 重构追踪

为什么要有这一步（对应工单第 2 节「极限显存瘦身」）：
    常规做法是「detect 模型找人 → pose 模型估骨架」两个模型，
    显存翻倍。工单要求只加载 pose 一个模型，那人的框从哪来？
    答案就是：**从 17 个关键点自己算出外接矩形**。

    这样做的附带好处是抗遮挡：detect 模型一旦被人挡住半身就检不到，
    但 pose 只要还剩几个关键点（哪怕是单侧肩膀+髋），
    就能推一个略小的框出来，交给跟踪器用运动预测补齐。

关键点排列（COCO-17，ultralytics pose 输出顺序）：
    0 nose        1 left_eye    2 right_eye   3 left_ear    4 right_ear
    5 l_shoulder  6 r_shoulder  7 l_elbow     8 r_elbow
    9 l_wrist    10 r_wrist    11 l_hip      12 r_hip
   13 l_knee     14 r_knee     15 l_ankle    16 r_ankle

纯关键点算出来的框会比真实人体框「瘦」：头顶秃出去一块、脚尖也够不到。
所以这里做了两处几何外扩，用骨架自身的尺度当尺子（不依赖任何像素常数）：
    · 头部：以鼻/眼为基准向上扩，长度取「肩中点到鼻的距离」
    · 脚部：以踝为基准向下扩，长度取「髋中点到踝距离」的一小部分
    · 左右：以肩宽为准略微向外扩
"""

from __future__ import annotations

import numpy as np

# COCO-17 关键点下标
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16

KPT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# 至少几个关键点可见才认为这是一个能出框的人
MIN_VISIBLE_KPTS = 3


def _pair_mid(kpts: np.ndarray, conf: np.ndarray, i: int, j: int, thr: float):
    """取两个关键点的中点，都不可见时返回 None。"""
    a, b = conf[i] >= thr, conf[j] >= thr
    if a and b:
        return (kpts[i] + kpts[j]) / 2.0
    if a:
        return kpts[i].copy()
    if b:
        return kpts[j].copy()
    return None


def keypoints_to_bbox(
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray,
    img_shape: tuple[int, int],
    conf_thr: float = 0.25,
    pad_ratio: float = 0.04,
) -> tuple[list[float], float] | None:
    """把一个人的 17 个关键点换算成外接检测框。

    Args:
        kpts_xy: (17, 2) 关键点像素坐标
        kpts_conf: (17,) 关键点置信度
        img_shape: (h, w) 原图尺寸，用来裁剪越界
        conf_thr: 关键点可见阈值，低于它视为被遮挡
        pad_ratio: 最终框四周额外的相对留白

    Returns:
        ([x1, y1, x2, y2], quality) 或 None（可见点太少，出不了框）
        quality 是该框的置信度，取可见关键点置信度的均值乘一个可见度系数。
    """
    kpts_xy = np.asarray(kpts_xy, dtype=np.float32).reshape(-1, 2)
    kpts_conf = np.asarray(kpts_conf, dtype=np.float32).reshape(-1)
    if kpts_xy.shape[0] < 17:
        return None

    vis = kpts_conf >= conf_thr
    n_vis = int(vis.sum())
    if n_vis < MIN_VISIBLE_KPTS:
        return None

    pts = kpts_xy[vis]
    x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())

    # ---- 用骨架自身的尺度做几何外扩 ----
    sho_mid = _pair_mid(kpts_xy, kpts_conf, L_SHO, R_SHO, conf_thr)
    hip_mid = _pair_mid(kpts_xy, kpts_conf, L_HIP, R_HIP, conf_thr)

    shoulder_w = 0.0
    if kpts_conf[L_SHO] >= conf_thr and kpts_conf[R_SHO] >= conf_thr:
        shoulder_w = float(np.linalg.norm(kpts_xy[L_SHO] - kpts_xy[R_SHO]))
    elif sho_mid is not None and hip_mid is not None:
        # 单肩不可见时，用肩到髋的距离近似肩宽（人体比例约 1:1.4）
        shoulder_w = float(np.linalg.norm(sho_mid - hip_mid)) * 0.7

    # 头部：鼻尖可见时向上扩「肩中→鼻」的距离，头顶不会被切
    head_up = 0.0
    if kpts_conf[NOSE] >= conf_thr and sho_mid is not None:
        head_up = float(np.linalg.norm(kpts_xy[NOSE] - sho_mid))
    elif sho_mid is not None and shoulder_w > 0:
        head_up = shoulder_w * 0.8          # 鼻不可见（背对/低头）时的兜底
    y1 -= head_up

    # 脚部：踝可见时向下补一点点足长
    foot_down = 0.0
    if kpts_conf[L_ANK] >= conf_thr or kpts_conf[R_ANK] >= conf_thr:
        foot_down = max(shoulder_w * 0.25, 6.0)
    elif kpts_conf[L_KNEE] >= conf_thr or kpts_conf[R_KNEE] >= conf_thr:
        # 只看到膝盖（下半身被挡），按「膝→髋」距离的下半段估一条小腿
        if hip_mid is not None:
            knee_pts = [kpts_xy[i] for i in (L_KNEE, R_KNEE) if kpts_conf[i] >= conf_thr]
            knee_mid = np.mean(knee_pts, axis=0)
            foot_down = float(np.linalg.norm(knee_mid - hip_mid)) * 0.6
    y2 += foot_down

    # 左右：肩宽向外各扩一点，手臂张开时不够宽也不会切掉
    if shoulder_w > 0:
        side = shoulder_w * 0.25
        x1 -= side
        x2 += side

    # ---- 相对留白 + 越界裁剪 ----
    h, w = img_shape
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    x1 -= bw * pad_ratio
    x2 += bw * pad_ratio
    y1 -= bh * pad_ratio
    y2 += bh * pad_ratio

    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    x2 = min(float(w - 1), x2)
    y2 = min(float(h - 1), y2)

    if x2 - x1 < 8 or y2 - y1 < 16:
        return None

    # ---- 质量分：可见点越多越可信 ----
    quality = float(kpts_conf[vis].mean()) * min(1.0, n_vis / 8.0)
    return [x1, y1, x2, y2], quality


def bbox_from_result(
    kpts_tensor,
    img_shape: tuple[int, int],
    conf_thr: float = 0.25,
) -> list[tuple[list[float], float, np.ndarray]]:
    """从 ultralytics 的 pose 输出批量重构框。

    Args:
        kpts_tensor: Results.keypoints.data，形状 (N, 17, 3)，最后一维是 (x, y, conf)
        img_shape: (h, w)

    Returns:
        [(bbox, quality, kpts_17x2), ...]
    """
    out: list[tuple[list[float], float, np.ndarray]] = []
    if kpts_tensor is None or len(kpts_tensor) == 0:
        return out

    data = kpts_tensor.cpu().numpy() if hasattr(kpts_tensor, "cpu") else np.asarray(kpts_tensor)
    for person in data:
        person = np.asarray(person, dtype=np.float32)
        if person.shape != (17, 3):
            continue
        got = keypoints_to_bbox(person[:, :2], person[:, 2], img_shape, conf_thr=conf_thr)
        if got is None:
            continue
        bbox, quality = got
        out.append((bbox, quality, person[:, :2]))
    return out


def aspect_ok(bbox: list[float], min_ratio: float = 1.2, max_ratio: float = 6.0) -> bool:
    """人形框的长宽比粗筛，把误检成横躺的框丢掉。"""
    w = max(1.0, bbox[2] - bbox[0])
    h = max(1.0, bbox[3] - bbox[1])
    r = h / w
    return min_ratio <= r <= max_ratio
