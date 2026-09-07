from __future__ import annotations

from typing import Any

import numpy as np


PIXEL_TO_MM = 0.143


def _parse_vertebra(row: np.ndarray) -> dict[str, Any]:
    return {
        "center": (float(row[0]), float(row[1])),
        "upper": ((float(row[2]), float(row[3])), (float(row[4]), float(row[5]))),
        "lower": ((float(row[6]), float(row[7])), (float(row[8]), float(row[9]))),
        "conf": float(row[10]) if len(row) > 10 else 1.0,
    }


def _make_vector(left: tuple[float, float], right: tuple[float, float]) -> np.ndarray:
    return np.array(right, dtype=np.float32) - np.array(left, dtype=np.float32)


def _signed_angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_theta))
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    return float(angle if cross >= 0 else -angle)


def _infer_lr_from_keypoints(rows: np.ndarray, flat_threshold_px: float = 2.0) -> dict[str, Any]:
    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] < 10:
        return {
            "lr_label": "L",
            "method": "default",
            "confidence": 0.0,
            "detail": "invalid_keypoints",
        }

    sacrum = rows[-1]
    tl_y = float(sacrum[3])
    tr_y = float(sacrum[5])
    dy = tr_y - tl_y

    if dy > flat_threshold_px:
        return {
            "lr_label": "L",
            "method": "sacrum_slope",
            "confidence": 1.0,
            "detail": f"dy={dy:.3f}",
        }
    if dy < -flat_threshold_px:
        return {
            "lr_label": "R",
            "method": "sacrum_slope",
            "confidence": 1.0,
            "detail": f"dy={dy:.3f}",
        }

    sacrum_cx = float(sacrum[0])
    lumbar_count = min(5, max(rows.shape[0] - 1, 0))
    centers = rows[:lumbar_count, 0] if lumbar_count > 0 else np.array([], dtype=float)
    left_count = int(np.sum(centers < sacrum_cx))
    right_count = int(np.sum(centers > sacrum_cx))

    if left_count > right_count:
        label = "R"
    elif right_count > left_count:
        label = "L"
    else:
        label = "L"

    confidence = abs(left_count - right_count) / max(lumbar_count, 1)
    return {
        "lr_label": label,
        "method": "center_vote",
        "confidence": float(confidence),
        "detail": f"dy={dy:.3f},left={left_count},right={right_count},sacrum_cx={sacrum_cx:.3f}",
    }


def _posterior_point(p1: tuple[float, float], p2: tuple[float, float], lr_label: str = "L") -> tuple[float, float]:
    if lr_label == "R":
        return p1 if p1[0] > p2[0] else p2
    return p1 if p1[0] < p2[0] else p2


def _anterior_point(p1: tuple[float, float], p2: tuple[float, float], lr_label: str = "L") -> tuple[float, float]:
    if lr_label == "R":
        return p1 if p1[0] < p2[0] else p2
    return p1 if p1[0] > p2[0] else p2


def _point_line_distance(point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    point_arr = np.array(point, dtype=float)
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    ab = b_arr - a_arr
    norm_ab = np.linalg.norm(ab)
    if norm_ab < 1e-6:
        return float(np.linalg.norm(point_arr - a_arr))
    return float(abs(np.cross(ab, a_arr - point_arr)) / norm_ab)


def _compute_isa(rows: np.ndarray) -> list[dict[str, Any]]:
    vertebrae = [_parse_vertebra(row) for row in rows]
    names = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]
    results: list[dict[str, Any]] = []
    for index in range(4):
        upper = vertebrae[index]
        lower = vertebrae[index + 1]
        angle = _signed_angle_deg(_make_vector(*upper["lower"]), _make_vector(*lower["upper"]))
        results.append(
            {
                "name": f"ISA({names[index]})",
                "angle": round(float(angle), 2),
                "direction": "left" if angle > 0 else "right",
                "line1": upper["lower"],
                "line2": lower["upper"],
                "text_pos": lower["center"],
            }
        )
    angle = _signed_angle_deg(_make_vector(*vertebrae[4]["lower"]), _make_vector(*vertebrae[5]["upper"]))
    results.append(
        {
            "name": "ISA(L5-S1)",
            "angle": round(float(angle), 2),
            "direction": "left" if angle > 0 else "right",
            "line1": vertebrae[4]["lower"],
            "line2": vertebrae[5]["upper"],
            "text_pos": vertebrae[4]["center"],
        }
    )
    return results


def _compute_sd_between(upper: dict[str, Any], lower: dict[str, Any], lr_label: str = "L") -> dict[str, Any]:
    lower_upper_posterior = _posterior_point(*lower["upper"], lr_label=lr_label)
    lower_lower_posterior = _posterior_point(*lower["lower"], lr_label=lr_label)
    upper_lower_posterior = _posterior_point(*upper["lower"], lr_label=lr_label)
    sd = _point_line_distance(upper_lower_posterior, lower_upper_posterior, lower_lower_posterior)
    if lr_label == "R":
        direction = "forward" if upper_lower_posterior[0] > lower_lower_posterior[0] else "backward"
    else:
        direction = "forward" if upper_lower_posterior[0] < lower_lower_posterior[0] else "backward"
    return {
        "sd": round(float(sd) * PIXEL_TO_MM, 2),
        "direction": direction,
        "lineA": (lower_upper_posterior, lower_lower_posterior),
        "pointB": upper_lower_posterior,
    }


def _compute_sd_l5_s1(l5: dict[str, Any], sacrum: dict[str, Any], lr_label: str = "L") -> dict[str, Any]:
    posterior = _posterior_point(*sacrum["upper"], lr_label=lr_label)
    anterior = _anterior_point(*sacrum["upper"], lr_label=lr_label)
    posterior_l5 = _posterior_point(*l5["lower"], lr_label=lr_label)

    sd = _point_line_distance(posterior_l5, posterior, anterior)
    if lr_label == "R":
        direction = "forward" if posterior_l5[0] > posterior[0] else "backward"
    else:
        direction = "forward" if posterior_l5[0] < posterior[0] else "backward"

    return {
        "sd": round(float(sd) * PIXEL_TO_MM, 2),
        "direction": direction,
        "lineA": (posterior, anterior),
        "pointB": posterior_l5,
    }


def _compute_sd(rows: np.ndarray) -> list[dict[str, Any]]:
    vertebrae = [_parse_vertebra(row) for row in rows]
    lr_info = _infer_lr_from_keypoints(rows)
    lr_label = str(lr_info["lr_label"])
    names = ["L1-L2", "L2-L3", "L3-L4", "L4-L5"]
    results = []
    for index in range(4):
        item = _compute_sd_between(vertebrae[index], vertebrae[index + 1], lr_label=lr_label)
        results.append({"name": f"SD({names[index]})", "lr_label": lr_label, "lr_method": lr_info["method"], **item})
    item = _compute_sd_l5_s1(vertebrae[4], vertebrae[5], lr_label=lr_label)
    results.append({"name": "SD(L5-S1)", "lr_label": lr_label, "lr_method": lr_info["method"], **item})
    return results


def _meyerding_single(upper: np.ndarray, lower: np.ndarray, lr_label: str = "L") -> dict[str, Any] | None:
    lower_tl = np.array([lower[2], lower[3]], dtype=float)
    lower_tr = np.array([lower[4], lower[5]], dtype=float)
    lower_post = np.array(_posterior_point(tuple(lower_tl), tuple(lower_tr), lr_label=lr_label), dtype=float)
    lower_ant = np.array(_anterior_point(tuple(lower_tl), tuple(lower_tr), lr_label=lr_label), dtype=float)
    base_vec = lower_ant - lower_post
    base_len = np.linalg.norm(base_vec)
    if base_len < 1e-6:
        return None
    base_unit = base_vec / base_len
    left_lower = np.array([upper[6], upper[7]], dtype=float)
    right_lower = np.array([upper[8], upper[9]], dtype=float)
    point = np.array(_posterior_point(tuple(left_lower), tuple(right_lower), lr_label=lr_label), dtype=float)
    ratio = np.dot(point - lower_post, base_unit) / base_len
    foot = lower_post + ratio * base_vec
    abs_ratio = abs(float(ratio))
    if abs_ratio <= 0.25:
        grade = "I"
    elif abs_ratio <= 0.50:
        grade = "II"
    elif abs_ratio <= 0.75:
        grade = "III"
    elif abs_ratio <= 1.0:
        grade = "IV"
    else:
        grade = "V"
    return {
        "ratio": round(abs_ratio, 4),
        "direction": "anterior" if ratio >= 0 else "posterior",
        "grade": grade,
        "lower_endplate": [lower_post.tolist(), lower_ant.tolist()],
        "upper_point": point.tolist(),
        "foot_point": foot.tolist(),
    }


def _compute_meyerding(rows: np.ndarray) -> list[dict[str, Any]]:
    lr_info = _infer_lr_from_keypoints(rows)
    lr_label = str(lr_info["lr_label"])
    results = []
    for index in range(len(rows) - 1):
        grade_info = _meyerding_single(rows[index], rows[index + 1], lr_label=lr_label)
        if grade_info is None:
            continue
        results.append(
            {
                "level_index": index,
                "upper_center": rows[index][:2].tolist(),
                "lower_center": rows[index + 1][:2].tolist(),
                "lr_label": lr_label,
                "lr_method": lr_info["method"],
                **grade_info,
            }
        )
    return results


def calculate_slippage_metrics(keypoints: list[list[float]]) -> dict[str, Any]:
    rows = np.asarray(keypoints, dtype=float)
    if rows.shape[0] < 6:
        return {"meyerding": [], "isa": [], "sd": []}
    return {
        "meyerding": _compute_meyerding(rows),
        "isa": _compute_isa(rows),
        "sd": _compute_sd(rows),
    }
