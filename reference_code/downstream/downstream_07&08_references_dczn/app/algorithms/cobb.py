from __future__ import annotations

import math
from typing import Any

import numpy as np


def _edge_angle(left: np.ndarray, right: np.ndarray) -> float:
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    return math.atan2(dy, dx)


def _compose_result(angle: float, indices: list[int], upper_left: np.ndarray, upper_right: np.ndarray, lower_left: np.ndarray, lower_right: np.ndarray) -> dict[str, Any]:
    return {
        "angle": round(float(angle), 2),
        "indices": indices,
        "endpoints": {
            "start_left": upper_left.tolist(),
            "start_right": upper_right.tolist(),
            "end_left": lower_left.tolist(),
            "end_right": lower_right.tolist(),
        },
    }


def _angle_between(upper_left: np.ndarray, upper_right: np.ndarray, lower_left: np.ndarray, lower_right: np.ndarray) -> float:
    upper_angle = _edge_angle(upper_left, upper_right)
    lower_angle = _edge_angle(lower_left, lower_right)
    cobb_angle = abs(upper_angle - lower_angle) * 180 / math.pi
    return 180 - cobb_angle if cobb_angle > 90 else cobb_angle


def calculate_ap_cobb(keypoints: list[list[list[float]]]) -> list[dict[str, Any]]:
    points = np.asarray(keypoints, dtype=float)
    count = len(points)
    if count < 2:
        return []
    angle_matrix = np.zeros((count, count), dtype=float)
    endpoint_matrix: list[list[dict[str, Any] | None]] = [[None] * count for _ in range(count)]
    for upper_index in range(count):
        for lower_index in range(upper_index + 1, count):
            upper_left = points[upper_index][0]
            upper_right = points[upper_index][1]
            lower_left = points[lower_index][2]
            lower_right = points[lower_index][3]
            angle_matrix[upper_index][lower_index] = _angle_between(upper_left, upper_right, lower_left, lower_right)
            endpoint_matrix[upper_index][lower_index] = _compose_result(
                angle_matrix[upper_index][lower_index],
                [upper_index, lower_index],
                upper_left,
                upper_right,
                lower_left,
                lower_right,
            )
    results: list[dict[str, Any]] = []
    used_ranges: list[tuple[int, int]] = []
    for _ in range(min(3, count - 1)):
        best_angle = 0.0
        best_pair = None
        for upper_index in range(count):
            for lower_index in range(upper_index + 1, count):
                overlaps = any(not (lower_index < start or upper_index > end) for start, end in used_ranges)
                if overlaps:
                    continue
                if angle_matrix[upper_index][lower_index] > best_angle:
                    best_angle = angle_matrix[upper_index][lower_index]
                    best_pair = (upper_index, lower_index)
        if best_pair is None:
            break
        start, end = best_pair
        used_ranges.append(best_pair)
        result = endpoint_matrix[start][end]
        if result is not None:
            results.append(result)
    return results


def calculate_lat_alignment(keypoints: list[list[list[float]]]) -> list[dict[str, Any]]:
    points = np.asarray(keypoints, dtype=float)
    count = len(points)
    results: list[dict[str, Any]] = []
    if count >= 12:
        tk = _compose_result(
            _angle_between(points[0][0], points[0][1], points[11][2], points[11][3]),
            [0, 11],
            points[0][0],
            points[0][1],
            points[11][2],
            points[11][3],
        )
        tk["name"] = "TK"
        results.append(tk)
    if count >= 14:
        lower_index = min(16, count - 1)
        ll = _compose_result(
            _angle_between(points[12][0], points[12][1], points[lower_index][2], points[lower_index][3]),
            [12, lower_index],
            points[12][0],
            points[12][1],
            points[lower_index][2],
            points[lower_index][3],
        )
        ll["name"] = "LL"
        results.append(ll)
    return results
