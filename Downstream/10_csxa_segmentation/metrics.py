"""SpineFM-compatible instance localization and Dice, with matching audit."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from common import LEVELS


def dice_matrix(predictions, targets):
    targets = np.asarray(targets, dtype=bool)
    if targets.ndim != 3 or targets.shape[0] != 5:
        raise ValueError("Expected five GT masks in C3-C7 order")
    if len(predictions) == 0:
        return np.zeros((5, 0), dtype=np.float64)
    predictions = np.asarray(predictions, dtype=bool)
    if predictions.shape[1:] != targets.shape[1:]:
        raise ValueError("Prediction masks must be restored to the original image size")
    a, b = targets.reshape(5, -1), predictions.reshape(len(predictions), -1)
    intersection = np.asarray([[np.count_nonzero(x & y) for y in b] for x in a])
    denominator = a.sum(1)[:, None] + b.sum(1)[None, :]
    return np.divide(2*intersection, denominator, out=np.zeros_like(intersection, dtype=float), where=denominator > 0)


def match_scores(matrix, mode="reference", threshold=0.4):
    """Reference preserves the notebook's LAST passing prediction, including reuse.

    one_to_one maximizes passing match count first, then total Dice.
    """
    scores = np.zeros(matrix.shape[0], dtype=float)
    assignment = np.full(matrix.shape[0], -1, dtype=int)
    if mode == "reference":
        for j, row in enumerate(matrix):
            valid = np.flatnonzero(row > threshold)
            if len(valid):
                assignment[j] = valid[-1]
                scores[j] = row[valid[-1]]
    elif mode == "one_to_one":
        if matrix.shape[1]:
            valid = matrix > threshold
            reward = valid * (matrix.shape[0] + 1 + matrix)
            rows, cols = linear_sum_assignment(-reward)
            for j, k in zip(rows, cols):
                if valid[j, k]:
                    assignment[j], scores[j] = k, matrix[j, k]
    else:
        raise ValueError(mode)
    return scores, assignment


class SpineFMEvaluator:
    def __init__(self, mode="reference", located_threshold=0.4):
        self.mode, self.threshold = mode, located_threshold
        self.records = {}

    def add(self, identifier, predictions, targets):
        if identifier in self.records:
            raise ValueError(f"Duplicate evaluation image: {identifier}")
        matrix = dice_matrix(predictions, targets)
        scores, assignment = match_scores(matrix, self.mode, self.threshold)
        self.records[identifier] = dict(scores=scores.tolist(), assignment=assignment.tolist(),
                                        predictions=matrix.shape[1],
                                        multiple_candidates=(matrix > self.threshold).sum(1).tolist(),
                                        dice_matrix=matrix.tolist())

    def summary(self, expected_ids=None):
        if expected_ids is not None and set(expected_ids) != set(self.records):
            raise ValueError("Every split image must be evaluated, including zero-output failures")
        if not self.records:
            raise ValueError("Empty evaluation")
        values = np.asarray([r["scores"] for r in self.records.values()])

        def row(v):
            found = v > self.threshold
            return dict(identified_percent=float(found.mean()*100),
                        located_dsc=float(v[found].mean()) if found.any() else None,
                        overall_dsc=float(v.mean()), located=int(found.sum()), total=int(v.size))

        result = {level: row(values[:, j]) for j, level in enumerate(LEVELS)}
        result["Avg"] = row(values.reshape(-1))
        return dict(matching=self.mode, located_dice_threshold=self.threshold, images=len(values), table=result)


def table_text(result):
    columns = [*LEVELS, "Avg"]
    lines = ["| Metric | " + " | ".join(columns) + " |", "|---|" + "---|"*len(columns)]
    for title, key in [("% identified", "identified_percent"), ("Located DSC", "located_dsc"), ("Overall DSC", "overall_dsc")]:
        vals = [result["table"][c][key] for c in columns]
        lines.append("| " + title + " | " + " | ".join("N/A" if x is None else (f"{x:.2f}%" if key == "identified_percent" else f"{x:.3f}") for x in vals) + " |")
    return "\n".join(lines) + "\n"
