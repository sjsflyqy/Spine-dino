from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Detection:
    # 单个检测框的标准结构：
    # label 表示语义标签，score 表示置信度，box 统一使用 [x1, y1, x2, y2]。
    label: str
    score: float
    box: list[float]


@dataclass
class ViewPrediction:
    # 每个视图都会输出一份独立结果。
    # detections 是检测框，keypoints 是关键点，metrics 是最终分析指标。
    view: str
    detections: list[Detection] = field(default_factory=list)
    keypoints: list[Any] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        # 统一把 dataclass 递归展开成普通 dict，
        # 这样 Flask 在 jsonify 时不需要再关心对象类型。
        return asdict(self)


@dataclass
class PipelineResponse:
    # 统一任务响应最外层结构：
    # task 标识任务类型，device 标识推理设备，views 存放各个视图的结果。
    task: str
    device: str
    views: dict[str, ViewPrediction]

    def to_dict(self) -> dict[str, Any]:
        # 这里手动展开 views，而不是直接 asdict(self)，
        # 是为了保证每个视图都走 ViewPrediction.to_dict() 的同一出口。
        return {
            "task": self.task,
            "device": self.device,
            "views": {key: value.to_dict() for key, value in self.views.items()},
        }
