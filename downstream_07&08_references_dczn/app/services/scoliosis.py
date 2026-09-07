from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import nms as tv_nms

from .. import config
from ..algorithms import calculate_ap_cobb, calculate_lat_alignment
from ..schemas import Detection, PipelineResponse, ViewPrediction
from .model_registry import ModelRegistry


class ScoliosisPipeline:
    def __init__(self, registry: ModelRegistry) -> None:
        # 侧弯流水线只负责三件事：
        # 检测椎体、推理关键点、根据关键点计算侧弯相关指标。
        self.registry = registry
        self.device = registry.device

    def predict_keypoints(self, images: dict[str, np.ndarray], bbox_expand_px: int = 0) -> PipelineResponse:
        # 标准流程第一步：上传影像后先产出关键点，不直接做最终分析。
        views = {view: self._predict_view(view, image, include_metrics=False, bbox_expand_px=bbox_expand_px) for view, image in images.items() if image is not None}
        return PipelineResponse(task="scoliosis", device=self.device, views=views)

    def analyze(self, images: dict[str, np.ndarray], bbox_expand_px: int = 0) -> PipelineResponse:
        # 这个接口保留给“上传影像后一步直出”的快捷测试。
        views = {view: self._predict_view(view, image, include_metrics=True, bbox_expand_px=bbox_expand_px) for view, image in images.items() if image is not None}
        return PipelineResponse(task="scoliosis", device=self.device, views=views)

    def analyze_keypoints(self, views_payload: dict[str, dict[str, Any]]) -> PipelineResponse:
        # 标准流程第三步：使用前端确认后的关键点重新做分析。
        views = {}
        for view, payload in views_payload.items():
            keypoints = self._coerce_scoliosis_keypoints(payload.get("keypoints", []))
            detections = self._coerce_detections(payload.get("detections", []))
            metrics = self._build_metrics(view, keypoints)
            views[view] = ViewPrediction(view=view, detections=detections, keypoints=keypoints, metrics=metrics)
        return PipelineResponse(task="scoliosis", device=self.device, views=views)

    def _predict_view(self, view: str, image: np.ndarray, include_metrics: bool, bbox_expand_px: int = 0) -> ViewPrediction:
        # 单个视图的处理顺序固定为：检测框 -> 关键点 -> 指标。
        detections = self._detect_vertebrae(view, image, bbox_expand_px=bbox_expand_px)
        boxes = [item.box for item in detections]
        keypoints = self._predict_keypoints_for_boxes(view, image, boxes)
        metrics = self._build_metrics(view, keypoints) if include_metrics else {}
        return ViewPrediction(view=view, detections=detections, keypoints=keypoints, metrics=metrics)

    def _build_metrics(self, view: str, keypoints: list[list[list[float]]]) -> dict[str, Any]:
        # AP 视图负责 Cobb，LAT 视图负责矢状面相关指标。
        metrics: dict[str, Any] = {}
        if not keypoints:
            return metrics
        if view == "ap":
            metrics["cobb"] = calculate_ap_cobb(keypoints)
        if view == "lat":
            metrics["alignment"] = calculate_lat_alignment(keypoints)
        return metrics

    def _coerce_scoliosis_keypoints(self, keypoints: list[Any]) -> list[list[list[float]]]:
        # 前端允许用户拖拽关键点，这里负责把 JSON 中的动态数据
        # 转回算法层需要的 [[x, y], ...] 浮点坐标结构。
        normalized = []
        for group in keypoints:
            if not isinstance(group, list):
                continue
            normalized_group = []
            for point in group:
                if isinstance(point, list) and len(point) >= 2:
                    normalized_group.append([float(point[0]), float(point[1])])
            if normalized_group:
                normalized.append(normalized_group)
        return normalized

    def _coerce_detections(self, detections: list[dict[str, Any]]) -> list[Detection]:
        # 分析接口允许前端把检测框一起传回来，
        # 这样后端响应里可以保留完整上下文，前端也方便继续复现画面。
        normalized = []
        for item in detections:
            if not isinstance(item, dict):
                continue
            box = [float(value) for value in item.get("box", [])[:4]]
            if len(box) != 4:
                continue
            normalized.append(
                Detection(
                    label=str(item.get("label", "")),
                    score=float(item.get("score", 0.0)),
                    box=box,
                )
            )
        return normalized

    def _detect_vertebrae(self, view: str, image: np.ndarray, bbox_expand_px: int = 0) -> list[Detection]:
        model = self.registry.get_scoliosis_yolo(view)
        results = model.predict(image, device=self.device, verbose=False)
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or boxes.xyxy is None:
            return []
        xyxy_tensor = boxes.xyxy.detach()
        scores_tensor = boxes.conf.detach() if boxes.conf is not None else torch.ones(len(xyxy_tensor), device=xyxy_tensor.device)
        if len(xyxy_tensor) > 0:
            keep_indices = tv_nms(xyxy_tensor.float(), scores_tensor.float(), iou_threshold=0.5)
            xyxy_tensor = xyxy_tensor[keep_indices]
            scores_tensor = scores_tensor[keep_indices]
        xyxy = xyxy_tensor.cpu().numpy()
        scores = scores_tensor.cpu().numpy() if len(xyxy) > 0 else np.empty((0,), dtype=float)
        # 检测结果按垂直方向排序，再依次映射成 T1 ~ L5 标签。
        order = np.argsort(((xyxy[:, 1] + xyxy[:, 3]) / 2.0))
        detections = []
        expanded_xyxy = self._expand_boxes(xyxy, image.shape, bbox_expand_px)
        for index in order[: len(config.VERTEBRA_LABELS)]:
            label = config.VERTEBRA_LABELS[len(detections)]
            detections.append(Detection(label=label, score=float(scores[index]), box=expanded_xyxy[index].astype(float).tolist()))
        return detections

    def _expand_boxes(self, boxes: np.ndarray, image_shape: tuple[int, int, int], bbox_expand_px: int = 0) -> np.ndarray:
        bbox_expand_px = max(0, int(bbox_expand_px or 0))
        if len(boxes) == 0:
            return boxes.copy()
        height, width = image_shape[:2]
        expanded_boxes = []
        for box in boxes:
            xmin, ymin, xmax, ymax = map(int, box[:4])
            xmin -= bbox_expand_px
            ymin -= bbox_expand_px
            xmax += bbox_expand_px
            ymax += bbox_expand_px
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = max(xmin + 1, min(width, xmax))
            ymax = max(ymin + 1, min(height, ymax))
            expanded_boxes.append([xmin, ymin, xmax, ymax])
        return np.asarray(expanded_boxes, dtype=np.float32)

    def _predict_keypoints_for_boxes(self, view: str, image: np.ndarray, boxes: list[list[float]]) -> list[list[list[float]]]:
        if not boxes:
            return []
        # 关键点网络输入必须是对齐后的 patch，所以先切框预处理。
        patches, normalized_boxes = self._preprocess_keypoints(image, boxes)
        if patches is None:
            return []
        model = self.registry.get_scoliosis_keypoint(view)
        with torch.no_grad():
            predictions = model(patches.to(self.device))
        return self._postprocess_keypoints(predictions, normalized_boxes)

    def _preprocess_keypoints(self, image: np.ndarray, boxes: list[list[float]]) -> tuple[torch.Tensor | None, np.ndarray]:
        # 每个检测框都裁成固定大小的 patch，
        # 同时保留原图坐标，供后处理时把关键点映射回原图。
        patches = []
        normalized_boxes = []
        height, width = image.shape[:2]
        dst_wh = (128, 100)
        for box in boxes:
            xmin, ymin, xmax, ymax = map(int, box[:4])
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(width, xmax)
            ymax = min(height, ymax)
            if xmax <= xmin or ymax <= ymin:
                continue
            patch = image[ymin:ymax, xmin:xmax]
            # 旧模型训练时就是按 128x100 + [-0.5, 0.5] 归一化输入，这里保持一致。
            patch = cv2.resize(patch, dst_wh).astype(np.float32) / 255.0
            patch = np.transpose(patch - 0.5, (2, 0, 1))
            patches.append(torch.tensor(patch, dtype=torch.float32).unsqueeze(0))
            normalized_boxes.append([xmin, ymin, xmax, ymax])
        if not patches:
            return None, np.empty((0, 4), dtype=np.float32)
        return torch.cat(patches, dim=0), np.asarray(normalized_boxes, dtype=np.float32)

    def _postprocess_keypoints(self, predictions: dict[str, torch.Tensor], boxes: np.ndarray, dst_wh: tuple[int, int] = (128, 100), down_ratio: int = 4, top_k: int = 1) -> list[list[list[float]]]:
        # 这部分是关键点后处理核心：
        # 从 hm/reg/wh 三个头中解码出 8 个关键点，再映射回原图。
        def gather_feat(feat: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
            dim = feat.size(2)
            ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
            return feat.gather(1, ind)

        def transpose_and_gather_feat(feat: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
            feat = feat.permute(0, 2, 3, 1).contiguous()
            feat = feat.view(feat.size(0), -1, feat.size(3))
            return gather_feat(feat, ind)

        def topk(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            batch, cat, height, width = scores.size()
            topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), k)
            topk_inds = topk_inds % (height * width)
            topk_ys = (topk_inds // width).int().float()
            topk_xs = (topk_inds % width).int().float()
            topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), k)
            topk_inds = gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, k)
            topk_ys = gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, k)
            topk_xs = gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, k)
            return topk_score, topk_inds, topk_ys, topk_xs

        all_keypoints: list[list[list[float]]] = []
        for index in range(predictions["wh"].shape[0]):
            wh = predictions["wh"][index : index + 1]
            reg = predictions["reg"][index : index + 1]
            hm = predictions["hm"][index : index + 1]
            xmin, ymin, xmax, ymax = boxes[index]
            hm = F.max_pool2d(hm.detach(), (3, 3), stride=1, padding=1).eq(hm.detach()).float() * hm.detach()
            _, inds, ys, xs = topk(hm[:, 0:1], top_k)
            reg_values = transpose_and_gather_feat(reg[:, 0:2], inds).view(1, top_k, 2)
            xs = xs.view(1, top_k, 1) + reg_values[:, :, 0:1]
            ys = ys.view(1, top_k, 1) + reg_values[:, :, 1:2]
            wh_values = transpose_and_gather_feat(wh[:, :16], inds).view(1, top_k, 16)
            coords = [xs, ys]
            for point_index in range(8):
                coords.append(xs - wh_values[:, :, 2 * point_index : 2 * point_index + 1])
                coords.append(ys - wh_values[:, :, 2 * point_index + 1 : 2 * (point_index + 1)])
            points = torch.cat(coords, dim=2).squeeze(0).detach().cpu().numpy().reshape(top_k, 9, 2)
            width = int(xmax) - int(xmin)
            height = int(ymax) - int(ymin)
            # 输出坐标最初位于下采样后的 patch 坐标系，需要恢复到原图坐标系。
            points *= down_ratio
            points[:, :, 0] = points[:, :, 0] / dst_wh[0] * width + int(xmin)
            points[:, :, 1] = points[:, :, 1] / dst_wh[1] * height + int(ymin)
            all_keypoints.append(points[0, 1:, :].tolist())
        return all_keypoints
