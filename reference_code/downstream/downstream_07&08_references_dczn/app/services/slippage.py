from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .. import config
from ..algorithms import calculate_slippage_metrics
from ..schemas import Detection, PipelineResponse, ViewPrediction
from .model_registry import ModelRegistry


class SlippagePipeline:
    def __init__(self, registry: ModelRegistry) -> None:
        # 滑脱流水线和侧弯保持同一设计：
        # 先预测关键点，再基于“最终确认的关键点”做指标分析。
        self.registry = registry
        self.device = registry.device

    def predict_keypoints(self, images: dict[str, np.ndarray]) -> PipelineResponse:
        # 标准流程第一步：上传影像后返回腰椎框与关键点。
        views = {view: self._predict_view(view, image, include_metrics=False) for view, image in images.items() if image is not None}
        return PipelineResponse(task="slippage", device=self.device, views=views)

    def analyze(self, images: dict[str, np.ndarray]) -> PipelineResponse:
        # 兼容保留的一步直出接口，用于快捷测试影像到指标的完整链路。
        views = {view: self._predict_view(view, image, include_metrics=True) for view, image in images.items() if image is not None}
        if "gs" in views and "gq" in views:
            comparison = self._build_motion_delta(views["gs"].metrics, views["gq"].metrics)
            if comparison:
                views["gs"].metrics["delta_vs_gq"] = comparison
                views["gq"].metrics["delta_vs_gs"] = comparison
        return PipelineResponse(task="slippage", device=self.device, views=views)

    def analyze_keypoints(self, views_payload: dict[str, dict[str, Any]]) -> PipelineResponse:
        # 标准流程第三步：接收前端调整后的关键点，重新输出分析指标。
        views = {}
        for view, payload in views_payload.items():
            detections = self._coerce_detections(payload.get("detections", []))
            keypoints = self._coerce_slippage_keypoints(payload.get("keypoints", []))
            metrics = self._build_metrics(view, keypoints)
            views[view] = ViewPrediction(view=view, detections=detections, keypoints=keypoints, metrics=metrics)
        if "gs" in views and "gq" in views:
            comparison = self._build_motion_delta(views["gs"].metrics, views["gq"].metrics)
            if comparison:
                views["gs"].metrics["delta_vs_gq"] = comparison
                views["gq"].metrics["delta_vs_gs"] = comparison
        return PipelineResponse(task="slippage", device=self.device, views=views)

    def _predict_view(self, view: str, image: np.ndarray, include_metrics: bool) -> ViewPrediction:
        # 每个视图都共享同样的顺序：腰椎区域检测 -> 关键点预测 -> 指标计算。
        detections = self._detect_lumbar(image)
        keypoints = self._predict_keypoints_for_box(image, detections[0].box) if detections else []
        metrics = self._build_metrics(view, keypoints) if include_metrics else {}
        return ViewPrediction(view=view, detections=detections, keypoints=keypoints, metrics=metrics)

    def _build_metrics(self, view: str, keypoints: list[list[float]]) -> dict[str, Any]:
        # LAT 负责 Meyerding，GS/GQ 负责 ISA 和 SD，差值在最后统一汇总。
        if not keypoints:
            return {}
        all_metrics = calculate_slippage_metrics(keypoints)
        if view == "lat":
            return {"meyerding": all_metrics["meyerding"]}
        return {"isa": all_metrics["isa"], "sd": all_metrics["sd"]}

    def _coerce_detections(self, detections: list[dict[str, Any]]) -> list[Detection]:
        # 把前端 JSON 中的检测框恢复成统一 Detection 对象。
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

    def _coerce_slippage_keypoints(self, keypoints: list[Any]) -> list[list[float]]:
        # 滑脱关键点以一维数组形式存储：
        # [cx, cy, tlx, tly, trx, try, blx, bly, brx, bry, score]
        normalized = []
        for group in keypoints:
            if not isinstance(group, list):
                continue
            row = [float(value) for value in group]
            if len(row) >= 10:
                normalized.append(row)
        return normalized

    def _detect_lumbar(self, image: np.ndarray) -> list[Detection]:
        model = self.registry.get_slippage_yolo()
        results = model.predict(image, device=self.device, verbose=False)
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
            return []
        xyxy = boxes.xyxy.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy), dtype=float)
        # 滑脱只需要一个最佳腰椎区域，因此只保留最高置信度框。
        best_index = int(np.argmax(scores))
        return [Detection(label="lumbar", score=float(scores[best_index]), box=xyxy[best_index].astype(float).tolist())]

    def _predict_keypoints_for_box(self, image: np.ndarray, box: list[float]) -> list[list[float]]:
        # 腰椎区域先裁剪，再送入关键点网络，最后映射回原图坐标。
        model = self.registry.get_slippage_keypoint()
        cropped, top, left = self._crop_image(image, box)
        crop_h, crop_w = cropped.shape[:2]
        tensor = self._prepare_slippage_input(cropped)
        with torch.no_grad():
            predictions = model(tensor)
        decoded = self._decode_slippage_predictions(predictions["hm"], predictions["wh"], predictions["reg"])
        decoded[:, :10] *= 4
        x_indices = range(0, 10, 2)
        y_indices = range(1, 10, 2)
        decoded[:, x_indices] = decoded[:, x_indices] / 512 * crop_w + left
        decoded[:, y_indices] = decoded[:, y_indices] / 1024 * crop_h + top
        decoded = decoded[np.argsort(decoded[:, 1])]
        return decoded.tolist()

    def _crop_image(self, image: np.ndarray, box: list[float], expand_ratio: float = 0.1) -> tuple[np.ndarray, int, int]:
        # 对检测框做适度扩边，避免关键点靠近边缘时被裁掉。
        height, width = image.shape[:2]
        expand_pixels = int(max(height, width) * expand_ratio)
        x1, y1, x2, y2 = box
        xmin = max(int(x1 - expand_pixels), 0)
        xmax = min(int(x2 + expand_pixels), width)
        ymin = max(int(y1 - expand_pixels / 2), 0)
        ymax = min(int(y2 + expand_pixels), height)
        return image[ymin:ymax, xmin:xmax], ymin, xmin

    def _prepare_slippage_input(self, image: np.ndarray) -> torch.Tensor:
        # 保持与旧模型训练时一致的输入规格：512x1024、三通道、[-0.5, 0.5]。
        resized = cv2.resize(image, (512, 1024)).astype(np.float32) / 255.0
        normalized = resized - 0.5
        chw = normalized.transpose(2, 0, 1).reshape(1, 3, 1024, 512)
        return torch.from_numpy(chw).to(self.device)

    def _decode_slippage_predictions(self, hm: torch.Tensor, wh: torch.Tensor, reg: torch.Tensor, top_k: int = 6) -> np.ndarray:
        # 这里把网络输出的热力图和偏移量解码成 6 个椎体的四角关键点。
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
            topk_ys = (topk_inds / width).int().float()
            topk_xs = (topk_inds % width).int().float()
            topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), k)
            topk_inds = gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, k)
            topk_ys = gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, k)
            topk_xs = gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, k)
            return topk_score, topk_inds, topk_ys, topk_xs

        heat = F.max_pool2d(hm, (3, 3), stride=1, padding=1).eq(hm).float() * hm
        scores, inds, ys, xs = topk(heat, top_k)
        scores = scores.view(1, top_k, 1)
        reg_values = transpose_and_gather_feat(reg, inds).view(1, top_k, 2)
        xs = xs.view(1, top_k, 1) + reg_values[:, :, 0:1]
        ys = ys.view(1, top_k, 1) + reg_values[:, :, 1:2]
        wh_values = transpose_and_gather_feat(wh, inds).view(1, top_k, 8)
        tl_x = xs - wh_values[:, :, 0:1]
        tl_y = ys - wh_values[:, :, 1:2]
        tr_x = xs - wh_values[:, :, 2:3]
        tr_y = ys - wh_values[:, :, 3:4]
        bl_x = xs - wh_values[:, :, 4:5]
        bl_y = ys - wh_values[:, :, 5:6]
        br_x = xs - wh_values[:, :, 6:7]
        br_y = ys - wh_values[:, :, 7:8]
        points = torch.cat([xs, ys, tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y, scores], dim=2).squeeze(0)
        return points.detach().cpu().numpy()

    def _build_motion_delta(self, gs_metrics: dict[str, Any], gq_metrics: dict[str, Any]) -> list[dict[str, float]]:
        # GS 和 GQ 的差值是滑脱任务特有的动态信息，用于描述过伸/过屈变化幅度。
        gs_isa = gs_metrics.get("isa") or []
        gq_isa = gq_metrics.get("isa") or []
        gs_sd = gs_metrics.get("sd") or []
        gq_sd = gq_metrics.get("sd") or []
        count = min(len(gs_isa), len(gq_isa), len(gs_sd), len(gq_sd))
        deltas = []
        for index in range(count):
            deltas.append(
                {
                    "name": gs_isa[index]["name"],
                    "isa_delta": round(abs(gs_isa[index]["angle"] - gq_isa[index]["angle"]), 2),
                    "sd_delta": round(abs(gs_sd[index]["sd"] - gq_sd[index]["sd"]), 2),
                }
            )
        return deltas
