from __future__ import annotations

import importlib
import threading
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

from .. import config


class ModelRegistry:
    def __init__(self, device: str | None = None) -> None:
        # 所有模型都通过这个注册表统一缓存，避免每次请求都重复加载权重。
        self.device = device or config.DEVICE
        self._lock = threading.Lock()
        self._models: dict[str, Any] = {}

    def warmup(self) -> None:
        # 预热时主动把四类模型全部放进缓存，
        # 这样首个请求不会把大部分时间耗在模型加载上。
        for view in config.SCOLIOSIS_VIEWS:
            self.get_scoliosis_yolo(view)
            self.get_scoliosis_keypoint(view)
        self.get_slippage_yolo()
        self.get_slippage_keypoint()

    def get_scoliosis_yolo(self, view: str) -> YOLO:
        model_key = f"scoliosis_yolo:{view}"
        return self._get_or_create(model_key, lambda: YOLO(str(config.SCOLIOSIS_YOLO_MODELS[view].weights)))

    def get_slippage_yolo(self) -> YOLO:
        return self._get_or_create("slippage_yolo", lambda: YOLO(str(config.SLIPPAGE_YOLO_MODEL.weights)))

    def get_scoliosis_keypoint(self, view: str) -> torch.nn.Module:
        model_key = f"scoliosis_keypoint:{view}"
        cfg = config.SCOLIOSIS_KEYPOINT_MODELS[view]
        return self._get_or_create(model_key, lambda: self._load_scoliosis_keypoint(cfg.weights))

    def get_slippage_keypoint(self) -> torch.nn.Module:
        return self._get_or_create("slippage_keypoint", lambda: self._load_slippage_keypoint(config.SLIPPAGE_KEYPOINT_MODEL.weights))

    def _get_or_create(self, key: str, factory) -> Any:
        # 双重检查加锁：
        # 多线程请求同时到达时，只允许一个线程真正创建模型实例。
        if key in self._models:
            return self._models[key]
        with self._lock:
            if key not in self._models:
                self._models[key] = factory()
        return self._models[key]

    def _load_scoliosis_keypoint(self, weight_path: Path) -> torch.nn.Module:
        # 侧弯关键点模型结构已经变成统一工程自己的原生包导入。
        module = importlib.import_module("assets.scoliosis.model_architecture.scoliosis_spinenet")
        model = module.SpineNet()
        # 部分旧 checkpoint 外层包了一层 state_dict，统一在这里兼容处理。
        checkpoint = torch.load(weight_path, map_location=self.device)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        return model

    def _load_slippage_keypoint(self, weight_path: Path) -> torch.nn.Module:
        # 滑脱关键点模型结构同样改为统一工程内的原生包导入。
        module = importlib.import_module("assets.slippage.model_architecture.slippage_spinenet")
        model = module.SpineNet()
        checkpoint = torch.load(weight_path, map_location=self.device)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        return model
