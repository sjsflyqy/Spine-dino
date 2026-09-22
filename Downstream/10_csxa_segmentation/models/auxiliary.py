"""Local or official SpineFM auxiliaries; explicit format and provenance."""
import hashlib
import torch
from torch import nn
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.models import resnet50
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from common import resolve, sha256


def split_fingerprints(splits):
    return {name: hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
            for name, ids in splits.items()}


def freeze_batch_norm_layers(module):
    """Match the pretrained Torchvision Mask R-CNN used by official SpineFM."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, FrozenBatchNorm2d(child.num_features, eps=child.eps))
        else:
            freeze_batch_norm_layers(child)


class PointPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def build_auxiliary(stage, config=None):
    config = config or {}
    initialization = config.get("initialization")
    if stage == "detector":
        model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None,
                                      min_size=int(config.get("min_size", 640)),
                                      max_size=int(config.get("max_size", 1024)))
        if initialization:
            model.load_state_dict(torch.load(resolve(initialization), map_location="cpu", weights_only=True), strict=True)
        model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)
        model.roi_heads.mask_predictor = MaskRCNNPredictor(model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, 2)
        if config.get("frozen_batch_norm", False):
            freeze_batch_norm_layers(model.backbone)
    elif stage == "classifier":
        model = resnet50(weights=None)
        if initialization:
            model.load_state_dict(torch.load(resolve(initialization), map_location="cpu", weights_only=True), strict=True)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif stage == "point_predictor":
        model = PointPredictor()
    else:
        raise ValueError(stage)
    return model


def load_auxiliaries(config, store, device):
    models, fingerprints = {}, {}
    source = config.get("source", "local")
    if source not in {"local", "spinefm_official"}:
        raise ValueError(f"Unknown auxiliary source: {source}")
    if source == "spinefm_official":
        expected = config.get("reference_split_hashes")
        if expected != split_fingerprints(store.splits):
            raise ValueError("Official auxiliary configuration must use the published SpineFM split IDs")
    for stage in ("detector", "classifier", "point_predictor"):
        path = resolve(config[stage])
        if not path.is_file():
            hint = "download the official file" if source == "spinefm_official" else "run train_auxiliary.py first"
            raise FileNotFoundError(f"Missing {stage} checkpoint {path}; {hint}")
        if source == "spinefm_official":
            payload = torch.load(path, map_location="cpu", weights_only=True)
            state = payload.get("state_dict", payload)
            if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
                raise ValueError(f"Expected an official tensor state dictionary: {path}")
            model_config = (dict(min_size=800, max_size=1333, frozen_batch_norm=True)
                            if stage == "detector" else {})
            model = build_auxiliary(stage, model_config)
            model.load_state_dict(state, strict=True)
            if stage == "classifier":
                model.csxa_preprocessing = "spinefm_official"
            if stage == "detector":
                model.csxa_mask_threshold = 0.9
            models[stage] = model.to(device).eval().requires_grad_(False)
            fingerprints[stage] = dict(path=str(path), sha256=sha256(path), source=source,
                                       model_config=model_config, preprocessing="spinefm_reference",
                                       reference_split_hashes=expected,
                                       training_membership="author-reported; not embedded in raw weights")
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("stage") != stage or payload.get("data_fingerprint") != store.fingerprint:
            raise ValueError(f"Auxiliary checkpoint stage/GT provenance mismatch: {path}")
        if payload.get("fit_split") != "train" or payload.get("selection_split") != "val":
            raise ValueError(f"Auxiliary checkpoint has invalid split provenance: {path}")
        stage_config = dict(payload.get("model_config", {}))
        stage_config.pop("initialization", None)
        model = build_auxiliary(stage, stage_config)
        model.load_state_dict(payload["state_dict"], strict=True)
        models[stage] = model.to(device).eval().requires_grad_(False)
        fingerprints[stage] = {"path": str(path), "sha256": sha256(path)}
    return models, fingerprints
