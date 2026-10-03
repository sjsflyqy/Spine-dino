"""Exact step budgets without changing the upstream DINOv2 scheduler."""

from numbers import Integral


def get_schedule_iterations(cfg):
    explicit = cfg.optim.get("total_iterations", 0)
    if isinstance(explicit, bool) or not isinstance(explicit, Integral) or explicit < 0:
        raise ValueError("optim.total_iterations must be a non-negative integer")
    total = explicit or cfg.optim.epochs * cfg.train.OFFICIAL_EPOCH_LENGTH
    if isinstance(total, bool) or not isinstance(total, Integral) or total <= 0:
        raise ValueError("The scheduler must have a positive integer step budget")
    return int(total)


def build_schedulers(cfg):
    from omegaconf import OmegaConf
    from dinov2.train.train import build_schedulers as upstream_build_schedulers

    total = get_schedule_iterations(cfg)
    if not cfg.optim.get("total_iterations", 0):
        return upstream_build_schedulers(cfg)

    epoch_length = cfg.train.OFFICIAL_EPOCH_LENGTH
    warmup = cfg.optim.warmup_epochs * epoch_length
    teacher_warmup = cfg.teacher.warmup_teacher_temp_epochs * epoch_length
    freeze = cfg.optim.freeze_last_layer_epochs * epoch_length
    if not 0 <= warmup < total:
        raise ValueError("LR warmup must be shorter than optim.total_iterations")
    if not 0 <= teacher_warmup <= total or not 0 <= freeze < total:
        raise ValueError("Teacher warmup/last-layer freeze exceeds the step budget")

    # Upstream expresses durations as epochs. On this private copy one epoch
    # equals one step; warmup/freeze durations retain their original step counts.
    schedule_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    schedule_cfg.train.OFFICIAL_EPOCH_LENGTH = 1
    schedule_cfg.optim.epochs = total
    schedule_cfg.optim.warmup_epochs = warmup
    schedule_cfg.optim.freeze_last_layer_epochs = freeze
    schedule_cfg.teacher.warmup_teacher_temp_epochs = teacher_warmup
    return upstream_build_schedulers(schedule_cfg)
