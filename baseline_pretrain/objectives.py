# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0; see third-party notices.
"""DINO, CheXFound iBOT/KoLeo objectives; no GeoTopo losses are imported.

Equations adapted from the Apache-2.0 DINO/DINOv2/CheXFound references.
"""
from __future__ import annotations

import math

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


def distributed():
    return dist.is_available() and dist.is_initialized()


def all_sum(x):
    if distributed():
        dist.all_reduce(x)
    return x


@torch.no_grad()
def sinkhorn(logits, temperature, iterations=3):
    x = logits.float() / temperature
    maximum = x.max() if x.numel() else x.new_tensor(float("-inf"))
    if distributed():
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    q = (x - maximum).exp().t().clamp_min_(1e-20)
    count = all_sum(x.new_tensor(float(len(x))))
    if count <= 0:
        raise ValueError("Sinkhorn requires at least one token across ranks")
    q /= all_sum(q.sum()).clamp_min(1e-20)
    for _ in range(iterations):
        rows = all_sum(q.sum(1, keepdim=True)).clamp_min(1e-20)
        q /= rows
        q /= q.shape[0]
        q /= q.sum(0, keepdim=True).clamp_min(1e-20)
        q /= count
    return (q * count).t()


def koleo(features, scope):
    x = F.normalize(features.float(), dim=-1, eps=1e-8)
    candidates = x
    offset = 0
    if scope == "global" and distributed():
        from torch.distributed.nn.functional import all_gather
        candidates = torch.cat(all_gather(x), 0)
        offset = dist.get_rank() * len(x)
    if len(candidates) < 2:
        raise ValueError("KoLeo needs >=2 samples per global view. Use batch>=2 or global scope with >=2 ranks")
    with torch.no_grad():
        similarities = x @ candidates.t()
        similarities[torch.arange(len(x), device=x.device), offset + torch.arange(len(x), device=x.device)] = -float("inf")
        nearest = similarities.argmax(1)
    distances = F.pairwise_distance(x, candidates[nearest], eps=1e-8)
    return -(distances + 1e-8).log().mean()


class DistillationObjective(nn.Module):
    def __init__(self, method, cfg):
        super().__init__()
        self.method, self.cfg = method, cfg
        self.register_buffer("center", torch.zeros(1, cfg["prototypes"]))
        self.register_buffer("pending_sum", torch.zeros(1, cfg["prototypes"]))
        self.register_buffer("pending_count", torch.zeros(()))

    def temperature(self, update):
        warmup = self.cfg["temperature_warmup_updates"]
        f = min(update / max(warmup - 1, 1), 1) if warmup else 1
        return self.cfg["teacher_temperature_start"] + f * (self.cfg["teacher_temperature"] - self.cfg["teacher_temperature_start"])

    def forward(self, student, teacher, masks, update):
        c = self.cfg
        teacher_logits = teacher["cls_logits"].detach().float()
        temperature = self.temperature(update)
        with torch.no_grad():
            if c["centering"] == "sinkhorn_knopp":
                targets = sinkhorn(teacher_logits, temperature)
            else:
                targets = F.softmax((teacher_logits - self.center) / temperature, -1)
                self.pending_sum.add_(teacher_logits.sum(0, keepdim=True))
                self.pending_count.add_(len(teacher_logits))
        b = len(teacher_logits) // 2
        predictions = F.log_softmax(student["cls_logits"].float() / c["student_temperature"], -1).split(b)
        terms = [-(target * prediction).sum(-1).mean()
                 for i, target in enumerate(targets.split(b))
                 for j, prediction in enumerate(predictions) if i != j]
        dino = torch.stack(terms).mean()
        loss = dino * c["dino_weight"]
        metrics = {"dino_loss": dino.detach()}
        if self.method == "chexfound":
            with torch.no_grad():
                patch_targets = sinkhorn(teacher["patch_logits"].detach(), temperature)
            per_patch = -(patch_targets * F.log_softmax(student["patch_logits"].float() / c["student_temperature"], -1)).sum(-1)
            weights = masks.sum(-1).clamp_min(1).reciprocal().unsqueeze(1).expand_as(masks)[masks]
            ibot = (per_patch * weights).sum() / len(masks)
            loss = loss + c["ibot_weight"] * ibot
            metrics["ibot_loss"] = ibot.detach()
            if c["koleo_weight"]:
                regularizer = sum(koleo(x, c["koleo_scope"]) for x in student["global_cls"].chunk(2))
                loss = loss + c["koleo_weight"] * regularizer
                metrics["koleo_loss"] = regularizer.detach()
        return loss, metrics

    @torch.no_grad()
    def finish_update(self):
        if self.cfg["centering"] == "centering":
            all_sum(self.pending_sum)
            all_sum(self.pending_count)
            if self.pending_count > 0:
                self.center.lerp_(self.pending_sum / self.pending_count, 1 - self.cfg["center_momentum"])
            self.pending_sum.zero_()
            self.pending_count.zero_()


def schedule(start, end, index, total):
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * index / max(total - 1, 1)))


def learning_rate(training, update):
    warmup = training["warmup_updates"]
    if update < warmup:
        return training["learning_rate"] * (update + 1) / max(warmup, 1)
    return schedule(training["learning_rate"], training["min_learning_rate"], update - warmup,
                    training["total_updates"] - warmup)
