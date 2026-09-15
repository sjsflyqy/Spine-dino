"""DINOv2 meta-architecture with geometry-conditioned view distillation."""

from __future__ import annotations

import torch
from torch import nn

from dinov2.fsdp import reshard_fsdp_model
from dinov2.train.ssl_meta_arch import SSLMetaArch

try:
    from xformers.ops import fmha
except ImportError as exc:  # pragma: no cover - training dependency
    raise AssertionError("xFormers is required for training") from exc

from ..geometry.warp import warp_anchor_features_to_local
from ..losses.gcvd_loss import (
    GCVDPrototypeLoss,
    GeometryDistillationLoss,
    compute_gcvd_warmup_scale,
    valid_mean_pool,
)
from ..masking import BlockMaskPolicy, pack_masks
from .projection_head import GCVDPrototypeHead, GeometryProjectionHead


class GeoTopoSSLMetaArch(SSLMetaArch):
    """MVP: one anchor + one random global, original SSL losses, plus GCVD."""

    def __init__(self, cfg):
        super().__init__(cfg)
        # Future local-order predictors live here: they are optimized with the
        # student but deliberately excluded from teacher EMA updates.
        self.student_aux = nn.ModuleDict()
        self.gcvd_enabled = bool(cfg.gcvd.enabled)
        self.use_standard_local_dino = bool(cfg.gcvd.use_standard_local_dino)
        self.gcvd_loss_weight = float(cfg.gcvd.loss_weight)
        self.gcvd_dense_weight = float(cfg.gcvd.dense_weight)
        self.gcvd_region_weight = float(cfg.gcvd.region_weight)
        self.gcvd_loss_type = str(cfg.gcvd.loss_type)
        self.gcvd_warmup_type = str(cfg.gcvd.warmup_type)
        self.gcvd_warmup_fraction = float(cfg.gcvd.warmup_fraction)
        self.gcvd_warmup_iterations = int(cfg.gcvd.warmup_iterations)

        if self.gcvd_loss_type not in ("cosine", "prototype_ce"):
            raise ValueError("gcvd.loss_type must be cosine or prototype_ce")
        if min(
            self.gcvd_loss_weight,
            self.gcvd_dense_weight,
            self.gcvd_region_weight,
        ) < 0.0:
            raise ValueError("GCVD loss weights must be non-negative")
        self._gcvd_scale(
            iteration=0,
            schedule_max_iterations=int(
                cfg.optim.epochs * cfg.train.OFFICIAL_EPOCH_LENGTH
            ),
        )

        if bool(cfg.local_order.enabled):
            raise NotImplementedError("local_order is reserved but is not part of the GCVD MVP")
        if bool(cfg.tgsr.enabled):
            raise NotImplementedError("TGSR is reserved but is not part of the GCVD MVP")

        if self.gcvd_enabled:
            student_head = GeometryProjectionHead(
                self.embed_dim,
                out_dim=int(cfg.gcvd.projection_dim),
                hidden_dim=int(cfg.gcvd.projection_hidden_dim),
            )
            teacher_head = GeometryProjectionHead(
                self.embed_dim,
                out_dim=int(cfg.gcvd.projection_dim),
                hidden_dim=int(cfg.gcvd.projection_hidden_dim),
            )
            self.student["geom_head"] = student_head
            self.teacher["geom_head"] = teacher_head
            for parameter in self.teacher.geom_head.parameters():
                parameter.requires_grad = False
            self.gcvd_loss = GeometryDistillationLoss()
            if self.gcvd_loss_type == "prototype_ce":
                prototype_head_kwargs = {
                    "in_dim": self.embed_dim,
                    "out_dim": int(cfg.gcvd.head_n_prototypes),
                    "hidden_dim": int(cfg.gcvd.head_hidden_dim),
                    "bottleneck_dim": int(cfg.gcvd.head_bottleneck_dim),
                    "nlayers": int(cfg.gcvd.head_nlayers),
                }
                self.student["gcvd_proto_head"] = GCVDPrototypeHead(**prototype_head_kwargs)
                self.teacher["gcvd_proto_head"] = GCVDPrototypeHead(**prototype_head_kwargs)
                for parameter in self.teacher.gcvd_proto_head.parameters():
                    parameter.requires_grad = False
                self.gcvd_prototype_loss = GCVDPrototypeLoss(
                    int(cfg.gcvd.head_n_prototypes),
                    teacher_temp=float(cfg.gcvd.teacher_temp),
                    student_temp=float(cfg.gcvd.student_temp),
                    center_momentum=float(cfg.gcvd.center_momentum),
                    teacher_centering=str(
                        getattr(cfg.gcvd, "teacher_centering", "centering")
                    ),
                    sinkhorn_iterations=int(
                        getattr(cfg.gcvd, "sinkhorn_iterations", 3)
                    ),
                )

        anchor_policy = str(cfg.masking.anchor_policy)
        random_policy = str(cfg.masking.random_global_policy)
        if anchor_policy != "block" or random_policy != "block":
            raise NotImplementedError(
                "The MVP implements block/block masking; topology_mixed is reserved for TGSR"
            )
        self.mask_policy = BlockMaskPolicy()

    def get_params_groups(self):
        groups = super().get_params_groups()
        for module in self.student_aux.values():
            groups += self.get_maybe_fused_params_for_submodel(module)
        return groups

    def _gcvd_scale(self, iteration: int, schedule_max_iterations: int) -> float:
        return compute_gcvd_warmup_scale(
            warmup_type=self.gcvd_warmup_type,
            iteration=iteration,
            schedule_max_iterations=schedule_max_iterations,
            warmup_fraction=self.gcvd_warmup_fraction,
            warmup_iterations=self.gcvd_warmup_iterations,
        )

    def forward_backward(
        self,
        images,
        teacher_temp,
        *,
        iteration: int = 0,
        schedule_max_iterations: int | None = None,
    ):
        if schedule_max_iterations is None:
            schedule_max_iterations = int(
                self.cfg.optim.epochs * self.cfg.train.OFFICIAL_EPOCH_LENGTH
            )
        progress = float(iteration) / float(max(schedule_max_iterations - 1, 1))
        n_global_crops = 2
        n_local_crops = self.cfg.crops.local_crops_number
        global_crops = images["collated_global_crops"].cuda(non_blocking=True)
        local_crops = images["collated_local_crops"].cuda(non_blocking=True)
        candidate_masks = images["collated_masks"].cuda(non_blocking=True)
        anchor_transforms = images["anchor_transforms"].cuda(non_blocking=True)
        anchor_valid_masks = images["anchor_valid_masks"].cuda(non_blocking=True)
        local_transforms = images["local_transforms"].cuda(non_blocking=True)
        local_sample_ids = images["local_sample_ids"].cuda(non_blocking=True)
        upperbound = int(images["upperbound"])

        local_dino_terms = n_local_crops * n_global_crops if self.use_standard_local_dino else 0
        global_dino_terms = (n_global_crops - 1) * n_global_crops
        dino_term_count = max(global_dino_terms + local_dino_terms, 1)
        ibot_loss_scale = 1.0 / n_global_crops
        do_dino = self.do_dino
        do_ibot = self.do_ibot

        batch_size = global_crops.shape[0] // n_global_crops
        patch_size = int(self.cfg.student.patch_size)
        anchor_height = global_crops.shape[-2] // patch_size
        anchor_width = global_crops.shape[-1] // patch_size
        local_height = local_crops.shape[-2] // patch_size
        local_width = local_crops.shape[-1] // patch_size

        @torch.no_grad()
        def get_teacher_targets():
            teacher_output = self.teacher.backbone(global_crops, is_training=True)
            teacher_cls = teacher_output["x_norm_clstoken"]
            teacher_patch = teacher_output["x_norm_patchtokens"]
            teacher_anchor_patch = teacher_patch[:batch_size]

            final_masks, topology_state = self.mask_policy.select(
                candidate_masks,
                teacher_anchor_tokens=teacher_anchor_patch,
                anchor_valid_mask=anchor_valid_masks,
                progress=progress,
            )
            mask_state = pack_masks(
                final_masks,
                upperbound=upperbound,
                topology=topology_state,
            )

            teacher_cls_chunks = teacher_cls.chunk(n_global_crops)
            teacher_cls_reversed = torch.cat((teacher_cls_chunks[1], teacher_cls_chunks[0]))
            feature_dim = teacher_patch.shape[-1]
            n_cls_tokens = teacher_cls_reversed.shape[0]
            n_masked = mask_state.n_masked_patches

            if do_ibot and not self.ibot_separate_head:
                buffer = teacher_patch.new_zeros(upperbound + n_cls_tokens, feature_dim)
                buffer[:n_cls_tokens].copy_(teacher_cls_reversed)
                torch.index_select(
                    teacher_patch.flatten(0, 1),
                    dim=0,
                    index=mask_state.indices,
                    out=buffer[n_cls_tokens : n_cls_tokens + n_masked],
                )
                tokens_after_head = self.teacher.dino_head(buffer)
                teacher_cls_after_head = tokens_after_head[:n_cls_tokens]
                teacher_masked_patch_after_head = tokens_after_head[
                    n_cls_tokens : n_cls_tokens + n_masked
                ]
            elif do_ibot and self.ibot_separate_head:
                buffer = teacher_patch.new_zeros(upperbound, feature_dim)
                torch.index_select(
                    teacher_patch.flatten(0, 1),
                    dim=0,
                    index=mask_state.indices,
                    out=buffer[:n_masked],
                )
                teacher_cls_after_head = self.teacher.dino_head(teacher_cls_reversed)
                teacher_masked_patch_after_head = self.teacher.ibot_head(buffer)[:n_masked]
            else:
                teacher_cls_after_head = self.teacher.dino_head(teacher_cls_reversed)
                teacher_masked_patch_after_head = None

            if self.cfg.train.centering == "centering":
                teacher_dino_targets = self.dino_loss.softmax_center_teacher(
                    teacher_cls_after_head,
                    teacher_temp=teacher_temp,
                ).view(n_global_crops, -1, *teacher_cls_after_head.shape[1:])
                self.dino_loss.update_center(teacher_cls_after_head)
                if do_ibot:
                    teacher_masked_patch_after_head = teacher_masked_patch_after_head.unsqueeze(0)
                    teacher_ibot_targets = self.ibot_patch_loss.softmax_center_teacher(
                        teacher_masked_patch_after_head[:, :n_masked],
                        teacher_temp=teacher_temp,
                    ).squeeze(0)
                    self.ibot_patch_loss.update_center(teacher_masked_patch_after_head[:n_masked])
                else:
                    teacher_ibot_targets = None
            elif self.cfg.train.centering == "sinkhorn_knopp":
                teacher_dino_targets = self.dino_loss.sinkhorn_knopp_teacher(
                    teacher_cls_after_head,
                    teacher_temp=teacher_temp,
                ).view(n_global_crops, -1, *teacher_cls_after_head.shape[1:])
                teacher_ibot_targets = (
                    self.ibot_patch_loss.sinkhorn_knopp_teacher(
                        teacher_masked_patch_after_head,
                        teacher_temp=teacher_temp,
                        n_masked_patches_tensor=mask_state.n_masked_patches_tensor,
                    )
                    if do_ibot
                    else None
                )
            else:
                raise NotImplementedError(self.cfg.train.centering)

            if self.gcvd_enabled:
                aligned_teacher, correspondence_valid = warp_anchor_features_to_local(
                    teacher_anchor_patch,
                    anchor_transforms,
                    local_transforms,
                    local_sample_ids,
                    anchor_valid_masks,
                    patch_size=patch_size,
                    anchor_grid_size=(anchor_height, anchor_width),
                    local_grid_size=(local_height, local_width),
                )
                teacher_region = valid_mean_pool(aligned_teacher, correspondence_valid)
                teacher_region_geometry = self.teacher.geom_head(
                    teacher_region.to(dtype=teacher_anchor_patch.dtype)
                )
                if self.gcvd_loss_type == "cosine":
                    teacher_patch_geometry = self.teacher.geom_head(
                        aligned_teacher.to(dtype=teacher_anchor_patch.dtype)
                    )
                    teacher_patch_probabilities = None
                    teacher_prototype_diagnostics = None
                else:
                    teacher_patch_geometry = None
                    valid_teacher_tokens = aligned_teacher.reshape(-1, feature_dim)[
                        correspondence_valid.reshape(-1)
                    ]
                    teacher_gcvd_logits = self.teacher.gcvd_proto_head(
                        valid_teacher_tokens.to(dtype=teacher_anchor_patch.dtype)
                    )
                    (
                        teacher_patch_probabilities,
                        teacher_prototype_diagnostics,
                    ) = self.gcvd_prototype_loss.teacher_probabilities(teacher_gcvd_logits)
            else:
                teacher_patch_geometry = None
                teacher_region_geometry = None
                teacher_patch_probabilities = None
                teacher_prototype_diagnostics = None
                correspondence_valid = None

            return (
                teacher_dino_targets,
                teacher_ibot_targets,
                teacher_patch_geometry,
                teacher_region_geometry,
                teacher_patch_probabilities,
                teacher_prototype_diagnostics,
                correspondence_valid,
                mask_state,
            )

        (
            teacher_dino_targets,
            teacher_ibot_targets,
            teacher_patch_geometry,
            teacher_region_geometry,
            teacher_patch_probabilities,
            teacher_prototype_diagnostics,
            correspondence_valid,
            mask_state,
        ) = get_teacher_targets()
        reshard_fsdp_model(self.teacher)

        student_global, student_local = self.student.backbone(
            [global_crops, local_crops],
            masks=[mask_state.masks, None],
            is_training=True,
        )
        inputs_for_head = []
        if self.use_standard_local_dino:
            inputs_for_head.append(student_local["x_norm_clstoken"].unsqueeze(0))
        inputs_for_head.append(student_global["x_norm_clstoken"].unsqueeze(0))

        n_masked = mask_state.n_masked_patches
        if do_ibot:
            feature_dim = student_global["x_norm_clstoken"].shape[-1]
            student_patch = student_global["x_norm_patchtokens"]
            patch_buffer = student_patch.new_zeros(upperbound, feature_dim)
            patch_buffer[:n_masked].copy_(
                torch.index_select(student_patch.flatten(0, 1), dim=0, index=mask_state.indices)
            )
            if not self.ibot_separate_head:
                inputs_for_head.append(patch_buffer.unsqueeze(0))
            else:
                student_masked_patch_after_head = self.student.ibot_head(patch_buffer)[:n_masked]

        attention_bias, concatenated = fmha.BlockDiagonalMask.from_tensor_list(inputs_for_head)
        head_outputs = attention_bias.split(self.student.dino_head(concatenated))
        if self.use_standard_local_dino:
            student_local_cls_after_head = head_outputs.pop(0).squeeze(0)
        else:
            student_local_cls_after_head = None
        student_global_cls_after_head = head_outputs.pop(0).squeeze(0)
        if do_ibot and not self.ibot_separate_head:
            student_masked_patch_after_head = head_outputs.pop(0).squeeze(0)[:n_masked]

        loss_dict = {}
        loss_accumulator = 0.0
        if do_dino and self.use_standard_local_dino and n_local_crops > 0:
            dino_local_loss = self.dino_loss(
                student_output_list=student_local_cls_after_head.chunk(n_local_crops),
                teacher_out_softmaxed_centered_list=teacher_dino_targets,
            ) / dino_term_count
            loss_dict["dino_local_crops_loss"] = dino_local_loss
            loss_accumulator = loss_accumulator + self.dino_loss_weight * dino_local_loss

        loss_scales = 2
        if do_dino:
            dino_global_loss = (
                self.dino_loss(
                    student_output_list=[student_global_cls_after_head],
                    teacher_out_softmaxed_centered_list=[teacher_dino_targets.flatten(0, 1)],
                )
                * loss_scales
                / dino_term_count
            )
            loss_dict["dino_global_crops_loss"] = dino_global_loss
            loss_accumulator = loss_accumulator + self.dino_loss_weight * dino_global_loss
            if self.do_koleo:
                koleo_loss = self.cfg.dino.koleo_loss_weight * sum(
                    self.koleo_loss(tokens)
                    for tokens in student_global["x_norm_clstoken"].chunk(n_global_crops)
                )
                loss_accumulator = loss_accumulator + koleo_loss
                loss_dict["koleo_loss"] = koleo_loss / loss_scales

        if do_ibot:
            ibot_loss = (
                self.ibot_patch_loss.forward_masked(
                    student_masked_patch_after_head,
                    teacher_ibot_targets,
                    student_masks_flat=mask_state.masks,
                    n_masked_patches=n_masked,
                    masks_weight=mask_state.weights,
                )
                * loss_scales
                * ibot_loss_scale
            )
            loss_dict["ibot_loss"] = ibot_loss / 2
            loss_accumulator = loss_accumulator + self.ibot_loss_weight * ibot_loss

        if self.gcvd_enabled:
            student_local_patch = student_local["x_norm_patchtokens"]
            student_region = valid_mean_pool(student_local_patch.float(), correspondence_valid)
            student_region_geometry = self.student.geom_head(
                student_region.to(dtype=student_local_patch.dtype)
            )
            region_raw_loss = self.gcvd_loss.region_loss(
                student_region_geometry,
                teacher_region_geometry,
                correspondence_valid,
            )
            if self.gcvd_loss_type == "cosine":
                student_patch_geometry = self.student.geom_head(student_local_patch)
                dense_raw_loss = self.gcvd_loss.dense_loss(
                    student_patch_geometry,
                    teacher_patch_geometry,
                    correspondence_valid,
                )
                student_prototype_diagnostics = None
            else:
                valid_student_tokens = student_local_patch.reshape(-1, self.embed_dim)[
                    correspondence_valid.reshape(-1)
                ]
                student_gcvd_logits = self.student.gcvd_proto_head(valid_student_tokens)
                dense_raw_loss, student_prototype_diagnostics = self.gcvd_prototype_loss(
                    student_gcvd_logits,
                    teacher_patch_probabilities,
                )

            schedule_scale = self._gcvd_scale(iteration, schedule_max_iterations)
            dense_contribution = (
                self.gcvd_loss_weight
                * schedule_scale
                * self.gcvd_dense_weight
                * dense_raw_loss
            )
            region_contribution = (
                self.gcvd_loss_weight
                * schedule_scale
                * self.gcvd_region_weight
                * region_raw_loss
            )
            metric_reference = dense_raw_loss.detach()
            loss_dict["gcvd_warmup_scale"] = metric_reference.new_tensor(schedule_scale)
            loss_dict["gcvd_dense_raw_loss"] = dense_raw_loss.detach()
            loss_dict["gcvd_region_raw_loss"] = region_raw_loss.detach()
            loss_dict["gcvd_dense_weighted_loss"] = dense_contribution.detach()
            loss_dict["gcvd_region_weighted_loss"] = region_contribution.detach()
            loss_dict["gcvd_valid_ratio"] = correspondence_valid.float().mean()
            if self.gcvd_loss_type == "prototype_ce":
                loss_dict["gcvd_dense_raw_kl"] = (
                    dense_raw_loss.detach()
                    - teacher_prototype_diagnostics["entropy"]
                )
                loss_dict["gcvd_teacher_entropy"] = teacher_prototype_diagnostics["entropy"]
                loss_dict["gcvd_student_entropy"] = student_prototype_diagnostics["entropy"]
                loss_dict["gcvd_teacher_max_prob"] = teacher_prototype_diagnostics["max_prob"]
                loss_dict["gcvd_student_max_prob"] = student_prototype_diagnostics["max_prob"]
                loss_dict["gcvd_teacher_active_prototype_ratio"] = (
                    teacher_prototype_diagnostics["active_prototype_ratio"]
                )
                loss_dict["gcvd_student_active_prototype_ratio"] = (
                    student_prototype_diagnostics["active_prototype_ratio"]
                )
                loss_dict["gcvd_teacher_marginal_entropy"] = (
                    teacher_prototype_diagnostics["marginal_entropy"]
                )
                loss_dict["gcvd_student_marginal_entropy"] = (
                    student_prototype_diagnostics["marginal_entropy"]
                )
                loss_dict["gcvd_teacher_effective_prototype_ratio"] = (
                    teacher_prototype_diagnostics["effective_prototype_ratio"]
                )
                loss_dict["gcvd_student_effective_prototype_ratio"] = (
                    student_prototype_diagnostics["effective_prototype_ratio"]
                )
            loss_accumulator = loss_accumulator + dense_contribution + region_contribution

        if not torch.isfinite(loss_accumulator.detach()).all():
            raise FloatingPointError("Non-finite optimization loss detected")
        loss_dict["optimization_loss"] = loss_accumulator.detach()
        self.backprop_loss(loss_accumulator)
        self.fsdp_synchronize_streams()
        return loss_dict

    def fsdp_synchronize_streams(self):
        if self.need_to_synchronize_fsdp_streams:
            torch.cuda.synchronize()
            if hasattr(self.teacher.backbone, "_streams"):
                streams = self.teacher.backbone._streams
                for module in self.student.values():
                    module._streams = streams
                for module in self.teacher.values():
                    module._streams = streams
            self.need_to_synchronize_fsdp_streams = False
