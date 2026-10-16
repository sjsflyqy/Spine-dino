"""Distributed training entry point for the GCVD MVP."""

from __future__ import annotations

import logging
import math
import os
from functools import partial
from numbers import Integral

import torch
from fvcore.common.checkpoint import PeriodicCheckpointer

import dinov2.distributed as distributed
from dinov2.data import MaskingGenerator, SamplerType, make_data_loader, make_dataset
from dinov2.data.spine_webdataset import make_spine_webdataset
from dinov2.logging import MetricLogger
from dinov2.train.train import (
    apply_optim_scheduler,
    build_optimizer,
    cleanup_recovery_checkpoints,
    do_test,
    get_args_parser,
)
from dinov2.utils.config import setup

from ..data import DataAugmentationGeoTopoDINO, collate_data_and_cast_gcvd
from ..models.ssl_meta_arch import GeoTopoSSLMetaArch
from .checkpoint import GeoTopoCheckpointer
from .schedules import build_schedulers, get_schedule_iterations


torch.backends.cuda.matmul.allow_tf32 = True
logger = logging.getLogger("dinov2")


def get_accumulation_steps(cfg):
    steps = cfg.train.get("gradient_accumulation_steps", 1)
    if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 1:
        raise ValueError("train.gradient_accumulation_steps must be a positive integer")
    return int(steps)


def do_train(cfg, model, resume: bool = False):
    accumulation_steps = get_accumulation_steps(cfg)
    model.train()
    inputs_dtype = torch.half
    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)

    checkpointables = {"optimizer": optimizer}
    if model.fp16_scaler is not None:
        checkpointables["fp16_scaler"] = model.fp16_scaler
    checkpointer = GeoTopoCheckpointer(
        model, cfg.train.output_dir, save_to_disk=True,
        gradient_accumulation_steps=accumulation_steps, **checkpointables,
    )
    checkpoint_state = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume)
    start_iter = checkpoint_state.get("iteration", -1) + 1
    optimizer_updates = checkpoint_state.get("optimizer_updates", start_iter)
    schedule_max_iter = get_schedule_iterations(cfg)
    stop_after = int(getattr(cfg.train, "stop_after_iterations", 0))
    max_iter = min(schedule_max_iter, stop_after) if stop_after > 0 else schedule_max_iter
    logger.info("Scheduler budget=%d steps; training budget=%d steps", schedule_max_iter, max_iter)
    global_batch_size = cfg.train.batch_size_per_gpu * distributed.get_global_size()
    logger.info(
        "Gradient accumulation=%d; global microbatch=%d; effective batch=%d. "
        "Iteration budgets and schedules count microbatches; LR scaling is unchanged.",
        accumulation_steps, global_batch_size, global_batch_size * accumulation_steps,
    )
    if start_iter >= max_iter:
        logger.info("Target already reached: start_iter=%d max_iter=%d", start_iter, max_iter)
        return {}

    checkpoint_period = int(cfg.train.checkpoint_period)
    if checkpoint_period < 1:
        raise ValueError("train.checkpoint_period must be positive")
    eval_period = int(cfg.evaluation.eval_period_iterations)
    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period=checkpoint_period,
        max_iter=max_iter,
        max_to_keep=None,
    )

    image_size = int(cfg.crops.global_crops_size)
    patch_size = int(cfg.student.patch_size)
    grid_size = image_size // patch_size
    n_tokens = grid_size**2
    mask_generator = MaskingGenerator(
        input_size=(grid_size, grid_size),
        max_num_patches=0.5 * n_tokens,
    )
    data_transform = DataAugmentationGeoTopoDINO(
        cfg.crops.global_crops_scale,
        cfg.crops.local_crops_scale,
        cfg.crops.local_crops_number,
        global_crops_size=image_size,
        local_crops_size=int(cfg.crops.local_crops_size),
        patch_size=patch_size,
        crop_ratio=tuple(cfg.crops.crop_ratio),
        horizontal_flip_probability=float(cfg.crops.horizontal_flip_probability),
    )
    collate_fn = partial(
        collate_data_and_cast_gcvd,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )

    if cfg.train.dataset_path.startswith("SpineWebDataset:shards="):
        shards = cfg.train.dataset_path.split("=", 1)[1]
        dataset = make_spine_webdataset(
            shards,
            transform=data_transform,
            shuffle_buffer=5000,
            seed=cfg.train.seed,
        )
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.train.batch_size_per_gpu,
            num_workers=cfg.train.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=cfg.train.num_workers > 0,
            prefetch_factor=cfg.train.prefetch_factor if cfg.train.num_workers > 0 else None,
            collate_fn=collate_fn,
        )
    else:
        dataset = make_dataset(
            dataset_str=cfg.train.dataset_path,
            transform=data_transform,
            target_transform=lambda _: (),
        )
        data_loader = make_data_loader(
            dataset=dataset,
            batch_size=cfg.train.batch_size_per_gpu,
            num_workers=cfg.train.num_workers,
            shuffle=True,
            seed=start_iter,
            sampler_type=SamplerType.SHARDED_INFINITE,
            sampler_advance=0,
            drop_last=True,
            collate_fn=collate_fn,
            prefetch_factor=cfg.train.prefetch_factor,
        )

    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    logger.info("Starting GeoTopo-DINO training from iteration %d", start_iter)
    iteration = start_iter
    microbatches_in_update = 0
    for data in metric_logger.log_every(
        data_loader,
        10,
        "Training",
        max_iter,
        start_iter,
    ):
        if iteration >= max_iter:
            break
        if microbatches_in_update == 0:
            update_start_iter = iteration
            # Normalize a shortened final group by its actual size.
            update_size = min(accumulation_steps, max_iter - iteration)
            optimizer.zero_grad(set_to_none=True)
        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        momentum = momentum_schedule[iteration]
        teacher_temp = teacher_temp_schedule[iteration]
        last_layer_lr = last_layer_lr_schedule[iteration]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        loss_dict = model.forward_backward(
            data,
            teacher_temp=teacher_temp,
            iteration=iteration,
            schedule_max_iterations=schedule_max_iter,
            loss_scale=1.0 / update_size,
        )
        microbatches_in_update += 1
        is_update_boundary = microbatches_in_update == update_size
        if is_update_boundary:
            did_step = True
            if model.fp16_scaler is not None:
                if cfg.optim.clip_grad:
                    model.fp16_scaler.unscale_(optimizer)
                    for module in model.student.values():
                        module.clip_grad_norm_(cfg.optim.clip_grad)
                    for module in model.student_aux.values():
                        module.clip_grad_norm_(cfg.optim.clip_grad)
                previous_scale = model.fp16_scaler.get_scale()
                model.fp16_scaler.step(optimizer)
                model.fp16_scaler.update()
                # GradScaler lowers its scale when an overflow skips the step.
                did_step = model.fp16_scaler.get_scale() >= previous_scale
            else:
                if cfg.optim.clip_grad:
                    for module in model.student.values():
                        module.clip_grad_norm_(cfg.optim.clip_grad)
                    for module in model.student_aux.values():
                        module.clip_grad_norm_(cfg.optim.clip_grad)
                optimizer.step()
            if did_step:
                model.update_teacher(momentum)
                optimizer_updates += 1
            microbatches_in_update = 0

        if distributed.get_global_size() > 1:
            for value in loss_dict.values():
                torch.distributed.all_reduce(value)
        reduced = {
            key: value.item() / distributed.get_global_size()
            for key, value in loss_dict.items()
        }
        non_finite = {key: value for key, value in reduced.items() if not math.isfinite(value)}
        if non_finite:
            raise FloatingPointError(f"Non-finite training metrics detected: {non_finite}")
        metric_logger.update(lr=lr, wd=wd, mom=momentum, last_layer_lr=last_layer_lr)
        metric_logger.update(
            current_batch_size=data["collated_global_crops"].shape[0] / 2,
            gradient_accumulation_steps=accumulation_steps,
            effective_batch_size=global_batch_size * update_size,
            optimizer_updates=optimizer_updates,
            total_loss=reduced["optimization_loss"],
            **reduced,
        )

        is_final = iteration + 1 == max_iter
        eval_due = eval_period > 0 and (
            (iteration + 1) // eval_period > update_start_iter // eval_period
        )
        if is_update_boundary and (is_final or eval_due):
            do_test(cfg, model, f"training_{iteration}")
            torch.cuda.synchronize()
        checkpoint_due = (
            (iteration + 1) // checkpoint_period > update_start_iter // checkpoint_period
        )
        if is_update_boundary and (is_final or checkpoint_due):
            if checkpoint_due and (iteration + 1) % checkpoint_period and not is_final:
                # A period crossed inside a group is saved at the group's end.
                periodic_checkpointer.save(
                    f"model_{iteration:07d}", iteration=iteration, optimizer_updates=optimizer_updates,
                )
            else:
                periodic_checkpointer.step(iteration, optimizer_updates=optimizer_updates)
            cleanup_recovery_checkpoints(
                cfg.train.output_dir,
                int(cfg.train.checkpoint_max_to_keep),
            )
        iteration += 1

    if microbatches_in_update:
        raise RuntimeError(
            "Data loader exhausted before a gradient accumulation group completed. "
            "Check the data loader and train.stop_after_iterations; partial gradients were not checkpointed."
        )
    metric_logger.synchronize_between_processes()
    return {key: meter.global_avg for key, meter in metric_logger.meters.items()}


def main(args):
    cfg = setup(args)
    get_accumulation_steps(cfg)
    if int(cfg.crops.global_crops_size) != int(cfg.gcvd.anchor_size):
        raise ValueError("gcvd.anchor_size must equal crops.global_crops_size in the MVP")
    model = GeoTopoSSLMetaArch(cfg).to(torch.device("cuda"))
    model.prepare_for_distributed_training()
    logger.info("Model:\n%s", model)
    if args.eval_only:
        iteration = (
            GeoTopoCheckpointer(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1)
            + 1
        )
        return do_test(cfg, model, f"manual_{iteration}")
    return do_train(cfg, model, resume=not args.no_resume)


if __name__ == "__main__":
    main(get_args_parser(add_help=True).parse_args())
