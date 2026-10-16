"""Run the actual accumulated trainer on two GPUs with synthetic images.

torchrun --standalone --nproc-per-node=2 --module \
    methods.geotopo_dino.tests.smoke_accumulation_fsdp --output-dir /tmp/accumulation-fsdp

Only dataset access is replaced. Uses real ViT-small, GCVD prototype heads,
pixel decoder, mixed precision, FSDP, AdamW, EMA and staged checkpoint resume.
"""

import argparse
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

import dinov2.distributed as distributed
from methods.geotopo_dino.models.ssl_meta_arch import GeoTopoSSLMetaArch
from methods.geotopo_dino.train import train as trainer
from methods.geotopo_dino.train.checkpoint import GeoTopoCheckpointer
from methods.geotopo_dino.tests.smoke_pixel_fsdp import check_decoder_sync
from methods.geotopo_dino.tests.test_pixel_reconstruction import synthetic_batch, tiny_config


def run_stage(cfg, model, count, expected_updates, resume=False, expected_parameters=None):
    data = []
    for _ in range(count):
        batch = synthetic_batch()
        for key in ("collated_global_crops", "collated_local_crops"):
            batch[key] = batch[key].half()
        if distributed.get_global_rank() % 2:
            batch["collated_masks"][0, 0] = False
        data.append(batch)
    original_resume = GeoTopoCheckpointer.resume_or_load

    def checked_resume(checkpointer, *args, **kwargs):
        state = original_resume(checkpointer, *args, **kwargs)
        if expected_parameters is not None:
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(parameter, expected_parameters[name], rtol=0, atol=0)
            assert model.fp16_scaler.get_scale() == 128.0
        return state

    with patch.object(trainer, "make_dataset", return_value=None), \
         patch.object(trainer, "make_data_loader", return_value=data), \
         patch.object(GeoTopoCheckpointer, "resume_or_load", autospec=True, side_effect=checked_resume), \
         patch.object(model, "update_teacher", wraps=model.update_teacher) as ema, \
         patch.object(model.fp16_scaler, "step", wraps=model.fp16_scaler.step) as step, \
         patch.object(model.fp16_scaler, "unscale_", wraps=model.fp16_scaler.unscale_) as unscale:
        metrics = trainer.do_train(cfg, model, resume=resume)
        assert step.call_count == expected_updates, (step.call_count, expected_updates)
        assert ema.call_count == expected_updates, (ema.call_count, expected_updates)
        assert unscale.call_count == expected_updates, (unscale.call_count, expected_updates)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert all(parameter.grad is None for parameter in model.teacher.parameters())
    check_decoder_sync(model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this FSDP integration test.")
    distributed.enable(overwrite=True)
    torch.set_num_threads(1)
    torch.manual_seed(42 + distributed.get_global_rank())
    cfg = tiny_config(True)
    # Avoid the xFormers stochastic-depth indexing kernel's tiny-batch limit.
    cfg.student.drop_path_rate = 0.0
    cfg.compute_precision.grad_scaler = True
    cfg.train.update(output_dir=args.output_dir, batch_size_per_gpu=1, num_workers=0,
                     OFFICIAL_EPOCH_LENGTH=1, gradient_accumulation_steps=3,
                     checkpoint_period=2, checkpoint_max_to_keep=4, stop_after_iterations=3)
    cfg.optim.update(total_iterations=7, lr=1e-4, warmup_epochs=0, freeze_last_layer_epochs=0)
    cfg.teacher.warmup_teacher_temp_epochs = 0
    cfg.evaluation.eval_period_iterations = 2
    cfg.gcvd.update(enabled=True, loss_type="prototype_ce", head_n_prototypes=8,
                    head_hidden_dim=16, head_bottleneck_dim=8, head_nlayers=2,
                    teacher_centering="sinkhorn_knopp", warmup_type="no_warmup")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model = GeoTopoSSLMetaArch(cfg).cuda()
    model.prepare_for_distributed_training()
    model.fp16_scaler = ShardedGradScaler(init_scale=128.0)
    before = [parameter.detach().clone() for parameter in model.student_aux.parameters()]
    run_stage(cfg, model, count=3, expected_updates=1)
    changed = torch.tensor(int(any(not torch.equal(old, parameter) for old, parameter in
                                  zip(before, model.student_aux.parameters()))), device="cuda")
    dist.all_reduce(changed)
    assert changed.item() > 0, "Accumulated gradients did not update the pixel decoder"
    saved_parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    # A fresh model/scaler must load all training state and continue at microbatch 3.
    cfg.train.stop_after_iterations = 7
    restored = GeoTopoSSLMetaArch(cfg).cuda()
    restored.prepare_for_distributed_training()
    restored.fp16_scaler = ShardedGradScaler(init_scale=16.0)
    run_stage(cfg, restored, count=4, expected_updates=2, resume=True, expected_parameters=saved_parameters)
    path = Path(args.output_dir) / f"model_final.rank_{distributed.get_global_rank()}.pth"
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["iteration"] == 6
    assert state["optimizer_updates"] == 3
    assert state["gradient_accumulation_steps"] == 3
    assert state["fp16_scaler"]["scale"] == 128.0
    if distributed.is_main_process():
        print("PASS: real two-GPU FSDP accumulation, AMP, GCVD/pixel, EMA, partial final group and resume")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
