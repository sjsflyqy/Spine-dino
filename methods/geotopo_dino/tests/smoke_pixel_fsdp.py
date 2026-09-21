"""Real CUDA/FSDP integration check on synthetic images, with no dataset needed.

torchrun --standalone --nproc-per-node=2 --module \
    methods.geotopo_dino.tests.smoke_pixel_fsdp --output-dir /tmp/pixel-fsdp-check

Uses the actual ViT-small/xFormers teacher/student, mixed precision, optimizer,
EMA, decoder synchronization and FSDP checkpoint restore. No CPU stand-ins.
"""

import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import dinov2.distributed as distributed
from methods.geotopo_dino.models.ssl_meta_arch import GeoTopoSSLMetaArch
from methods.geotopo_dino.train.checkpoint import GeoTopoCheckpointer
from methods.geotopo_dino.tests.test_pixel_reconstruction import synthetic_batch, tiny_config


def check_decoder_sync(model):
    with FSDP.summon_full_params(model.student_aux.pixel_decoder, writeback=False):
        parameters = torch.cat([p.detach().flatten() for p in model.student_aux.parameters()])
        gathered = [torch.empty_like(parameters) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, parameters)
        for other in gathered:
            torch.testing.assert_close(parameters, other, rtol=0, atol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This smoke test requires CUDA; run the unittest suite for CPU checks.")
    distributed.enable(overwrite=True)
    rank = distributed.get_global_rank()
    torch.manual_seed(42 + rank)
    cfg = tiny_config(True)
    cfg.compute_precision.grad_scaler = True
    cfg.train.output_dir = args.output_dir
    model = GeoTopoSSLMetaArch(cfg).cuda()
    model.prepare_for_distributed_training()
    model.train()
    check_decoder_sync(model)
    optimizer = torch.optim.AdamW(model.get_params_groups(), lr=1e-4)
    before = [p.detach().clone() for p in model.student_aux.parameters()]
    for iteration in range(2):
        batch = synthetic_batch(device="cuda")
        batch["collated_global_crops"] = batch["collated_global_crops"].half()
        batch["collated_local_crops"] = batch["collated_local_crops"].half()
        # Unequal masked counts across ranks exercise global loss normalization.
        if rank % 2:
            batch["collated_masks"][0, 0] = False
        optimizer.zero_grad(set_to_none=True)
        result = model.forward_backward(batch, teacher_temp=0.07, iteration=iteration)
        assert all(torch.isfinite(value).all() for value in result.values())
        assert all(p.grad is None for p in model.teacher.parameters())
        model.fp16_scaler.unscale_(optimizer)
        for module in list(model.student.values()) + list(model.student_aux.values()):
            norm = module.clip_grad_norm_(3.0)
            assert torch.isfinite(norm)
        model.fp16_scaler.step(optimizer)
        model.fp16_scaler.update()
        model.update_teacher(0.994)
    changed = torch.tensor(
        int(any(not torch.equal(old, p) for old, p in zip(before, model.student_aux.parameters()))),
        device="cuda",
    )
    dist.all_reduce(changed)
    assert changed.item() > 0, "decoder was not updated"
    check_decoder_sync(model)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    checkpointer = GeoTopoCheckpointer(model, args.output_dir, optimizer=optimizer)
    checkpointer.save("pixel_smoke", iteration=1)
    saved = [p.detach().clone() for p in model.student_aux.parameters()]
    with torch.no_grad():
        for parameter in model.student_aux.parameters():
            parameter.add_(0.25)
    state = checkpointer.resume_or_load("", resume=True)
    assert state["iteration"] == 1
    for expected, actual in zip(saved, model.student_aux.parameters()):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    check_decoder_sync(model)
    if distributed.is_main_process():
        print("PASS: real ViT forward/backward, decoder update/sync, EMA and FSDP checkpoint restore")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
