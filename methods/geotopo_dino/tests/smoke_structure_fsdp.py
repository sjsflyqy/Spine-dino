"""Real CUDA/xFormers/FSDP check; synthetic images, no training dataset needed.

Run with two ranks using torchrun --module. Exercises warmup, active local-width
mixed masks, GCVD/iBOT/pixel backward, EMA, optimizer and checkpoint resume.
Requires CUDA. The smaller geometry and p_max=1 are smoke-only overrides.
"""

import argparse
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

import dinov2.distributed as distributed
from methods.geotopo_dino.models.ssl_meta_arch import GeoTopoSSLMetaArch
from methods.geotopo_dino.tests.test_structure_mask import structure_tiny_config, training_batch
from methods.geotopo_dino.tests.smoke_pixel_fsdp import check_decoder_sync
from methods.geotopo_dino.train.checkpoint import GeoTopoCheckpointer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this FSDP smoke; use test_structure_mask for CPU verification.")
    distributed.enable(overwrite=True)
    rank = distributed.get_global_rank()
    torch.manual_seed(42 + rank)
    cfg = structure_tiny_config()
    cfg.compute_precision.grad_scaler = True
    cfg.train.output_dir = str(args.output_dir)
    cfg.masking.structure.update(warmup_iterations=6000, ramp_iterations=6000, max_probability=1.0,
                                 width_mode="local", min_width=3, max_width=5, visualization_period=1)
    model = GeoTopoSSLMetaArch(cfg).cuda()
    # Random tiny heads can overflow with the default 65536 scale before the
    # scaler calibrates. Keep real mixed precision, with a smoke-only low scale.
    model.fp16_scaler = ShardedGradScaler(init_scale=128.0)
    model.prepare_for_distributed_training()
    model.train()
    optimizer = torch.optim.AdamW(model.get_params_groups(), lr=1e-4)
    check_decoder_sync(model)
    seen = {}
    hook = model.pixel_loss.register_forward_pre_hook(lambda module, inputs: seen.update(final=inputs[2].clone()))
    applied = 0
    steps = []
    initial_student = [parameter.detach().clone() for parameter in model.student.parameters()]
    initial_decoder = [parameter.detach().clone() for parameter in model.student_aux.parameters()]
    for iteration in (5999, 9000, 12000):
        started = time.perf_counter()
        batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in training_batch().items()}
        batch["collated_global_crops"] = batch["collated_global_crops"].half()
        batch["collated_local_crops"] = batch["collated_local_crops"].half()
        original = batch["collated_masks"].clone()
        optimizer.zero_grad(set_to_none=True)
        result = model.forward_backward(batch, .07, iteration=iteration)
        assert all(torch.isfinite(value).all() for value in result.values())
        torch.testing.assert_close(seen["final"].sum(-1), original.sum(-1))
        torch.testing.assert_close(seen["final"][2:], original[2:])
        if iteration == 5999:
            torch.testing.assert_close(seen["final"], original)
        for example in model.mask_policy.last_examples:
            index = example["anchor_index"]
            torch.testing.assert_close(seen["final"][index].cpu(), example["ribbon"].mask.flatten())
        applied += int(result["mask_structure_applied_anchors"].item())
        assert all(parameter.grad is None for parameter in model.teacher.parameters())
        model.fp16_scaler.unscale_(optimizer)
        for name, module in list(model.student.items()) + list(model.student_aux.items()):
            norm = module.clip_grad_norm_(3.0)
            assert torch.isfinite(norm), f"nonfinite gradient: iteration={iteration}, module={name}, norm={norm}"
        model.fp16_scaler.step(optimizer)
        model.fp16_scaler.update()
        model.update_teacher(.994)
        torch.cuda.synchronize()
        steps.append({"iteration": iteration, "seconds": time.perf_counter() - started,
                      "metrics": {name: value.item() for name, value in result.items()},
                      "records": model.mask_policy.last_records,
                      "masked_counts": seen["final"].sum(-1).tolist()})
    hook.remove()
    assert any(not torch.equal(before, actual) for before, actual in zip(initial_student, model.student.parameters()))
    assert any(not torch.equal(before, actual) for before, actual in zip(initial_decoder, model.student_aux.parameters()))
    del initial_student, initial_decoder
    total_applied = torch.tensor(applied, device="cuda")
    dist.all_reduce(total_applied)
    assert total_applied.item() > 0, "active structure stage did not apply any ribbon"
    check_decoder_sync(model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpointer = GeoTopoCheckpointer(model, str(args.output_dir), optimizer=optimizer)
    checkpointer.save("structure_smoke", iteration=12000)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    before_optimizer = {parameter: {key: value.detach().clone() for key, value in state.items() if isinstance(value, torch.Tensor)}
                        for parameter, state in optimizer.state.items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(.2)
        for state in optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    value.zero_()
    state = checkpointer.resume_or_load("", resume=True)
    assert state["iteration"] == 12000
    for expected, actual in zip(before, model.parameters()):
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)
    assert len(before_optimizer) == len(optimizer.state)
    for parameter, expected in before_optimizer.items():
        for key, value in expected.items():
            torch.testing.assert_close(value, optimizer.state[parameter][key], atol=0, rtol=0)
    assert model.mask_policy.settings.warmup_iterations == 6000
    check_decoder_sync(model)
    (args.output_dir / f"verification.rank_{rank}.json").write_text(json.dumps({
        "passed": True, "scope": "real CUDA/xFormers/two-rank FSDP ViT-small on synthetic images",
        "rank": rank, "world_size": dist.get_world_size(), "smoke_initial_grad_scale": 128.0,
        "steps": steps, "total_applied_anchors_all_ranks": int(total_applied.item()),
        "student_updated": True, "decoder_updated": True, "full_model_and_optimizer_restored": True,
        "original_weight_files_modified": False,
    }, indent=2) + "\n")
    if distributed.is_main_process():
        print("PASS: real CUDA/FSDP structure masks, unchanged budgets, GCVD/iBOT/pixel backward, EMA and resume")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
