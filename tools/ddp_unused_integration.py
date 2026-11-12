"""Standalone find_unused_parameters=True qualification; run with torchrun.

Every potential parameter is trainable at DDP construction. A branch is used
only on rank 0, a second branch first activates globally after AdaGC warmup,
and a permanently dormant parameter must retain None gradients/unseeded history.
"""

import argparse
import copy
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from _integration_common import capabilities
from ddp_integration import (
    assert_scaler_agreement,
    collective_gradient_finiteness,
    cpu,
    equal,
    gradient_replay_copy,
)


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


class UnusedBranchProbe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dormant = torch.nn.Parameter(torch.ones(4097))
        self.common = torch.nn.Linear(64, 128)
        self.local = torch.nn.Linear(64, 128)
        self.shared = torch.nn.Linear(64, 128)

    def forward(self, inputs, *, local_branch, shared_branch):
        value = self.common(inputs)
        if local_branch:
            value = value + self.local(inputs) * 0.3
        if shared_branch:
            value = value + self.shared(inputs) * 0.2
        return value


def branch_schedule(step, rank):
    return dict(
        local_branch=rank == 0 and step % 4 != 2, shared_branch=step >= 5 and (step - 5) % 3 != 2
    )


def _all_equal(value, world):
    gathered = [None] * world
    dist.all_gather_object(gathered, cpu(value))
    for peer in gathered[1:]:
        equal(gathered[0], peer)


def _case(local, rank, world, accumulation, steps):
    from lionplus8bit import LionPlus8bit

    torch.manual_seed(711)
    model = UnusedBranchProbe().cuda()
    assert all(parameter.requires_grad for parameter in model.parameters())
    ddp = DDP(model, device_ids=[local], find_unused_parameters=True, gradient_as_bucket_view=True)
    reference = copy.deepcopy(model)
    options = dict(clipping="adagc", adagc_warmup_steps=3, max_grad_norm=0.4)
    optimizer = LionPlus8bit(model.parameters(), **options)
    replay = LionPlus8bit(reference.parameters(), **options)
    scaler = torch.amp.GradScaler("cuda", init_scale=32, growth_interval=4)
    slots = {name: index for index, (name, _) in enumerate(model.named_parameters())}
    parameters = dict(model.named_parameters())
    reference_parameters = dict(reference.named_parameters())
    rows = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        replay.zero_grad(set_to_none=True)
        assert_scaler_agreement(scaler)
        local_choices = branch_schedule(step, rank)
        global_local = any(branch_schedule(step, peer)["local_branch"] for peer in range(world))
        global_shared = branch_schedule(step, 0)["shared_branch"]
        expected_active = {
            name: name.startswith("common.")
            or (name.startswith("local.") and global_local)
            or (name.startswith("shared.") and global_shared)
            for name in parameters
        }
        inactive = {}
        for name, parameter in parameters.items():
            if not expected_active[name]:
                clipping = optimizer._clipping_state
                history = (
                    clipping.history[slots[name]].clone()
                    if clipping is not None
                    else parameter.new_zeros(())
                )
                initialized = (
                    clipping.initialized[slots[name]].clone()
                    if clipping is not None
                    else torch.zeros((), dtype=torch.bool, device=parameter.device)
                )
                inactive[name] = (
                    copy.deepcopy(optimizer.state.get(parameter, {})),
                    history,
                    initialized,
                )
        for micro in range(accumulation):
            context = ddp.no_sync() if micro < accumulation - 1 else nullcontext()
            with context:
                inputs = torch.randn(4, 64, device="cuda") * 0.2 + rank * 0.01
                with torch.autocast("cuda", dtype=torch.float16):
                    prediction = ddp(inputs, **local_choices)
                    loss = prediction.float().square().mean() / accumulation
                scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        assert collective_gradient_finiteness(model.parameters())
        availability = {name: parameter.grad is not None for name, parameter in parameters.items()}
        assert availability == expected_active, (step, rank, availability, expected_active)
        _all_equal({name: parameter.grad for name, parameter in parameters.items()}, world)
        gradients = {name: parameter.grad for name, parameter in parameters.items()}
        gradient_values = {
            name: None if gradient is None else gradient.detach().clone()
            for name, gradient in gradients.items()
        }
        for name, parameter in parameters.items():
            reference_parameters[name].grad = (
                None if parameter.grad is None else gradient_replay_copy(parameter.grad)
            )
        bootstrap = {}
        if step == 5:
            assert optimizer._clipping_state.steps > optimizer.clipping_config.warmup
            for name, parameter in parameters.items():
                if name.startswith("shared."):
                    assert not optimizer.state.get(parameter)
                    assert not bool(optimizer._clipping_state.initialized[slots[name]])
                    bootstrap[name] = float(
                        torch.linalg.vector_norm(parameter.grad, dtype=torch.float64)
                    )
        replay.step()
        scaler.step(optimizer)
        scaler.update()
        equal(optimizer.state_dict(), replay.state_dict())
        for name, parameter in parameters.items():
            equal(parameter, reference_parameters[name])
            assert parameter.grad is gradients[name]
            if parameter.grad is not None:
                equal(parameter.grad, gradient_values[name])
        assert optimizer._clipping_state.steps == step + 1
        for name, (native_state, history, initialized) in inactive.items():
            parameter = parameters[name]
            equal(optimizer.state.get(parameter, {}), native_state)
            equal(optimizer._clipping_state.history[slots[name]], history)
            equal(optimizer._clipping_state.initialized[slots[name]], initialized)
        assert parameters["dormant"].grad is None
        assert not optimizer.state.get(parameters["dormant"])
        assert not bool(optimizer._clipping_state.initialized[slots["dormant"]])
        active_names = [name for name in parameters if expected_active[name]]
        bootstrap_scales = {}
        for name, norm in bootstrap.items():
            scale = optimizer.last_clipping["scales"][active_names.index(name)]
            expected_scale = min(1.0, options["max_grad_norm"] / norm) if norm else 1.0
            assert math.isclose(scale, expected_scale, rel_tol=2e-6, abs_tol=1e-8)
            assert bool(optimizer._clipping_state.initialized[slots[name]])
            assert optimizer.state[parameters[name]]["step"] == 1
            bootstrap_scales[name] = dict(norm=norm, scale=scale, expected_scale=expected_scale)
        for name, parameter in parameters.items():
            if expected_active[name]:
                assert optimizer.state[parameter]["state1"].dtype == (
                    torch.uint8 if name.endswith("weight") else torch.float32
                )
        _all_equal(
            dict(
                optimizer=optimizer.state_dict(),
                scaler=scaler.state_dict(),
                weights=list(model.parameters()),
                availability=availability,
            ),
            world,
        )
        rows.append(
            dict(
                step=step,
                local_forward=local_choices,
                synchronized_gradient_availability=availability,
                late_bootstrap=bootstrap_scales,
                status="passed",
            )
        )
    for prefix in ("local.", "shared."):
        expected_steps = sum(
            any(branch_schedule(step, peer)["local_branch"] for peer in range(world))
            if prefix == "local."
            else branch_schedule(step, 0)["shared_branch"]
            for step in range(steps)
        )
        for name, parameter in parameters.items():
            if name.startswith(prefix):
                assert optimizer.state[parameter]["step"] == expected_steps
    return dict(
        status="passed",
        amp="fp16",
        accumulation=accumulation,
        find_unused_parameters=True,
        bucket_views=True,
        attempts=steps,
        successful=steps,
        warmup_steps=3,
        universally_dormant_history="unseeded",
        late_shared_activation_step=5,
        steps=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--accumulation", type=int, nargs="+", default=[1, 3])
    args = parser.parse_args()
    if args.steps < 8 or min(args.accumulation) < 1:
        parser.error("Require at least 8 attempts and positive accumulation")
    rank, local, world = (int(os.environ[key]) for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if world < 2:
        raise RuntimeError("This unused-rank qualification requires at least two CUDA ranks")
    torch.cuda.set_device(local)
    dist.init_process_group(
        "nccl", timeout=timedelta(seconds=180), device_id=torch.device("cuda", local)
    )
    try:
        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=False)
            evidence = capabilities()
            evidence["worker_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            write_json(args.output / "environment.json", evidence)
        dist.barrier()
        cases = [
            _case(local, rank, world, accumulation, args.steps)
            for accumulation in args.accumulation
        ]
        write_json(
            args.output / f"rank{rank}.json",
            dict(rank=rank, local_device=local, world_size=world, cases=cases),
        )
        dist.barrier()
        if rank == 0:
            write_json(
                args.output / "result.json",
                dict(
                    status="passed",
                    world_size=world,
                    cases=cases,
                    distributed_collectives="runner only",
                ),
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
