"""Real replicated DDP qualification with independent reduced-gradient replay."""

import argparse
import copy
from contextlib import nullcontext
from datetime import timedelta
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from _integration_common import capabilities, write_json


def equal(a, b, path="root"):
    if isinstance(a, torch.Tensor):
        if not torch.equal(a, b):
            mismatch = a != b
            indices = mismatch.flatten().nonzero().flatten()[:5]
            raise AssertionError(
                dict(
                    path=path,
                    dtype=str(a.dtype),
                    shape=list(a.shape),
                    mismatches=int(mismatch.sum()),
                    maximum_absolute_error=float((a.double() - b.double()).abs().max()),
                    actual=a.flatten()[indices].tolist(),
                    expected=b.flatten()[indices].tolist(),
                )
            )
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            equal(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for index, (x, y) in enumerate(zip(a, b)):
            equal(x, y, f"{path}[{index}]")
    else:
        assert a == b, (path, a, b)


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu(v) for v in value)
    return value


class ProbeNetwork(torch.nn.Module):
    def __init__(self, channels_last):
        super().__init__()
        self.aux = torch.nn.Parameter(torch.ones(()))
        self.layer = (
            torch.nn.Conv2d(64, 64, 3, padding=1) if channels_last else torch.nn.Linear(64, 128)
        )

    def forward(self, x):
        return self.layer(x), self.aux


def gradient_replay_copy(gradient):
    """Preserve vector-reduction alignment as well as DDP bucket gradient values.

    A plain clone aligns bucket views differently, so foreach_norm can accumulate
    the same coordinates in a different order and shift AdaGC history by an ULP.
    This is an oracle allocation, with at most 256 extra bytes per gradient.
    """
    alignment = 256
    span = 1 + sum((size - 1) * stride for size, stride in zip(gradient.shape, gradient.stride()))
    padding = alignment // gradient.element_size()
    storage = torch.empty(span + padding, dtype=gradient.dtype, device=gradient.device)
    offset = ((gradient.data_ptr() - storage.data_ptr()) % alignment) // gradient.element_size()
    result = storage.as_strided(gradient.shape, gradient.stride(), offset)
    result.copy_(gradient)
    return result


def global_mean_loss_weight(local_count, global_count, world_size):
    """Undo DDP's rank averaging for a local mean over uneven sample counts."""
    if any(
        isinstance(v, bool) or not isinstance(v, int)
        for v in (local_count, global_count, world_size)
    ):
        raise ValueError("sample counts and world size must be positive integers")
    if local_count <= 0 or global_count < local_count or world_size <= 0:
        raise ValueError("Invalid sample counts or world size; exhausted ranks cannot continue")
    return world_size * local_count / global_count


def validate_scaler_states(states):
    """Public GradScaler dictionaries include growth tracking, not just scale."""
    if not states or any(state != states[0] for state in states[1:]):
        raise RuntimeError("Mismatched public AMP scaler state; refusing backward")


def assert_scaler_agreement(scaler):
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, scaler.state_dict())
    validate_scaler_states(states)


def validate_input_availability(flags):
    """Stop together at end of input; refuse partial participation before forward."""
    if not flags:
        raise ValueError("Expected an input-availability flag for each rank")
    if any(bool(flag) != bool(flags[0]) for flag in flags[1:]):
        raise RuntimeError("A rank exhausted input; refusing unequal update boundaries")
    return bool(flags[0])


def run_participation_oracle(local, world):
    """Actually exhaust one rank's iterator and prove every replica stops unchanged."""
    from lionplus8bit import LionPlus8bit

    if world < 2:
        return dict(name="exhausted_input", status="skipped", reason="requires two ranks")
    torch.manual_seed(42)
    model = DDP(torch.nn.Linear(64, 128).cuda(), device_ids=[local])
    optimizer = LionPlus8bit(model.parameters(), clipping="adagc", adagc_warmup_steps=3)
    scaler = torch.amp.GradScaler("cuda", init_scale=32)
    batches = iter([torch.ones(2, 64, device="cuda")] * (1 if dist.get_rank() == 0 else 2))
    for attempt in range(2):
        before = cpu(
            dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scaler=scaler.state_dict(),
            )
        )
        batch = next(batches, None)
        flags = [None] * world
        dist.all_gather_object(flags, batch is not None)
        caught = False
        try:
            available = validate_input_availability(flags)
        except RuntimeError as error:
            assert "exhausted input" in str(error)
            equal(
                before,
                cpu(
                    dict(
                        model=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        scaler=scaler.state_dict(),
                    )
                ),
            )
            caught = True
        stopped = [None] * world
        dist.all_gather_object(stopped, caught)
        assert stopped == [attempt == 1] * world
        if caught:
            return dict(
                name="exhausted_input",
                status="passed",
                flags=flags,
                successful_updates=1,
                all_ranks_stopped_before_forward=True,
            )
        assert available
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(batch).float().square().mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        assert all(state["step"] == 1 for state in optimizer.state.values())
    raise AssertionError("Expected mismatched input exhaustion")


def validate_finite_flags(flags):
    if not flags or any(bool(flag) != bool(flags[0]) for flag in flags[1:]):
        raise RuntimeError(
            "Rank-local nonfinite gradients after reduction; refusing divergent update"
        )
    return bool(flags[0])


def collective_gradient_finiteness(parameters):
    parameters = list(parameters)
    finite = all(bool(torch.isfinite(p.grad).all()) for p in parameters if p.grad is not None)
    flag = torch.tensor([int(finite)], device=parameters[0].device)
    flags = [torch.empty_like(flag) for _ in range(dist.get_world_size())]
    dist.all_gather(flags, flag)
    return validate_finite_flags([bool(int(value)) for value in flags])


def independent_gradient_norm(gradients):
    """FP64 sum of coordinate squares, with no replicated-rank norm reduction."""
    squares = [
        gradient.detach().double().square().sum() for gradient in gradients if gradient is not None
    ]
    return torch.stack(squares).sum().sqrt()


def run_correctness_oracles(local, world, channels_last):
    """Independent full-batch gradients, clipping equations, and unsafe-driver probes."""
    from lionplus8bit import LionPlus8bit

    rows = [run_participation_oracle(local, world)]
    counts = [rank + 2 for rank in range(world)]
    for amp, accumulation in ((False, 1), (True, 3)):
        torch.manual_seed(173)
        model = ProbeNetwork(channels_last).cuda()
        if channels_last:
            model = model.to(memory_format=torch.channels_last)
        ddp = DDP(model, device_ids=[local], gradient_as_bucket_view=True)
        reference = copy.deepcopy(model)
        scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=32)
        assert_scaler_agreement(scaler)
        optimizer = LionPlus8bit(model.parameters(), min_8bit_size=10**9, lr=0.01)
        all_inputs = []
        for rank in range(world):
            generator = torch.Generator().manual_seed(9100 + rank)
            shape = (counts[rank], 64, 8, 8) if channels_last else (counts[rank], 64)
            all_inputs.append(
                [torch.randn(shape, generator=generator).cuda() * 0.2 for _ in range(accumulation)]
            )
        weight = global_mean_loss_weight(counts[dist.get_rank()], sum(counts), world)
        for micro in range(accumulation):
            context = ddp.no_sync() if micro < accumulation - 1 else nullcontext()
            with context:
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    prediction, auxiliary = ddp(all_inputs[dist.get_rank()][micro])
                    loss = prediction.float().square().mean() + auxiliary.float() * 0.001
                    loss = loss * weight / accumulation
                scaler.scale(loss).backward()
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                batch = torch.cat([inputs[micro] for inputs in all_inputs])
                prediction, auxiliary = reference(batch)
                full_loss = prediction.float().square().mean() + auxiliary.float() * 0.001
                full_loss = full_loss / accumulation
            full_loss.backward()
        scaler.unscale_(optimizer)
        assert collective_gradient_finiteness(model.parameters())
        rtol, atol = (2e-3, 2e-5) if amp else (2e-5, 1e-7)
        errors, mismatches, margins = [], 0, []
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=rtol, atol=atol)
            errors.append(float((actual.grad - expected.grad).abs().max()))
            disagreement = actual.grad.sign() != expected.grad.sign()
            mismatches += int(disagreement.sum())
            if bool(disagreement.any()):
                margins.append(float(expected.grad[disagreement].abs().min()))
        gradients = [parameter.grad.clone() for parameter in model.parameters()]
        norm = independent_gradient_norm(gradients)
        incorrect_norm = norm.square().clone()
        dist.all_reduce(incorrect_norm)
        incorrect_norm.sqrt_()
        torch.testing.assert_close(incorrect_norm, norm * world**0.5, atol=1e-10, rtol=1e-10)
        # Force clipping to an independently calculated 0.3 factor. A mistaken
        # sqrt(world)-larger norm would fail the FP32 momentum equation below.
        optimizer.param_groups[0]["max_grad_norm"] = float(norm) * 0.3
        optimizer.step()
        momentum_errors = []
        for parameter, gradient in zip(model.parameters(), gradients):
            expected = gradient * (0.3 * (1 - optimizer.param_groups[0]["betas"][1]))
            actual = optimizer.state[parameter]["state1"]
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-9)
            momentum_errors.append(float((actual - expected).abs().max()))
        rows.append(
            dict(
                name="global_batch_and_norm_oracle",
                status="passed",
                amp=amp,
                accumulation=accumulation,
                per_rank_sample_counts=counts,
                rtol=rtol,
                atol=atol,
                gradient_max_abs_error=max(errors),
                sign_mismatches=mismatches,
                minimum_mismatched_reference_margin=min(margins) if margins else None,
                independent_gradient_norm=float(norm),
                incorrect_summed_rank_norm=float(incorrect_norm),
                momentum_max_abs_error=max(momentum_errors),
            )
        )
    if world > 1:
        bad_scaler = torch.amp.GradScaler("cuda", init_scale=32 if dist.get_rank() == 0 else 64)
        stopped = False
        try:
            assert_scaler_agreement(bad_scaler)
        except RuntimeError as error:
            assert "scaler" in str(error)
            stopped = True
        assert stopped
        rows.append(
            dict(
                name="mismatched_scaler_startup",
                status="passed",
                behavior="all ranks stopped before backward",
            )
        )
        before = copy.deepcopy(optimizer.state_dict())
        weights = [parameter.detach().clone() for parameter in model.parameters()]
        if dist.get_rank() == 0:
            next(model.parameters()).grad.view(-1)[0] = float("inf")
        stopped = False
        try:
            collective_gradient_finiteness(model.parameters())
        except RuntimeError as error:
            assert "Rank-local nonfinite" in str(error)
            stopped = True
        assert stopped
        equal(before, optimizer.state_dict())
        for parameter, value in zip(model.parameters(), weights):
            equal(parameter, value)
        rows.append(
            dict(
                name="rank_local_post_reduction_inf",
                status="passed",
                behavior="all ranks stopped before optimizer update",
            )
        )
    else:
        rows.append(
            dict(
                name="rank_local_negative_cases",
                status="skipped",
                reason="requires at least two ranks",
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite", choices=["smoke", "core", "amp", "buckets", "participation"], default="core"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=204)
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--paged", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local)
    print(f"rank {rank}: initializing NCCL", flush=True)
    dist.init_process_group(
        "nccl", timeout=timedelta(seconds=180), device_id=torch.device("cuda", local)
    )
    print(f"rank {rank}: NCCL ready", flush=True)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "environment.json", capabilities())
    dist.barrier()
    from lionplus8bit import LionPlus8bit

    if args.suite == "participation":
        oracles = [run_participation_oracle(local, world)]
    else:
        oracles = (
            []
            if args.suite == "smoke"
            else run_correctness_oracles(local, world, args.channels_last)
        )
    rows = []
    cases = [
        ("global", False, False, 0, 1),
        ("adagc", True, True, 3, 3),
        ("adagc", True, False, 100, 1),
        ("adagc", False, True, 0, 1),
    ]
    if args.suite == "smoke":
        cases = cases[:1]
    elif args.suite == "participation":
        cases = []
    for mode, amp, buckets, warmup, accum in cases:
        torch.manual_seed(52)
        model = ProbeNetwork(args.channels_last).cuda()
        if args.channels_last:
            model = model.to(memory_format=torch.channels_last)
        ddp = DDP(model, device_ids=[local], gradient_as_bucket_view=buckets)
        control = copy.deepcopy(model)
        if args.compile_model:
            ddp = torch.compile(ddp)
        options = dict(
            clipping=mode, adagc_warmup_steps=warmup, max_grad_norm=0.4, is_paged=args.paged
        )
        opt = LionPlus8bit(model.parameters(), **options)
        ref = LionPlus8bit(control.parameters(), **options)
        scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=32, growth_interval=30)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.9)
        ref_scheduler = torch.optim.lr_scheduler.StepLR(ref, step_size=50, gamma=0.9)
        successes = 0
        for step in range(args.steps):
            assert_scaler_agreement(scaler)
            opt.zero_grad(set_to_none=bool(step % 2))
            ref.zero_grad(set_to_none=True)
            for micro in range(accum):
                context = ddp.no_sync() if micro < accum - 1 else nullcontext()
                with context:
                    shape = (4, 64, 8, 8) if args.channels_last else (4, 64)
                    x = torch.randn(shape, device="cuda") + (rank * 0.01)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                        pred, aux = ddp(x)
                        coefficient = 0 if step % 11 == 0 else 1e-30 if step % 17 == 0 else 0.001
                        loss = (pred.float().square().mean() + aux.float() * coefficient) / accum
                    if amp and step in (0, 3, 102) and rank == 0 and micro == accum - 1:
                        hook = next(model.parameters()).register_hook(
                            lambda g: torch.full_like(g, float("inf"))
                        )
                        scaler.scale(loss).backward()
                        hook.remove()
                    else:
                        scaler.scale(loss).backward()
            scaler.unscale_(opt)
            before = copy.deepcopy(opt.state_dict())
            weights = [p.detach().clone() for p in model.parameters()]
            grads = [p.grad for p in model.parameters()]
            finite = collective_gradient_finiteness(model.parameters())
            oldscale = scaler.get_scale()
            if finite:
                for p, q in zip(model.parameters(), control.parameters()):
                    q.grad = None if p.grad is None else gradient_replay_copy(p.grad)
                ref.param_groups[0]["lr"] = opt.param_groups[0]["lr"]
                ref.step()
            scaler.step(opt)
            scaler.update()
            if not finite:
                equal(before, opt.state_dict())
                for p, w in zip(model.parameters(), weights):
                    equal(p, w)
                assert scaler.get_scale() < oldscale
            else:
                successes += 1
                scheduler.step()
                ref_scheduler.step()
                for p, q in zip(model.parameters(), control.parameters()):
                    equal(p, q)
                # Scheduler LR is outside the update oracle's optimizer state.
                ref.param_groups[0]["lr"] = opt.param_groups[0]["lr"]
                try:
                    equal(opt.state_dict(), ref.state_dict())
                except AssertionError:
                    write_json(
                        args.output / f"replay_failure_rank{rank}.json",
                        dict(
                            step=step,
                            mode=mode,
                            amp=amp,
                            buckets=buckets,
                            actual_clipping=opt.last_clipping,
                            reference_clipping=ref.last_clipping,
                            gradients=[
                                dict(
                                    shape=list(p.grad.shape),
                                    strides=list(p.grad.stride()),
                                    offset=p.grad.storage_offset(),
                                    alignment=p.grad.data_ptr() % 256,
                                    replay_alignment=q.grad.data_ptr() % 256,
                                )
                                for p, q in zip(model.parameters(), control.parameters())
                                if p.grad is not None
                            ],
                        ),
                    )
                    raise
            for p, g in zip(model.parameters(), grads):
                assert p.grad is g
            state = opt.state_dict()
            # Compare complete public state including uint8 codes and histories across ranks.
            gathered = [None] * world
            dist.all_gather_object(
                gathered,
                cpu(
                    {
                        "optimizer": state,
                        "scaler": scaler.state_dict(),
                        "weights": list(model.parameters()),
                    }
                ),
            )
            for other in gathered[1:]:
                equal(gathered[0], other)
        if rank == 0:
            print(mode, amp, "case passed", flush=True)
        rows.append(
            dict(
                mode=mode,
                amp=amp,
                bucket_views=buckets,
                warmup=warmup,
                accumulation=accum,
                attempts=args.steps,
                successful=successes,
                status="passed",
            )
        )
    write_json(
        args.output / f"rank{rank}.json",
        dict(rank=rank, local_device=local, world_size=world, cases=rows, oracles=oracles),
    )
    dist.barrier()
    if rank == 0:
        write_json(
            args.output / "result.json",
            dict(status="passed", world_size=world, cases=rows, oracles=oracles),
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
