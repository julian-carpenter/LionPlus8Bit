"""Matched CUDA replay or instrumented AMP training, including host synchronization."""

import argparse
from collections import defaultdict
import json
import platform
import statistics
import time
from pathlib import Path

import bitsandbytes as bnb
import torch

from lionplus8bit import LionPlus8bit


CASES = ("ordinary", "zero", "tiny", "recovery", "strided", "variable-active")


def strided_buffer(p):
    return torch.empty((p.numel(), 2), device=p.device, dtype=p.dtype)[:, 0].view_as(p)


def condition_gradients(params, case, step, warmup, pattern, strided=None):
    """Apply finite stress inputs after unscale, outside optimizer timing."""
    condition = case
    if case == "recovery":
        condition = "ordinary" if step < warmup else "tiny" if step == warmup else "recovery"
    if case == "variable-active":
        for i, p in enumerate(params):
            if (i + step) % 2:
                p.grad = None
    if condition == "zero":
        params[-1].grad.zero_()
    elif condition == "tiny":
        params[-1].grad.view(-1)[0] = 1e-30
    elif condition == "strided" and params[-1].grad.is_contiguous():
        strided.copy_(params[-1].grad)
        params[-1].grad = strided
    scheduled = step % 23 == 0 and pattern != "none"
    indices = range(len(params)) if pattern == "widespread" else [step % len(params)]
    spiked = 0
    if scheduled:
        for i in indices:
            if params[i].grad is not None:
                params[i].grad.mul_(100)
                spiked += 1
    return dict(condition=condition, spike=bool(spiked), spiked_tensors=spiked)


class ReplayInputs:
    """Reuse replay storage and restore exact base gradients before every update."""

    def __init__(self, params, case):
        self.params, self.case = params, case
        self.base = [torch.randn_like(p).mul_(1e-4) for p in params]
        self.gradients = [torch.empty_like(p) for p in params]
        if case == "strided":
            self.gradients[-1] = strided_buffer(params[-1])

    def prepare(self, step, warmup, pattern):
        for p, g, base in zip(self.params, self.gradients, self.base):
            g.copy_(base)
            p.grad = g
        return condition_gradients(self.params, self.case, step, warmup, pattern)


def timing_summary(samples):
    wall = [s["wall_ms"] for s in samples]
    gpu = [s["cuda_ms"] for s in samples]
    return dict(
        count=len(samples),
        median_wall_ms=statistics.median(wall),
        p95_wall_ms=sorted(wall)[int(0.95 * (len(wall) - 1))],
        median_cuda_ms=statistics.median(gpu),
    )


def conditional_summary(samples):
    groups = defaultdict(list)
    for row in samples:
        groups["spike" if row["spike"] else "nonspike"].append(row)
        groups[f"condition:{row['condition']}"].append(row)
        paths = row.get("paths") or {}
        for name, field in (
            ("strict", "strict_tensors"),
            ("tiny", "tiny_tensors"),
            ("norm_recovery", "norm_recovery_tensors"),
            ("clipped", "clipped_tensors"),
        ):
            if paths.get(field, 0):
                groups[f"path:{name}"].append(row)
    return {key: timing_summary(rows) for key, rows in groups.items()}


def storage_bytes(values):
    """Count allocated tensor storage once, including shared native codebooks."""
    storages = {}

    def collect(value):
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            key = (value.device, storage.data_ptr())
            storages[key] = storage.nbytes()
        elif isinstance(value, dict):
            for child in value.values():
                collect(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                collect(child)

    collect(values)
    return sum(storages.values())


class CudaTimer:
    """Events are allocated once and reused after each synchronized measurement."""

    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def measure(self, function):
        begin = time.perf_counter()
        self.start.record()
        value = function()
        self.end.record()
        self.end.synchronize()
        return value, dict(
            wall_ms=(time.perf_counter() - begin) * 1000,
            cuda_ms=self.start.elapsed_time(self.end),
        )


def run_benchmark(args, repetition):
    torch.cuda.current_stream().synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(args.seed)
    count, size = (256, 4096) if args.layout == "many-small" else (4, 262144)
    dtype = torch.float32 if args.mode == "amp" else getattr(torch, args.dtype)
    width = int(size**0.5)
    shape = (width, width) if args.mode == "amp" else (size,)
    params = [
        torch.nn.Parameter(torch.randn(shape, device="cuda", dtype=dtype)) for _ in range(count)
    ]
    opt = (
        bnb.optim.Lion8bit(params)
        if args.strategy == "native"
        else LionPlus8bit(params, clipping=args.strategy, max_grad_norm=1)
    )
    calls = []
    opt.register_step_post_hook(lambda *unused: calls.append(1))
    replay = ReplayInputs(params, args.case) if args.mode == "replay" else None
    scaler = torch.amp.GradScaler("cuda", init_scale=32) if args.mode == "amp" else None
    inputs = torch.randn(args.batch_size, width, device="cuda") if scaler else None
    strided = strided_buffer(params[-1]) if args.case == "strided" and scaler else None
    timer, iteration_timer = CudaTimer(), CudaTimer()
    samples = []
    warmup_peak = baseline = None

    def train():
        opt.zero_grad(set_to_none=True)
        # Independent linear projections keep the same tensor sizes as the replay.
        with torch.autocast("cuda", dtype=getattr(torch, args.dtype)):
            losses = [torch.nn.functional.linear(inputs, p).float().square().mean() for p in params]
            loss = torch.stack(losses).mean()
        scaler.scale(loss).backward()

    def one_step(step):
        phases = {}
        before = len(calls)
        if replay is not None:
            metadata = replay.prepare(step, args.warmup, args.pattern)
            torch.cuda.current_stream().synchronize()
            _, elapsed = timer.measure(opt.step)
        else:
            metadata = {}

            def iteration():
                _, phases["forward_backward"] = timer.measure(train)
                _, phases["unscale"] = timer.measure(lambda: scaler.unscale_(opt))
                value, phases["gradient_conditioning"] = timer.measure(
                    lambda: condition_gradients(
                        params, args.case, step, args.warmup, args.pattern, strided
                    )
                )
                metadata.update(value)
                _, phases["optimizer"] = timer.measure(lambda: scaler.step(opt))
                _, phases["scaler_update"] = timer.measure(scaler.update)

            _, elapsed = iteration_timer.measure(iteration)
        success = len(calls) > before
        paths = (
            dict(opt.last_step_stats)
            if success and args.strategy != "native"
            else dict(
                active_tensors=sum(p.grad is not None for p in params),
                clipped_tensors=0,
                strict_tensors=0,
                tiny_tensors=0,
                norm_recovery_tensors=0,
            )
            if success
            else None
        )
        return dict(
            repetition=repetition,
            step=step,
            successful=success,
            **elapsed,
            **metadata,
            paths=paths,
            phases=phases,
        )

    for step in range(args.warmup + args.steps):
        if step == args.warmup:
            torch.cuda.current_stream().synchronize()
            warmup_peak = torch.cuda.max_memory_allocated()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        row = one_step(step)
        if step >= args.warmup:
            samples.append(row)
    peak = torch.cuda.max_memory_allocated()
    native_values = [value for state in opt.state.values() for value in state.values()]
    clipping = getattr(opt, "_clipping_state", None)
    metadata_values = list(vars(clipping).values()) if clipping is not None else []
    memory = dict(
        warmup_peak_allocated=warmup_peak,
        baseline_allocated=baseline,
        peak_allocated=peak,
        transient_peak_delta=peak - baseline,
        peak_reserved=torch.cuda.max_memory_reserved(),
        native_state_bytes=storage_bytes(native_values),
        clipping_metadata_bytes=storage_bytes(metadata_values),
        optimizer_state_bytes=storage_bytes(native_values + metadata_values),
    )
    profile = None
    if args.profile and repetition == 0:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            one_step(args.warmup + args.steps)
            torch.cuda.current_stream().synchronize()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(args.output.with_suffix(".trace.json")))
        profile = {
            e.key: e.count
            for e in prof.key_averages()
            if "Synchroniz" in e.key or "_local_scalar_dense" in e.key
        }
    return dict(samples=samples, memory=memory, synchronization_events=profile)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=["global", "adagc", "native"], required=True)
    parser.add_argument("--layout", choices=["many-small", "few-large"], required=True)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--mode", choices=["replay", "amp"], default="replay")
    parser.add_argument("--case", choices=CASES, default="ordinary")
    parser.add_argument("--stream", choices=["default", "nondefault"], default="default")
    parser.add_argument(
        "--pattern", choices=["none", "localized", "widespread"], default="localized"
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=150)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    args.dtype = args.dtype or ("float16" if args.mode == "amp" else "float32")
    if min(args.steps, args.repeats, args.batch_size) < 1 or args.warmup < 100:
        parser.error("steps, repeats and batch-size must be positive; timing warmup must be >=100")
    if args.mode == "amp" and args.dtype == "float32":
        parser.error("AMP requires float16 or bfloat16 autocast with FP32 parameters")
    if args.mode == "replay" and args.dtype == "float16" and args.case in ("tiny", "recovery"):
        parser.error("FP16 gradients cannot store 1e-30; use --mode amp for FP32 gradient storage")
    if not torch.cuda.is_available():
        parser.error("CUDA required; no CPU fallback")
    if args.dtype == "bfloat16" and torch.cuda.get_device_capability()[0] < 8:
        parser.error("native BF16 hardware required for BF16 performance measurement")
    if args.output.exists():
        parser.error("output already exists")
    if args.output.with_suffix(".trace.json").exists() and args.profile:
        parser.error("trace output already exists")
    torch.set_num_threads(1)
    stream = torch.cuda.Stream() if args.stream == "nondefault" else torch.cuda.current_stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        runs = [run_benchmark(args, i) for i in range(args.repeats)]
    samples = [sample for run in runs for sample in run["samples"]]
    medians = [timing_summary(run["samples"])["median_wall_ms"] for run in runs]
    phase_samples = defaultdict(list)
    for row in samples:
        for name, values in row["phases"].items():
            phase_samples[name].append(dict(row, **values))
    result = dict(
        measurement_version=2,
        arguments=vars(args) | {"output": str(args.output)},
        environment=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            bitsandbytes=bnb.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            capability=torch.cuda.get_device_capability(),
        ),
        parameter_count=256 if args.layout == "many-small" else 4,
        elements=1048576,
        timed_region="optimizer" if args.mode == "replay" else "instrumented AMP iteration",
        timing_notes="Host-inclusive; reused events; AMP phase boundaries synchronize explicitly.",
        native_control="Unclipped bitsandbytes Lion8bit dispatch baseline"
        if args.strategy == "native"
        else None,
        samples=samples,
        wall_ms=[row["wall_ms"] for row in samples],
        cuda_ms=[row["cuda_ms"] for row in samples],
        **timing_summary(samples),
        conditional=conditional_summary(samples),
        phases={
            name: dict(summary=timing_summary(rows), conditional=conditional_summary(rows))
            for name, rows in phase_samples.items()
        },
        repetition_medians_wall_ms=medians,
        repetition_stdev_wall_ms=statistics.stdev(medians) if len(medians) > 1 else None,
        memory_runs=[run["memory"] for run in runs],
        **{key: max(run["memory"][key] for run in runs) for key in runs[0]["memory"]},
        synchronization_events=runs[0]["synchronization_events"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("samples", "wall_ms", "cuda_ms", "arguments")
            }
        )
    )


if __name__ == "__main__":
    main()
