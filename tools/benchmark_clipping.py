"""Matched CUDA gradient replay; results include host synchronization costs."""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import bitsandbytes as bnb
import torch

from lionplus8bit import LionPlus8bit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=["global", "adagc"], required=True)
    parser.add_argument("--layout", choices=["many-small", "few-large"], required=True)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument(
        "--pattern", choices=["none", "localized", "widespread"], default="localized"
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 100:
        parser.error("steps must be positive and timing warmup >=100")
    if not torch.cuda.is_available():
        parser.error("CUDA required; no CPU fallback")
    if args.dtype == "bfloat16" and torch.cuda.get_device_capability()[0] < 8:
        parser.error("native BF16 hardware required for BF16 performance measurement")
    if args.output.exists():
        parser.error("output already exists")
    torch.manual_seed(args.seed)
    count, size = (256, 4096) if args.layout == "many-small" else (4, 262144)
    dtype = getattr(torch, args.dtype)
    params = [
        torch.nn.Parameter(torch.randn(size, device="cuda", dtype=dtype)) for _ in range(count)
    ]
    gradients = [torch.randn_like(p).mul_(1e-4) for p in params]
    opt = LionPlus8bit(params, clipping=args.strategy, max_grad_norm=1)
    wall, gpu = [], []
    first_peak = None
    for step in range(args.warmup + args.steps):
        for p, g in zip(params, gradients):
            p.grad = g
        spike = step % 23 == 0 and args.pattern != "none"
        indices = range(count) if args.pattern == "widespread" else [step % count]
        if spike:
            for i in indices:
                gradients[i].mul_(100)
        torch.cuda.synchronize()
        if step == args.warmup:
            first_peak = torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin = time.perf_counter()
        start.record()
        opt.step()
        end.record()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - begin
        if step >= args.warmup:
            wall.append(elapsed * 1000)
            gpu.append(start.elapsed_time(end))
        if spike:
            for i in indices:
                gradients[i].div_(100)
    result = dict(
        arguments=vars(args) | {"output": str(args.output)},
        environment=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            bitsandbytes=bnb.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            capability=torch.cuda.get_device_capability(),
        ),
        parameter_count=count,
        elements=count * size,
        wall_ms=wall,
        cuda_ms=gpu,
        median_wall_ms=statistics.median(wall),
        p95_wall_ms=sorted(wall)[int(0.95 * (len(wall) - 1))],
        median_cuda_ms=statistics.median(gpu),
        warmup_peak_allocated=first_peak,
        peak_allocated=torch.cuda.max_memory_allocated(),
        peak_reserved=torch.cuda.max_memory_reserved(),
    )
    if args.profile:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            opt.step()
            torch.cuda.synchronize()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(args.output.with_suffix(".trace.json")))
        result["synchronization_events"] = {
            e.key: e.count
            for e in prof.key_averages()
            if "Synchroniz" in e.key or "_local_scalar_dense" in e.key
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("wall_ms", "cuda_ms", "arguments")}
        )
    )


if __name__ == "__main__":
    main()
