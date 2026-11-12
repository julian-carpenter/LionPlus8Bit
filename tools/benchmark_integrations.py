"""Fresh-process DDP/AMP/AdaGC campaigns; profiling never contributes timings.

Defaults: 3 rotated repetitions x 1/2/4 ranks x contiguous/channels-last,
100 warmup + 300 timed updates. Add --baseline-root for paired revision bounds,
for contiguous layouts, and --scaling strong weak for both batch-scaling conventions.
Channels-last measurements describe new layout costs without an old-layout baseline. Without a
baseline the results are descriptive, with no regression or speedup verdict.
"""

import argparse
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def build_schedule(
    world_sizes, layouts, scaling_modes, repetitions, *, candidate_root, baseline_root=None
):
    conditions = list(itertools.product(world_sizes, layouts, scaling_modes))
    rows = []
    for repetition in range(repetitions):
        offset = repetition % len(conditions)
        rotated = conditions[offset:] + conditions[:offset]
        for index, (world, layout, scaling) in enumerate(rotated):
            revisions = [("candidate", candidate_root)]
            if baseline_root is not None and layout == "contiguous":
                revisions.insert(0, ("baseline", baseline_root))
                if (repetition + index) % 2:
                    revisions.reverse()
            for revision, root in revisions:
                rows.append(
                    dict(
                        repetition=repetition,
                        world_size=world,
                        layout=layout,
                        scaling=scaling,
                        revision=revision,
                        source_root=str(root),
                    )
                )
    return rows


def paired_slowdown_verdict(baseline, candidate, *, warmup, steps):
    if len(baseline) != len(candidate) or len(baseline) < 3 or warmup < 100 or steps < 300:
        return dict(
            status="inconclusive",
            reason="Requires 3 paired repetitions, 100 warmup and 300 timed updates",
        )
    ratios = [math.log(new / old) for old, new in zip(baseline, candidate)]
    # One-sided t(.95, df=2), conservative for any n>=3. Normality of paired
    # log-ratios is an explicit assumption; no GPU result is inferred from tests.
    upper = (
        math.exp(
            statistics.mean(ratios)
            + 2.919985580355516 * statistics.stdev(ratios) / math.sqrt(len(ratios))
        )
        - 1
    )
    return dict(
        status="passed" if upper <= 0.1 else "inconclusive",
        upper_95_slowdown=upper,
        margin=0.1,
        repetitions=len(ratios),
        estimate=math.exp(statistics.mean(ratios)) - 1,
        interval_method="one-sided paired log-ratio t bound, df=2 conservative for n>=3",
    )


def _source_hashes():
    import lionplus8bit
    import bitsandbytes as bnb

    package = Path(lionplus8bit.__file__).resolve().parent
    expected = os.environ.get("LIONPLUS_BENCH_SOURCE_ROOT")
    if expected and package != (Path(expected) / "src/lionplus8bit").resolve():
        raise RuntimeError(
            f"Imported lionplus8bit from {package}, expected {expected}/src/lionplus8bit"
        )
    return dict(
        package_path=str(package),
        bitsandbytes=bnb.__version__,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        source_hashes={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in package.glob("*.py")
        },
    )


def _worker(args):
    local, rank, world = (int(os.environ[key]) for key in ("LOCAL_RANK", "RANK", "WORLD_SIZE"))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180), device_id=device)
    from ddp_integration import ProbeNetwork
    from lionplus8bit import LionPlus8bit

    torch.manual_seed(52)
    channels_last = args.worker_layout == "channels-last"
    model = ProbeNetwork(channels_last=True).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    ddp = DDP(model, device_ids=[local], gradient_as_bucket_view=True)
    optimizer = LionPlus8bit(model.parameters(), clipping="adagc", max_grad_norm=0.4)
    scaler = torch.amp.GradScaler("cuda", init_scale=32)
    if args.worker_scaling == "strong":
        if args.global_batch % world:
            raise ValueError("Fixed global batch must divide evenly by world size")
        local_batch = args.global_batch // world
    else:
        local_batch = args.per_rank_batch
    generator = torch.Generator(device=device).manual_seed(8000 + rank)
    inputs = torch.randn((local_batch, 64, 8, 8), device=device, generator=generator) * 0.2
    if channels_last:
        inputs = inputs.contiguous(memory_format=torch.channels_last)

    def update(phase):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            prediction, auxiliary = ddp(inputs)
            coefficient = 1e-35 if phase == "tiny_auxiliary" else 0.001
            loss = prediction.float().square().mean() + auxiliary.float() * coefficient
            if phase == "zero":
                loss = loss * 0
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    for _ in range(args.warmup):
        update("ordinary")
    torch.cuda.synchronize(device)
    dist.barrier()
    torch.cuda.reset_peak_memory_stats(device)
    phases = ("ordinary", "zero", "tiny_auxiliary", "recovery")
    phase_counts = [
        args.steps // len(phases) + int(index < args.steps % len(phases))
        for index in range(len(phases))
    ]
    profile = (
        torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        )
        if args.worker_profile
        else nullcontext()
    )
    rows = []
    with profile as profiler:
        for phase, count in zip(phases, phase_counts):
            if not count:
                continue
            dist.barrier()
            torch.cuda.synchronize(device)
            before = min(state["step"] for state in optimizer.state.values())
            start = time.perf_counter()
            for _ in range(count):
                update(phase)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            successful = min(state["step"] for state in optimizer.state.values()) - before
            if successful != count:
                raise RuntimeError(
                    "AMP skipped an update in a benchmark condition; refusing a matched-update timing"
                )
            # These allocations and collectives occur strictly after timed work.
            duration = torch.tensor(elapsed, dtype=torch.float64, device=device)
            dist.all_reduce(duration, op=dist.ReduceOp.MAX)
            slowest = float(duration)
            rows.append(
                dict(
                    phase=phase,
                    attempted_updates=count,
                    successful_updates=successful,
                    local_samples=local_batch * count,
                    aggregate_samples=local_batch * world * count,
                    slowest_rank_seconds=slowest,
                    slowest_rank_ms_per_step=slowest * 1000 / count,
                    aggregate_samples_per_second=local_batch * world * count / slowest,
                )
            )
    if args.worker_profile:
        profiler.export_chrome_trace(str(args.output / f"profile_rank{rank}.json"))
    local_record = dict(
        rank=rank,
        local_device=local,
        world_size=world,
        local_batch=local_batch,
        phases=rows,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        native_managed_bytes=0,
        managed_storage_requested=False,
    )
    write_json(args.output / f"rank{rank}.json", local_record)
    gathered = [None] * world
    dist.all_gather_object(gathered, local_record)
    if rank == 0:
        total_seconds = sum(row["slowest_rank_seconds"] for row in rows)
        write_json(
            args.output / "result.json",
            dict(
                status="profiled" if args.worker_profile else "measured",
                warmup_updates=args.warmup,
                timed_updates=args.steps,
                slowest_rank_ms_per_step=total_seconds * 1000 / args.steps,
                aggregate_samples=sum(row["aggregate_samples"] for row in rows),
                aggregate_samples_per_second=sum(row["aggregate_samples"] for row in rows)
                / total_seconds,
                ranks=gathered,
                phases=rows,
                sources=_source_hashes(),
                timing="wall clock between device synchronizations; maximum rank duration per segment",
                diagnostics_inside_timing=False,
                workload="synthetic repeated minibatch; tiny segment changes FP32 auxiliary gradient",
                quality="not measured by this throughput campaign",
                communication_breakdown="separate profiler trace only"
                if args.worker_profile
                else "not profiled",
            ),
        )
    dist.destroy_process_group()


def _launch(args, row, index, *, profile=False):
    name = f"{index:03d}-{row['revision']}-r{row['repetition']}-w{row['world_size']}-{row['layout']}-{row['scaling']}"
    if profile:
        name += "-profile"
    output = args.output / name
    output.mkdir()
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={row['world_size']}",
        str(Path(__file__).resolve()),
        "--worker",
        "--output",
        str(output),
        "--worker-layout",
        row["layout"],
        "--worker-scaling",
        row["scaling"],
        "--global-batch",
        str(args.global_batch),
        "--per-rank-batch",
        str(args.per_rank_batch),
        "--warmup",
        str(min(args.warmup, 3) if profile else args.warmup),
        "--steps",
        str(min(args.steps, 12) if profile else args.steps),
    ]
    if profile:
        command.append("--worker-profile")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        str(Path(row["source_root"]) / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    )
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment["LIONPLUS_BENCH_SOURCE_ROOT"] = str(Path(row["source_root"]).resolve())
    write_json(
        output / "command.json",
        dict(
            command=command,
            condition=row,
            pythonpath=environment["PYTHONPATH"],
            transport={key: environment[key] for key in environment if key.startswith("NCCL_")},
        ),
    )
    with (output / "process.log").open("x") as log:
        process = subprocess.Popen(
            command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            returncode = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            write_json(output / "failure.json", dict(status="timeout", seconds=args.timeout))
            raise RuntimeError(f"Campaign child timed out: {output}") from None
    if returncode:
        write_json(output / "failure.json", dict(status="failed", returncode=returncode))
        raise RuntimeError(f"Campaign child failed: {output}")
    result = json.loads((output / "result.json").read_text())
    return dict(condition=row, output=str(output), measurements=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=["contiguous", "channels-last"],
        default=["contiguous", "channels-last"],
    )
    parser.add_argument("--scaling", nargs="+", choices=["strong", "weak"], default=["strong"])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--global-batch", type=int, default=64)
    parser.add_argument("--per-rank-batch", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--candidate-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--profile", action="store_true", help="Separate fresh profiling processes after timing"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-layout", choices=["contiguous", "channels-last"], help=argparse.SUPPRESS
    )
    parser.add_argument("--worker-scaling", choices=["strong", "weak"], help=argparse.SUPPRESS)
    parser.add_argument("--worker-profile", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        min(
            args.world_sizes
            + [
                args.repetitions,
                args.warmup,
                args.steps,
                args.global_batch,
                args.per_rank_batch,
                args.timeout,
            ]
        )
        <= 0
    ):
        parser.error("All counts and timeouts must be positive")
    if args.worker:
        _worker(args)
        return
    for source_root in (args.candidate_root, args.baseline_root):
        if source_root is not None and not (source_root / "src/lionplus8bit/__init__.py").is_file():
            parser.error(f"Source tree has no src/lionplus8bit package: {source_root}")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = build_schedule(
        args.world_sizes,
        args.layouts,
        args.scaling,
        args.repetitions,
        candidate_root=args.candidate_root.resolve(),
        baseline_root=args.baseline_root.resolve() if args.baseline_root else None,
    )
    write_json(
        args.output / "manifest.json",
        dict(
            status="planned" if args.dry_run else "running",
            warmup=args.warmup,
            timed_updates=args.steps,
            schedule=rows,
            driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            profiling_separate=args.profile,
        ),
    )
    if args.dry_run:
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() < max(args.world_sizes):
        write_json(
            args.output / "failure.json", dict(status="failed", reason="Insufficient CUDA devices")
        )
        raise RuntimeError("Insufficient CUDA devices; CPU fallback forbidden")
    measurements = [_launch(args, row, index) for index, row in enumerate(rows)]
    paired = []
    if args.baseline_root:
        for condition in itertools.product(args.world_sizes, args.layouts, args.scaling):
            if condition[1] != "contiguous":
                continue  # New layout costs have no unchanged-layout baseline.
            times = {}
            for revision in ("baseline", "candidate"):
                matching = [
                    value
                    for value in measurements
                    if (
                        value["condition"]["world_size"],
                        value["condition"]["layout"],
                        value["condition"]["scaling"],
                    )
                    == condition
                    and value["condition"]["revision"] == revision
                ]
                matching.sort(key=lambda value: value["condition"]["repetition"])
                times[revision] = [
                    value["measurements"]["slowest_rank_ms_per_step"] for value in matching
                ]
            paired.append(
                dict(
                    world_size=condition[0],
                    layout=condition[1],
                    scaling=condition[2],
                    verdict=paired_slowdown_verdict(
                        times["baseline"], times["candidate"], warmup=args.warmup, steps=args.steps
                    ),
                )
            )
    profiles = []
    if args.profile:
        for index, row in enumerate(rows):
            if row["repetition"] == 0:
                profiles.append(_launch(args, row, index, profile=True))
    write_json(
        args.output / "result.json",
        dict(
            status="measured",
            measurements=measurements,
            paired_slowdown=paired,
            profiles=profiles,
            descriptive_layouts=[layout for layout in args.layouts if layout != "contiguous"],
            qualification="descriptive only" if not paired else "see paired bounds",
            quality="not measured",
            native_paging="unsupported; not requested",
        ),
    )


if __name__ == "__main__":
    main()
