#!/usr/bin/env python3
"""Controlled full-group crash/relaunch fixture for deterministic training replay.

Fresh output directories only. This fixture uses replayable map-style sample IDs
and stateless augmentation; arbitrary external streams are outside its contract.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import signal
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from _training_state import (  # noqa: E402
    _base_model,
    capture_training_rng,
    load_training_checkpoint,
    restore_training_rng,
    save_training_checkpoint,
)

FAULTS = (
    "before-forward",
    "inside-accumulation",
    "after-backward",
    "after-step",
    "during-write",
    "after-publication",
)


class ReplayDataset(Dataset):
    def __init__(self, sample_ids):
        self.sample_ids = sample_ids

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        sample = int(self.sample_ids[index])
        generator = torch.Generator().manual_seed(1753 + sample)
        augmentation = torch.rand((), generator=generator)
        inputs = torch.randn(65, generator=generator) + augmentation
        target = torch.tensor([sample % 3 / 3, sample % 7 / 7], dtype=torch.float32)
        return sample, augmentation, inputs, target


class StochasticModel(torch.nn.Module):
    def __init__(self, layout="contiguous"):
        super().__init__()
        self.layout = layout
        if layout == "channels-last":
            self.layers = torch.nn.Sequential(
                torch.nn.Conv2d(65, 65, 3, padding=1),
                torch.nn.Dropout(0.2),
                torch.nn.Tanh(),
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(65, 2),
            ).to(memory_format=torch.channels_last)
        else:
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(65, 65),
                torch.nn.Dropout(0.2),
                torch.nn.Tanh(),
                torch.nn.Linear(65, 2),
            )
        if layout == "transposed":
            weight = self.layers[0].weight
            self.layers[0].weight = torch.nn.Parameter(weight.detach().t().contiguous().t())
        elif layout not in ("contiguous", "channels-last"):
            raise ValueError("layout must be contiguous, transposed or channels-last")
        self.register_buffer("local_forwards", torch.zeros((), dtype=torch.int64))
        self.register_buffer("local_rank_marker", torch.tensor(int(os.environ.get("RANK", 0))))

    def forward(self, inputs):
        self.local_forwards.add_(1)
        if self.layout == "channels-last":
            inputs = (
                inputs[:, :, None, None]
                .expand(-1, -1, 3, 3)
                .contiguous(memory_format=torch.channels_last)
            )
        return self.layers(inputs)


def _fault_marker(directory):
    try:
        descriptor = os.open(directory / "fault-fired", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w") as handle:
        handle.write("one controlled failure; subsequent launches recover\n")
        handle.flush()
        os.fsync(handle.fileno())
    return True


def _worker(args):
    from lionplus8bit import LionPlus8bit

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; refusing fallback")
        torch.cuda.set_device(local_rank)
    if world > 1:
        from datetime import timedelta

        dist.init_process_group(
            "nccl" if args.device == "cuda" else "gloo", timeout=timedelta(seconds=120)
        )
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(919)
    random.seed(919 + rank)
    np.random.seed(919 + rank)
    model = StochasticModel(layout=args.layout).to(args.device)
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if args.device == "cuda" else None,
            broadcast_buffers=False,
            bucket_cap_mb=4,
        )
    if args.compiled:
        model = torch.compile(model, backend=args.compiled_backend)
    optimizer = LionPlus8bit(
        model.parameters(), lr=0.003, clipping="adagc", adagc_warmup_steps=0, weight_decay=0
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    scaler = torch.amp.GradScaler(
        args.device, init_scale=128.0, growth_interval=2, enabled=args.precision != "fp32"
    )
    directory = Path(args.worker_directory)
    checkpoint = directory / "checkpoint"
    device_properties = (
        torch.cuda.get_device_properties(local_rank) if args.device == "cuda" else None
    )
    placement = {
        "rank": rank,
        "local_rank": local_rank,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "uuid": str(device_properties.uuid) if device_properties is not None else None,
        "name": device_properties.name if device_properties is not None else "cpu",
    }
    (directory / f"rank-{rank}-placement.json").write_text(json.dumps(placement, indent=2))
    torch.manual_seed(1119 + rank)
    order_rng = torch.Generator().manual_seed(77)
    order = torch.randperm(48 * world, generator=order_rng).tolist()[rank::world]
    data = {
        "consumed_batches": 0,
        "successful_updates": 0,
        "attempts": 0,
        "sample_ids": [],
        "augmentation": [],
        "sampler_order": order,
        "partial_accumulation": "rollback-entire-group",
        "world_size": world,
    }
    restored = None
    if (checkpoint / "manifest.json").exists():
        restored = load_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            restore_rng=False,
        )
        data = restored["data_state"]
    loader_rng = torch.Generator().manual_seed(9921 + rank)
    loader_kwargs = {
        "batch_size": 2,
        "num_workers": args.loader_workers,
        "generator": loader_rng,
        "persistent_workers": args.persistent_workers and args.loader_workers > 0,
    }
    if args.loader_workers:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(
        ReplayDataset(data["sampler_order"][data["consumed_batches"] * 2 :]), **loader_kwargs
    )
    data["sampler_generator_state"] = order_rng.get_state()
    data["loader_generator_initial_state"] = loader_rng.get_state()
    iterator = iter(loader)
    data["loader_generator_state"] = loader_rng.get_state()
    if restored is not None:
        restore_training_rng(restored["rng_state"])
        restored_state = {
            "model": _base_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict(),
            "data": data,
            "rng_state": capture_training_rng(),
        }
        torch.save(restored_state, directory / f"rank-{rank}-restored.pt")
    else:
        save_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            data_state=data,
        )

    def fail(point):
        if (
            rank == 0
            and args.fault_point == point
            and data["attempts"] == 2
            and _fault_marker(directory)
        ):
            if point == "loader":
                os.kill(iterator._workers[0].pid, signal.SIGKILL)
                # The parent must observe the actual child failure; never turn a
                # successful manually loaded snapshot into a recovery result.
                time.sleep(0.2)
                next(iterator)
                raise RuntimeError("Killed loader worker was not reported")
            os._exit(86)

    weight_pointer = _base_model(model).layers[0].weight.data_ptr()
    expected_stride = _base_model(model).layers[0].weight.stride()
    while data["attempts"] < 6:
        optimizer.zero_grad(set_to_none=True)
        fail("before-forward")
        fail("loader")
        for microbatch in range(2):
            sample, augmentation, inputs, target = next(iterator)
            data["consumed_batches"] += 1
            data["sample_ids"].extend(sample.tolist())
            data["augmentation"].extend(augmentation.tolist())
            with torch.autocast(
                args.device,
                dtype=torch.float16 if args.device == "cuda" else torch.bfloat16,
                enabled=args.precision != "fp32",
            ):
                loss = (model(inputs.to(args.device)) - target.to(args.device)).square().mean() / 2
            # A skipped AMP attempt still consumes the full accumulation group.
            if data["attempts"] == 1 and scaler.is_enabled():
                loss = loss * float("inf")
            scaler.scale(loss).backward()
            if microbatch == 0:
                fail("inside-accumulation")
        fail("after-backward")
        scaler.unscale_(optimizer)
        finite = all(
            p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()
        )
        scaler.step(optimizer)
        scaler.update()
        weight = _base_model(model).layers[0].weight
        if weight.data_ptr() != weight_pointer or weight.stride() != expected_stride:
            raise RuntimeError("Weight storage or layout changed during recovery trajectory")
        if finite:
            data["successful_updates"] += 1
            scheduler.step()
        fail("after-step")
        data["attempts"] += 1
        # Fault selection uses the attempt boundary before increment, so the
        # checkpoint writer sees attempt 3 while the trigger names attempt 2.
        active_write_fault = (
            rank == 0
            and args.fault_point == "during-write"
            and data["attempts"] == 3
            and _fault_marker(directory)
        )
        original_save = torch.save
        if active_write_fault:

            def interrupted_save(payload, handle, *unused, **kwargs):
                handle.write(b"deliberately interrupted component")
                handle.flush()
                os.fsync(handle.fileno())
                os._exit(86)

            torch.save = interrupted_save
        try:
            save_training_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                data_state=data,
            )
        finally:
            torch.save = original_save
        if (
            rank == 0
            and args.fault_point == "after-publication"
            and data["attempts"] == 3
            and _fault_marker(directory)
        ):
            os._exit(86)
    # Capture next draws and rank-local buffers alongside actual numerical state.
    base = _base_model(model)
    final = {
        "model": base.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "scheduler": scheduler.state_dict(),
        "data": data,
        "next_cpu_rng": torch.rand(4),
        "next_python_rng": random.random(),
        "next_numpy_rng": float(np.random.random()),
        "layout": {"kind": args.layout, "weight_stride": base.layers[0].weight.stride()},
        "next_cuda_rng": torch.rand(4, device="cuda").cpu() if args.device == "cuda" else None,
    }
    torch.save(final, directory / f"rank-{rank}-final.pt")
    if args.compiled:
        compiler = {
            "backend": args.compiled_backend,
            "graphs": int(torch._dynamo.utils.counters["stats"].get("unique_graphs", 0)),
            "cache_directory": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        }
        (directory / f"rank-{rank}-compiler.json").write_text(json.dumps(compiler, indent=2))
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


def _launch(args, directory, fault, visible_devices=None):
    command = [sys.executable]
    if args.world_size > 1:
        command += [
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc-per-node={args.world_size}",
            "--max-restarts=0",
        ]
    command += [
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-directory",
        str(directory),
        "--device",
        args.device,
        "--loader-workers",
        str(args.loader_workers),
        "--prefetch-factor",
        str(args.prefetch_factor),
        "--fault-point",
        fault,
        "--layout",
        args.layout,
        "--precision",
        args.precision,
    ]
    if args.persistent_workers:
        command.append("--persistent-workers")
    if args.compiled:
        command += ["--compiled-backend", args.compiled_backend]
    environment = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8")
    if visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = visible_devices
    if args.compiled:
        launch_number = len(list(directory.glob("launch-*.log")))
        environment["TORCHINDUCTOR_CACHE_DIR"] = str(directory / f"compiler-cache-{launch_number}")
    with (directory / f"launch-{len(list(directory.glob('launch-*.log')))}.log").open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True
        )
        try:
            code = process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            code = 124
    return {
        "command": command,
        "exit_code": code,
        "visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
        "compiler_cache_directory": environment.get("TORCHINDUCTOR_CACHE_DIR")
        if args.compiled
        else None,
    }


def run_restart_suite(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    from _integration_common import capabilities
    import hashlib

    environment = capabilities()
    for source in [Path(__file__).resolve(), ROOT / "examples/_training_state.py"]:
        environment["source_hashes"][str(source.relative_to(ROOT))] = hashlib.sha256(
            source.read_bytes()
        ).hexdigest()
    environment["determinism"] = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "NCCL_ALGO": os.environ.get("NCCL_ALGO", "default"),
        "NCCL_PROTO": os.environ.get("NCCL_PROTO", "default"),
        "NCCL_P2P_DISABLE": os.environ.get("NCCL_P2P_DISABLE", "default"),
        "deterministic_algorithms": True,
        "tf32": False,
        "bucket_cap_mb": 4,
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True))
    if args.case == "loader" and args.loader_workers < 1:
        raise RuntimeError("Loader failure case requires --loader-workers >= 1")
    baseline = output / "uninterrupted"
    baseline.mkdir()
    control = _launch(args, baseline, "none")
    cases = []
    faults = (
        [args.fault_point]
        if args.fault_point != "all"
        else (
            ["loader"]
            if args.case == "loader"
            else ["during-write"]
            if args.case == "checkpoint-write"
            else FAULTS
        )
    )
    for fault in faults:
        directory = output / fault
        directory.mkdir()
        injected = _launch(args, directory, fault)
        comparison_directory = baseline
        comparison_launch = control
        checkpoint_manifest = None
        restored_equal = False
        if injected["exit_code"] != 0 and (directory / "checkpoint/manifest.json").exists():
            checkpoint_manifest = json.loads((directory / "checkpoint/manifest.json").read_text())
            if args.resume_visible_devices is not None:
                comparison_directory = output / f"{fault}-control-on-resume-placement"
                comparison_directory.mkdir()
                shutil.copytree(directory / "checkpoint", comparison_directory / "checkpoint")
                comparison_launch = _launch(
                    args, comparison_directory, "none", args.resume_visible_devices
                )
        recovered = (
            _launch(args, directory, fault, args.resume_visible_devices)
            if injected["exit_code"] != 0
            else None
        )
        same = (
            comparison_launch["exit_code"] == 0
            and recovered is not None
            and recovered["exit_code"] == 0
        )
        if same:
            restored_equal = checkpoint_manifest is not None
            for rank in range(args.world_size):

                def load(path):
                    return torch.load(path, weights_only=False, map_location="cpu")

                same &= _equal(
                    load(comparison_directory / f"rank-{rank}-final.pt"),
                    load(directory / f"rank-{rank}-final.pt"),
                )
                component = next(
                    item for item in checkpoint_manifest["components"] if item["rank"] == rank
                )
                # The manifest can advance during recovery. Check the immutable
                # generation selected before relaunch, not the new latest one.
                payload = load(directory / "checkpoint" / component["file"])
                expected = {
                    "model": payload["model"],
                    "optimizer": payload["optimizer"],
                    "scaler": payload["scaler"],
                    "scheduler": payload["scheduler"],
                    "data": payload["data_state"],
                    "rng_state": payload["rng_state"],
                }
                restored_file = directory / f"rank-{rank}-restored.pt"
                restored_equal &= restored_file.exists() and _equal(load(restored_file), expected)
        compiler_records = [
            json.loads((directory / f"rank-{rank}-compiler.json").read_text())
            for rank in range(args.world_size)
            if (directory / f"rank-{rank}-compiler.json").exists()
        ]
        native_capture = len(compiler_records) == args.world_size and all(
            record["graphs"] > 0 for record in compiler_records
        )
        controlled_failure = (directory / "fault-fired").is_file()
        passed = bool(same and restored_equal and controlled_failure and control["exit_code"] == 0)
        if args.compiled and args.compiled_backend == "inductor":
            passed &= native_capture
        cases.append(
            {
                "compiler_records": compiler_records,
                "fault": fault,
                "status": "passed" if passed else "failed",
                "injected_exit_code": injected["exit_code"],
                "controlled_failure_observed": controlled_failure,
                "recovery_exit_code": recovered["exit_code"] if recovered else None,
                "bitwise_equal": bool(same),
                "restored_checkpoint_bitwise_equal": bool(restored_equal),
                "injection": injected,
                "recovery": recovered,
                "comparison": comparison_launch,
                "comparison_scope": "resume-placement checkpoint continuation"
                if args.resume_visible_devices
                else "same-placement uninterrupted",
                "initial_placement": [
                    json.loads((baseline / f"rank-{rank}-placement.json").read_text())
                    for rank in range(args.world_size)
                    if (baseline / f"rank-{rank}-placement.json").exists()
                ],
                "resumed_placement": [
                    json.loads((directory / f"rank-{rank}-placement.json").read_text())
                    for rank in range(args.world_size)
                    if (directory / f"rank-{rank}-placement.json").exists()
                ],
            }
        )
    for case in cases:
        initial_uuids = [placement["uuid"] for placement in case["initial_placement"]]
        resumed_uuids = [placement["uuid"] for placement in case["resumed_placement"]]
        case["physical_remap_qualified"] = bool(
            args.resume_visible_devices
            and case["status"] == "passed"
            and len(initial_uuids) == args.world_size
            and len(resumed_uuids) == args.world_size
            and all(
                old is not None and new is not None and old != new
                for old, new in zip(initial_uuids, resumed_uuids)
            )
        )
    result = {
        "suite": "restart",
        "status": "passed" if all(c["status"] == "passed" for c in cases) else "failed",
        "device": args.device,
        "world_size": args.world_size,
        "loader_workers": args.loader_workers,
        "prefetch_factor": args.prefetch_factor,
        "persistent_workers": args.persistent_workers,
        "compiled_model": args.compiled,
        "compiled_backend": args.compiled_backend if args.compiled else None,
        "native_compiled_cache_recreation_qualified": bool(
            args.compiled
            and args.compiled_backend == "inductor"
            and all(case["status"] == "passed" for case in cases)
        ),
        "precision": args.precision,
        "layout": args.layout,
        "resume_visible_devices": args.resume_visible_devices,
        "baseline": control,
        "cases": cases,
        "scope": "fixed-world-size replayable map dataset; rollback entire accumulation group",
        "stateful_loader": "unqualified; optional torchdata backend not exercised",
    }
    (output / "summary.json").write_text(json.dumps(result, sort_keys=True, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument("--loader-workers", type=int, choices=[0, 2, 4], default=0)
    parser.add_argument(
        "--case", choices=["trainer", "loader", "checkpoint-write"], default="trainer"
    )
    parser.add_argument("--output")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--fault-point", choices=[*FAULTS, "loader", "none", "all"], default="all")
    parser.add_argument("--prefetch-factor", type=int, choices=[1, 2], default=2)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--compiled-backend", choices=["eager", "inductor"])
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument(
        "--layout", choices=["contiguous", "transposed", "channels-last"], default="contiguous"
    )
    parser.add_argument(
        "--resume-visible-devices",
        help="CUDA_VISIBLE_DEVICES used only for recovery and its placement-matched control",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-directory", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.compiled = args.compiled or args.compiled_backend is not None
    args.compiled_backend = args.compiled_backend or "eager"
    if args.resume_visible_devices is not None:
        if args.device != "cuda":
            parser.error("--resume-visible-devices requires --device cuda")
        devices = args.resume_visible_devices.split(",")
        if (
            len(devices) != args.world_size
            or any(not item.strip() for item in devices)
            or len(set(devices)) != len(devices)
        ):
            parser.error(
                "resume device count must match fixed world size with distinct nonempty devices"
            )
    if args.worker:
        _worker(args)
    else:
        if not args.output:
            parser.error("--output is required")
        result = run_restart_suite(args)
        print(json.dumps({"status": result["status"], "output": args.output}))
        raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
