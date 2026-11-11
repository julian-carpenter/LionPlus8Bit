"""Matched real-audio subset study. External cached data/model sources are required."""

import argparse
import hashlib
import importlib
import inspect
import json
import math
import platform
import statistics
import time
from pathlib import Path

import bitsandbytes as bnb
import torch

from lionplus8bit import LionPlus8bit


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spike_score(losses, window=1000):
    """Fraction above the trailing mean by >=10 std; exclude the current point."""
    if len(losses) <= window:
        return None
    spikes = 0
    for i in range(window, len(losses)):
        history = losses[i - window : i]
        delta = losses[i] - statistics.mean(history)
        spikes += delta > 0 and delta >= 10 * statistics.pstdev(history)
    return spikes / (len(losses) - window)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--strategy", choices=["global", "adagc"], required=True)
    parser.add_argument("--momentum-bits", type=int, choices=[8, 32], default=8)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory exists; refusing overwrite")
    config = json.loads(args.config.read_text())
    for key in ("steps", "batch_size", "accumulation"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            parser.error(f"{key} must be a positive integer")
    if config["steps"] <= 150:
        parser.error("steps must exceed the 150-step timing warmup")
    if not torch.cuda.is_available():
        parser.error("CUDA required")
    if sha(config["data"]) != config["data_sha256"]:
        parser.error("data hash mismatch")
    module = importlib.import_module(config["model_module"])
    model_type = getattr(module, config["model_class"])
    source = inspect.getfile(model_type)
    if sha(source) != config["model_sha256"]:
        parser.error("model source hash mismatch")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    # Trusted tensor-only cache, not an arbitrary serialized Python object.
    data = torch.load(config["data"], map_location="cpu", weights_only=True)
    model = model_type(**config["model_kwargs"]).cuda()
    initial_hash = hashlib.sha256(
        b"".join(t.detach().cpu().numpy().tobytes() for t in model.state_dict().values())
    ).hexdigest()
    ps = list(model.parameters())
    opt = LionPlus8bit(
        ps,
        clipping=args.strategy,
        lr=config["lr"],
        betas=tuple(config["betas"]),
        weight_decay=config["weight_decay"],
        max_grad_norm=config["max_grad_norm"],
        min_8bit_size=4096 if args.momentum_bits == 8 else max(p.numel() for p in ps) + 1,
    )
    steps, batch, accumulation = config["steps"], config["batch_size"], config["accumulation"]
    order = torch.randint(
        len(data["train"][0]),
        (steps, accumulation, batch),
        generator=torch.Generator().manual_seed(args.seed + 1000),
    )
    order_hash = hashlib.sha256(order.numpy().tobytes()).hexdigest()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    scaler = torch.amp.GradScaler("cuda", init_scale=32)
    calls = []
    opt.register_step_post_hook(lambda *unused: calls.append(1))
    args.output.mkdir(parents=True)
    torch.save(order, args.output / "order.pt")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    rows = []
    begin = time.perf_counter()

    def forward(x):
        return model(x, torch.full((len(x),), 8000, device="cuda"), return_clip_logits=True)[-1]

    for step, microbatches in enumerate(order):
        torch.cuda.synchronize()
        start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        losses = []
        for indices in microbatches:
            x, y = (v[indices].cuda() for v in data["train"])
            with torch.autocast("cuda", dtype=torch.float16):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(forward(x), y)
            losses.append(float(loss.detach()))
            scaler.scale(loss / accumulation).backward()
        scaler.unscale_(opt)
        torch.cuda.synchronize()  # Charge queued training work before isolating diagnostics.
        diagnostic_start = time.perf_counter()
        raw_norms = (
            torch.stack(
                torch._foreach_norm(
                    [p.grad for p in ps if p.grad is not None], 2, dtype=torch.float32
                )
            )
            .cpu()
            .tolist()
        )
        diagnostic_seconds = time.perf_counter() - diagnostic_start
        before = len(calls)
        scaler.step(opt)
        success = len(calls) > before
        scaler.update()
        if success:
            scheduler.step()
        torch.cuda.synchronize()
        row = dict(
            step=step + 1,
            loss=statistics.mean(losses),
            successful=success,
            scale=scaler.get_scale(),
            seconds=time.perf_counter() - start - diagnostic_seconds,
            diagnostic_seconds=diagnostic_seconds,
            raw_norms=raw_norms,
        )
        if args.strategy == "global":
            norm = math.hypot(*raw_norms)
            row["clipping"] = dict(
                scales=[min(1.0, config["max_grad_norm"] / norm) if norm else 1.0] * len(raw_norms)
            )
        if success and opt.last_clipping is not None:
            row["clipping"] = opt.last_clipping
        rows.append(row)
    model.eval()
    values = []
    with torch.no_grad():
        for i in range(0, len(data["valid"][0]), batch):
            x, y = (v[i : i + batch].cuda() for v in data["valid"])
            with torch.autocast("cuda", dtype=torch.float16):
                output = forward(x)
            values.append(
                (
                    float(torch.nn.functional.binary_cross_entropy_with_logits(output.float(), y)),
                    len(x),
                )
            )
    result = dict(
        strategy=args.strategy,
        momentum_bits=args.momentum_bits,
        seed=args.seed,
        config=config,
        config_sha256=sha(args.config),
        initial_sha256=initial_hash,
        order_sha256=order_hash,
        environment=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            bitsandbytes=bnb.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
        ),
        validation_bce=sum(v * n for v, n in values) / sum(n for _, n in values),
        updates=len(rows),
        successful_updates=len(calls),
        overflow_skips=len(rows) - len(calls),
        spike_score=spike_score([r["loss"] for r in rows]),
        peak_allocated=torch.cuda.max_memory_allocated(),
        peak_reserved=torch.cuda.max_memory_reserved(),
        elapsed_seconds=time.perf_counter() - begin,
        examples_per_second=batch
        * accumulation
        / statistics.mean(r["seconds"] for r in rows[150:]),
        finite_losses=all(math.isfinite(r["loss"]) for r in rows),
        scope="Exploratory real-audio subset, custom clip-presence BCE; no production convergence claim",
    )
    (args.output / "raw_updates.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    torch.save(
        dict(
            model=model.state_dict(),
            optimizer=opt.state_dict(),
            scaler=scaler.state_dict(),
            scheduler=scheduler.state_dict(),
            rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(),
        ),
        args.output / "final.pt",
    )
    print(json.dumps(result))
    if (
        not result["finite_losses"]
        or not math.isfinite(result["validation_bce"])
        or len(calls) != steps
    ):
        raise SystemExit("Nonfinite loss or skipped update: study acceptance failed")


if __name__ == "__main__":
    main()
