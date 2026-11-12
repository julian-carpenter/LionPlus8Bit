"""Hardware-gated Transformer Engine FP8 example with FP32 trainable parameters.

Requires a separately qualified Transformer Engine build and native FP8 GPU.
This example illustrates the gradient/checkpoint contract; a successful short run
is not native FP8 numerical or performance qualification. SM75 fails before any
FP8 model allocation. No FP8 primary parameters or fused main_grad are used.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from _fp8_integration import probe_fp8  # noqa: E402
from _training_state import load_training_checkpoint, save_training_checkpoint  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--precision", choices=["fp8", "high"], default="fp8")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    capability = probe_fp8()
    if capability["status"] != "available-unqualified":
        print(capability["reason"], file=sys.stderr)
        raise SystemExit(1)
    # Optional imports and allocations occur only after both hardware/backend gates.
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe
    from lionplus8bit import LionPlus8bit

    torch.manual_seed(7)
    model = te.Linear(
        128, 128, params_dtype=torch.float32, fuse_wgrad_accumulation=False, device="cuda"
    )
    optimizer = LionPlus8bit(model.parameters(), lr=0.001, clipping="adagc")
    scaler = torch.amp.GradScaler("cuda")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    data = {"attempts": 0, "consumed_batches": 0, "successful_updates": 0}
    if args.resume:
        if args.checkpoint is None:
            parser.error("--resume requires --checkpoint")
        data = load_training_checkpoint(
            args.checkpoint, model=model, optimizer=optimizer, scaler=scaler, scheduler=scheduler
        )["data_state"]
    fp8_recipe = recipe.DelayedScaling()
    for attempt in range(data["attempts"], args.steps):
        optimizer.zero_grad(set_to_none=True)
        generator = torch.Generator().manual_seed(3301 + attempt)
        inputs = torch.randn(16, 128, generator=generator).cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            with te.fp8_autocast(enabled=args.precision == "fp8", fp8_recipe=fp8_recipe):
                loss = model(inputs).float().square().mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        for parameter in model.parameters():
            if parameter.dtype != torch.float32:
                raise RuntimeError("Initial FP8 contract requires FP32 primary parameters")
            if getattr(parameter, "main_grad", None) is not None:
                raise RuntimeError(
                    "Fused main_grad accumulation is outside this example's contract"
                )
            if parameter.grad is not None and parameter.grad.dtype != torch.float32:
                raise RuntimeError("Optimizer requires ordinary dequantized FP32 gradients")
        finite = all(
            p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()
        )
        scaler.step(optimizer)
        scaler.update()
        data["attempts"] += 1
        data["consumed_batches"] += 1
        if finite:
            data["successful_updates"] += 1
            scheduler.step()
        if args.checkpoint:
            # TE's supported model state_dict includes FP8 extra state. The
            # backend may advance amax/scales on forward even if AMP skips.
            save_training_checkpoint(
                args.checkpoint,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                data_state=data,
            )
    print({"precision": args.precision, **data, "native_fp8_qualified": False})


if __name__ == "__main__":
    main()
