"""Lion+ using native bitsandbytes 8-bit momentum and update kernels.

Algorithm 7, arXiv:2506.04192v1: https://arxiv.org/html/2506.04192v1#A3.SS2
Requires bitsandbytes >= 0.50.2 and its supported PyTorch/backend environment.
"""

from __future__ import annotations

import struct
from contextlib import nullcontext
import math
from collections.abc import Callable, Iterable
from typing import Any

import bitsandbytes as bnb
import torch
from packaging.version import Version
from torch import Tensor
from ._numerical_guard import has_tiny_values

__all__ = ["LionPlus8bit"]


def _validate_hyperparameters(group: dict[str, Any]) -> None:
    lr, wd = group["lr"], group["weight_decay"]
    if not math.isfinite(lr) or lr < 0:
        raise ValueError("lr must be finite and nonnegative")
    if not math.isfinite(wd) or wd < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    for name, value in [("lr", lr), ("weight_decay", wd)]:
        try:
            native = struct.unpack("f", struct.pack("f", value))[0]
        except OverflowError as error:
            raise ValueError(f"{name} must be representable as FP32") from error
        if not math.isfinite(native):
            raise ValueError(f"{name} must be representable as FP32")
    betas = group["betas"]
    if len(betas) != 2 or any(
        not 0 <= beta < 1 or struct.unpack("f", struct.pack("f", beta))[0] == 1 for beta in betas
    ):
        raise ValueError("betas must contain two values in [0, 1)")
    if not group["max_grad_norm"] > 0:
        raise ValueError("max_grad_norm must be positive (or math.inf)")


class LionPlus8bit(bnb.optim.Lion8bit):
    """Lion+ with native blockwise uint8 momentum and global clipping.

    Uses one device and contiguous parameters; preserves original gradient objects.
    FP16/BF16 parameters have FP32 master weights by default (additional 4N bytes).
    Prefer FP32 parameters with autocast. Disable master_weights only deliberately.
    Ordinary values use batched norm reduction and native gradient scaling.
    Tiny values use strict PyTorch arithmetic; extreme clipping uses FP64 work.
    Disable both optimization flags for legacy arithmetic compatibility.
    Native kernels run on the default stream with explicit dependencies to the
    caller stream. Nonpaged updates avoid per-parameter device synchronization.
    Checkpoints include allocation policy; native overrides are saved by group order.
    Paging, distributed shards, compilation and CUDA graphs remain unverified;
    gradient shards and CUDA graph capture are unsupported.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 5e-5,
        betas: tuple[float, float] = (0.95, 0.98),
        weight_decay: float = 1e-2,
        max_grad_norm: float = 4.0,
        min_8bit_size: int = 4096,
        is_paged: bool = False,
        master_weights: bool = True,
        native_gradient_scale: bool = True,
        batched_norm: bool = True,
    ) -> None:
        if Version(bnb.__version__) < Version("0.50.2"):
            raise ImportError("LionPlus8bit requires bitsandbytes >= 0.50.2")
        _validate_hyperparameters(
            dict(lr=lr, betas=betas, weight_decay=weight_decay, max_grad_norm=max_grad_norm)
        )
        if not isinstance(min_8bit_size, int) or min_8bit_size < 0:
            raise ValueError("min_8bit_size must be a nonnegative integer")

        self._initial_max_grad_norm = float(max_grad_norm)
        self._clip_scale: Tensor | None = None
        self.batched_norm = batched_norm
        self.master_weights = master_weights
        self.native_gradient_scale = native_gradient_scale
        self._needs_clip = False
        self._wide_clip = False
        self._subnormal = False
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            min_8bit_size=min_8bit_size,
            is_paged=is_paged,
        )
        self.non_castable_tensor_keys.add("master_weight")
        self.defaults["max_grad_norm"] = float(max_grad_norm)
        self._validate_groups()

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        param_group = dict(param_group)
        current = (
            self.param_groups[0]["max_grad_norm"]
            if self.param_groups
            else self._initial_max_grad_norm
        )
        param_group.setdefault("max_grad_norm", current)
        if param_group["max_grad_norm"] != current and self.param_groups:
            raise ValueError("All parameter groups must use the same max_grad_norm")
        super().add_param_group(param_group)

    def _validate_groups(self) -> float:
        max_norm = self.param_groups[0]["max_grad_norm"]
        for group in self.param_groups:
            _validate_hyperparameters(group)
            if group["max_grad_norm"] != max_norm:
                raise ValueError("All parameter groups must use the same max_grad_norm")
        return float(max_norm)

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        max_norm = self._validate_groups()
        params = [p for group in self.param_groups for p in group["params"] if p.grad is not None]
        if not params:
            return loss

        # Check every active gradient before any native state initialization.
        for p in params:
            if p.layout != torch.strided or p.grad.layout != torch.strided:
                raise RuntimeError("LionPlus8bit supports only dense parameters and gradients")
            if p.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                raise RuntimeError("LionPlus8bit requires float32, float16 or bfloat16 dtype")

        device = params[0].device
        for p in params:
            if not p.is_contiguous():
                raise RuntimeError(
                    "LionPlus8bit requires contiguous parameters; views must not be silently copied"
                )
            if p.device != device:
                raise RuntimeError("Use one optimizer per device")
        if not self.initialized:
            self.check_overrides()
        for gi, group in enumerate(self.param_groups):
            for pi, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                config = self.get_config(gi, pi, group)
                _validate_hyperparameters(dict(config, max_grad_norm=max_norm))
                if config["skip_zeros"] or config["max_unorm"]:
                    raise ValueError(
                        "skip_zeros and max_unorm overrides change Lion+ semantics and are unsupported"
                    )
        gradients = [p.grad.detach() for p in params]
        norms = (
            torch._foreach_norm(gradients, 2, dtype=torch.float32)
            if self.batched_norm
            else [torch.linalg.vector_norm(g, dtype=torch.float32) for g in gradients]
        )
        total_norm = torch.linalg.vector_norm(torch.stack(norms))
        norm_value = float(total_norm)
        # Rare FP64 path covers overflow, underflow and poorly resolved squared norms.
        self._wide_clip = not math.isfinite(norm_value) or norm_value < 1e-18
        if self._wide_clip:
            total_norm = torch.linalg.vector_norm(
                torch.stack([torch.linalg.vector_norm(g, dtype=torch.float64) for g in gradients])
            )
            norm_value = float(total_norm)
        if not math.isfinite(norm_value):
            raise RuntimeError("LionPlus8bit encountered a non-finite global gradient norm")
        self._needs_clip = norm_value > max_norm
        if self._needs_clip and max_norm < 2**-149:
            raise ValueError("Clipping threshold is below FP32 gradient representability")
        self._subnormal = (
            0 < norm_value < torch.finfo(torch.float32).tiny * 1024
        ) or max_norm < torch.finfo(torch.float32).tiny * 1024
        if self._needs_clip:
            ratio = max_norm / norm_value
            self._wide_clip |= ratio < torch.finfo(torch.float32).tiny
            self._clip_scale = (
                torch.tensor(max_norm, device=device, dtype=torch.float64) / total_norm.double()
                if self._wide_clip
                else total_norm.new_tensor(max_norm).div_(total_norm)
            )
        else:
            self._clip_scale = torch.ones_like(total_norm)
        self._native_scale = (
            float(self._clip_scale)
            if self.native_gradient_scale and self._needs_clip and not self._wide_clip
            else 1.0
        )
        # A large global norm can hide tiny coordinates. Also inspect existing
        # momentum scales so decay/cancellation cannot silently re-enter FTZ math.
        cutoff = 1e-20
        scale_value = max_norm / norm_value if self._needs_clip else 1.0
        self._subnormal |= has_tiny_values(params, self.state, cutoff / scale_value)
        try:
            if not self.initialized:
                self.check_overrides()
                self.to_gpu()
                self.initialized = True
            for gi, group in enumerate(self.param_groups):
                for pi, p in enumerate(group["params"]):
                    if p.grad is None:
                        continue
                    if not self.state[p]:
                        self.init_state(group, p, gi, pi)
                    self.prefetch_state(p)
                    self.update_step(group, p, gi, pi)
            if self.is_paged and device.type == "cuda":
                torch.cuda.synchronize(device)
        finally:
            self._clip_scale = None
        return loss

    def state_dict(self):
        result = super().state_dict()
        result["lionplus_config"] = dict(
            version=1,
            min_8bit_size=self.args.min_8bit_size,
            is_paged=self.is_paged,
            master_weights=self.master_weights,
            native_gradient_scale=self.native_gradient_scale,
            batched_norm=self.batched_norm,
            overrides=[
                [
                    dict(self.mng.pid2config.get(id(p), self.mng.index2config.get((gi, pi), {})))
                    for pi, p in enumerate(g["params"])
                ]
                for gi, g in enumerate(self.param_groups)
            ],
        )
        return result

    def load_state_dict(self, state_dict, move_to_device=True):
        config = state_dict.get("lionplus_config")
        if config is not None:
            if config.get("version") != 1:
                raise ValueError("Unsupported LionPlus checkpoint configuration version")
            if self.state and (
                self.is_paged != config["is_paged"]
                or self.master_weights != config["master_weights"]
            ):
                raise ValueError("Load changed allocation policy into a fresh optimizer")
            self.args.min_8bit_size = config["min_8bit_size"]
            self.is_paged = config["is_paged"]
            self.master_weights = config["master_weights"]
            self.native_gradient_scale = config["native_gradient_scale"]
            self.batched_norm = config.get("batched_norm", False)
            for group, overrides in zip(self.param_groups, config.get("overrides", [])):
                for p, override in zip(group["params"], overrides):
                    self.mng.pid2config[id(p)] = override
        super().load_state_dict(state_dict, move_to_device=move_to_device)
        self._validate_groups()
        self.initialized = False

    @torch.no_grad()
    def update_step(self, group, p, gindex, pindex):
        if self._clip_scale is None:
            raise RuntimeError("Call LionPlus8bit.step(), not update_step() directly")
        original_grad = p.grad
        state = self.state[p]
        config = self.get_config(gindex, pindex, group)
        use_master = self.master_weights and p.dtype != torch.float32
        if use_master and "master_weight" not in state:
            state["master_weight"] = p.detach().float().clone()
        target = state["master_weight"] if use_master else p
        use_scale = (
            self.native_gradient_scale
            and p.dtype == torch.float32
            and self._needs_clip
            and not self._wide_clip
            and not self._subnormal
            and not state.get("strict_math", False)
        )
        grad = original_grad
        if self._needs_clip and not use_scale:
            dtype = torch.float64 if self._wide_clip else torch.float32
            grad = (original_grad.detach().to(dtype) * self._clip_scale).to(
                torch.float32 if use_master else original_grad.dtype
            )
        if use_master:
            grad = grad.float()
        if use_master or use_scale:
            grad = grad.contiguous()
        # Once a FP32 state uses strict arithmetic, retain it across checkpoints.
        strict = self._subnormal or state.get("strict_math", False)
        current = torch.cuda.current_stream(p.device) if p.device.type == "cuda" else None
        default = torch.cuda.default_stream(p.device) if current is not None else None
        try:
            if strict:
                state["strict_math"] = True
                momentum = state["state1"]
                if momentum.dtype == torch.uint8:
                    momentum = (
                        state["qmap1"][momentum.long()].reshape(-1)
                        * state["absmax1"].repeat_interleave(256)[: momentum.numel()]
                    ).reshape_as(p)
                beta1, beta2 = config["betas"]
                direction = (momentum * beta1 + grad.float() * (1 - beta1)).sign()
                target.mul_(1 - config["lr"] * config["weight_decay"]).add_(
                    direction, alpha=-config["lr"]
                )
                momentum = momentum * beta2 + grad.float() * (1 - beta2)
                if state["state1"].dtype == torch.uint8:
                    peak = momentum.abs().max()
                    divisor = torch.where(peak > 0, peak, torch.ones_like(peak))
                    normalized = (momentum / divisor).contiguous()
                    if current is not None and current != default:
                        default.wait_stream(current)
                    context = torch.cuda.stream(default) if default is not None else nullcontext()
                    with context:
                        quantized, quant = bnb.functional.quantize_blockwise(
                            normalized, code=state["qmap1"], blocksize=256
                        )
                    if current is not None:
                        if current != default:
                            current.wait_stream(default)
                        normalized.record_stream(default)
                        state["qmap1"].record_stream(default)
                        quantized.record_stream(current)
                        quant.absmax.record_stream(current)
                    state["state1"].copy_(quantized)
                    state["absmax1"].copy_(quant.absmax * divisor)
                else:
                    state["state1"].copy_(momentum)
                state["step"] += 1
            elif use_master or use_scale:
                if current is not None:
                    if current != default:
                        default.wait_stream(current)
                context = torch.cuda.stream(default) if default is not None else nullcontext()
                with context:
                    state["step"] += 1
                    if state["state1"].dtype == torch.float32:
                        bnb.functional.optimizer_update_32bit(
                            "lion",
                            grad,
                            target,
                            state["state1"],
                            config["betas"][0],
                            0.0,
                            state["step"],
                            config["lr"],
                            None,
                            config["betas"][1],
                            weight_decay=config["weight_decay"],
                            gnorm_scale=self._native_scale if use_scale else 1.0,
                        )
                    else:
                        bnb.functional.optimizer_update_8bit_blockwise(
                            "lion",
                            grad,
                            target,
                            state["state1"],
                            None,
                            config["betas"][0],
                            config["betas"][1],
                            0.0,
                            0.0,
                            0.0,
                            state["step"],
                            config["lr"],
                            state["qmap1"],
                            None,
                            state["absmax1"],
                            None,
                            config["weight_decay"],
                            gnorm_scale=self._native_scale if use_scale else 1.0,
                        )
            else:
                if current is not None:
                    if current != default:
                        default.wait_stream(current)
                context = torch.cuda.stream(default) if default is not None else nullcontext()
                with context:
                    p.grad = grad
                    super().update_step(group, p, gindex, pindex)
            if current is not None and current != default and not strict:
                if current != default:
                    current.wait_stream(default)
                for tensor in [grad, target, *[v for v in state.values() if isinstance(v, Tensor)]]:
                    if tensor.device.type == "cuda":
                        tensor.record_stream(default)
            if use_master:
                p.copy_(target)
        finally:
            p.grad = original_grad
