"""Lion+ using native bitsandbytes 8-bit momentum and update kernels.

Algorithm 7, arXiv:2506.04192v1: https://arxiv.org/html/2506.04192v1#A3.SS2
Requires bitsandbytes >= 0.50.2 and its supported PyTorch/backend environment.
"""

from __future__ import annotations

import struct
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import math
from collections.abc import Callable, Iterable
from typing import Any

import bitsandbytes as bnb
import torch
from packaging.version import Version
from torch import Tensor
from ._numerical_guard import tiny_value_flag
from ._strict import strict_update
from ._layout import validate_parameter_layouts, parameter_layout, canonical_work, copy_back
from ._paging import allocate_state, validate_paging_request, checkpoint_values
from ._clipping import ClippingConfig, ClippingState, prepare_clipping, commit_clipping

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

    Uses one device and dense nonoverlapping parameters; preserves original storage
    layout and gradient objects. Native work and state use logical row-major order.
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
        *,
        clipping: str = "global",
        adagc_beta: float = 0.99,
        adagc_factor: float = 1.04,
        adagc_warmup_steps: int = 100,
    ) -> None:
        if Version(bnb.__version__) < Version("0.50.2"):
            raise ImportError("LionPlus8bit requires bitsandbytes >= 0.50.2")
        _validate_hyperparameters(
            dict(lr=lr, betas=betas, weight_decay=weight_decay, max_grad_norm=max_grad_norm)
        )
        if not isinstance(min_8bit_size, int) or min_8bit_size < 0:
            raise ValueError("min_8bit_size must be a nonnegative integer")

        self.clipping_config = ClippingConfig(
            clipping, adagc_beta, adagc_factor, adagc_warmup_steps
        )
        self._clipping_state = None
        self.last_clipping = None
        self.last_step_stats = None
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

    @property
    def clipping(self):
        return self.clipping_config.mode

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        param_group = dict(param_group)
        for key, expected in (
            ("clipping", self.clipping),
            ("adagc_beta", self.clipping_config.beta),
            ("adagc_factor", self.clipping_config.factor),
            ("adagc_warmup_steps", self.clipping_config.warmup),
        ):
            if key in param_group and param_group.pop(key) != expected:
                raise ValueError("Conflicting clipping policy in parameter group")
        current = (
            self.param_groups[0]["max_grad_norm"]
            if self.param_groups
            else self._initial_max_grad_norm
        )
        param_group.setdefault("max_grad_norm", current)
        if param_group["max_grad_norm"] != current and self.param_groups:
            raise ValueError("All parameter groups must use the same max_grad_norm")
        super().add_param_group(param_group)
        group_index = len(self.param_groups) - 1
        for parameter_index in range(len(self.param_groups[group_index]["params"])):
            self._effective_override(group_index, parameter_index)
        if self._clipping_state is not None:
            self._clipping_state.resize(sum(len(g["params"]) for g in self.param_groups))

    def _effective_override(self, group_index, parameter_index):
        """Own resolved defaults; accept subsequent parameter-specific manager updates."""
        if not hasattr(self, "_parameter_overrides"):
            self._parameter_overrides = {}
            self._manager_overrides = {}
            self._saved_override_params = set()
        manager = bnb.optim.GlobalOptimManager.get_instance()
        p = self.param_groups[group_index]["params"][parameter_index]
        observed = manager.pid2config.get(id(p), {})
        if p not in self._parameter_overrides:
            self._parameter_overrides[p] = deepcopy(
                manager.index2config.get((group_index, parameter_index), {})
            )
            self._manager_overrides[p] = {}
        effective = self._parameter_overrides[p]
        # Module registrations are defaults for parameters awaiting allocation. They
        # must not replace a checkpoint policy, or mutate another optimizer's index map.
        if p not in self._saved_override_params and not self.state.get(p):
            for module, attr, config in manager.module_weight_config_triple:
                if getattr(module, attr) is p:
                    effective.update(deepcopy(config))
            effective.update(deepcopy(observed))
        else:
            previous = self._manager_overrides[p]
            for key, value in observed.items():
                if key not in previous or value != previous[key]:
                    effective[key] = deepcopy(value)
        self._manager_overrides[p] = deepcopy(observed)
        return effective

    def check_overrides(self):
        for gi, group in enumerate(self.param_groups):
            for pi in range(len(group["params"])):
                self._effective_override(gi, pi)

    def get_config(self, gindex, pindex, group):
        # Match the pinned native defaults, with an optimizer-owned override merge.
        config = {key: group[key] for key in ("betas", "eps", "weight_decay", "lr")}
        config.update(
            alpha=group.get("alpha", 0.0),
            t_alpha=group.get("t_alpha"),
            t_beta3=group.get("t_beta3"),
            optim_bits=self.args.optim_bits,
            min_8bit_size=self.args.min_8bit_size,
            max_unorm=self.args.max_unorm,
            skip_zeros=self.args.skip_zeros,
        )
        config.update(self._effective_override(gindex, pindex))
        return config

    @contextmanager
    def _without_checkpoint_hooks(self, operation):
        names = [f"_optimizer_{operation}_{when}_hooks" for when in ("pre", "post")]
        saved = [getattr(self, name) for name in names]
        try:
            for name in names:
                setattr(self, name, {})
            yield
        finally:
            for name, hooks in zip(names, saved):
                setattr(self, name, hooks)

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
        if any(p.device != device for group in self.param_groups for p in group["params"]):
            raise RuntimeError("Use one optimizer per device, including inactive parameters")
        all_params = [p for group in self.param_groups for p in group["params"]]
        validate_parameter_layouts(all_params)
        validate_paging_request(all_params, is_paged=self.is_paged)
        layouts = getattr(self, "_parameter_layouts", {})
        parameter_storages = {
            (p.device, p.untyped_storage().data_ptr()) for p in all_params if p.numel()
        }
        for p in params:
            if id(p) in layouts and layouts[id(p)] != parameter_layout(p):
                raise RuntimeError(
                    "Live parameter dtype/device/layout changes require a fresh optimizer"
                )
            if (
                p.grad.numel()
                and (p.grad.device, p.grad.untyped_storage().data_ptr()) in parameter_storages
            ):
                raise RuntimeError("Parameter and gradient storage must not overlap")
            layouts[id(p)] = parameter_layout(p)
        self._parameter_layouts = layouts
        if not self.initialized:
            self.check_overrides()
            self.to_gpu()  # Guards consume native state, including deferred CPU checkpoints.
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
        if self.clipping == "adagc":
            return self._adagc_step(params, gradients, torch.stack(norms), max_norm, loss)
        total_norm = torch.linalg.vector_norm(torch.stack(norms))
        norm_value = float(total_norm)
        # Rare FP64 path covers overflow, underflow and poorly resolved squared norms.
        self._wide_clip = not math.isfinite(norm_value) or norm_value < 1e-18
        norm_recovery_count = len(params) if self._wide_clip else 0
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
        guard = (
            tiny_value_flag(
                params,
                self.state,
                torch.full(
                    (len(params),),
                    float("inf")
                    if cutoff / scale_value > torch.finfo(torch.float32).max
                    else cutoff / scale_value,
                    device=device,
                ),
            )
            .cpu()
            .tolist()
        )
        tiny_by_id = {id(p): bool(tiny) for p, tiny in zip(params, guard)}
        global_subnormal = self._subnormal
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
                    self._subnormal = global_subnormal or tiny_by_id[id(p)]
                    self.update_step(group, p, gi, pi)
            if self.is_paged and device.type == "cuda":
                torch.cuda.synchronize(device)
        finally:
            self._clip_scale = None
        self.last_step_stats = dict(
            active_tensors=len(params),
            clipped_tensors=len(params) if self._needs_clip else 0,
            strict_tensors=sum(bool(self.state[p].get("strict_math", False)) for p in params),
            tiny_tensors=sum(tiny_by_id.values()),
            norm_recovery_tensors=norm_recovery_count,
        )
        return loss

    def _adagc_step(self, params, gradients, norms, max_norm, loss):
        all_params = [p for group in self.param_groups for p in group["params"]]
        if len({id(p) for p in all_params}) != len(all_params):
            raise ValueError("Duplicate parameters are unsupported")
        state = self._clipping_state
        if state is None:
            state = ClippingState(len(all_params), params[0].device)
        state.resize(len(all_params))
        slot_map = {id(p): i for i, p in enumerate(all_params)}
        slots = [slot_map[id(p)] for p in params]
        plan = prepare_clipping(
            norms, slots, self.clipping_config, state, max_norm, validate_norms=False
        )
        guard, nonzero = tiny_value_flag(
            params, self.state, (1e-20 / plan.scales).float(), return_nonzero=True
        )
        packed = state.pack(norms, plan, guard, nonzero)
        suspect = [
            i
            for i, row in enumerate(packed)
            if not math.isfinite(row[0]) or (row[0] < 1e-18 and row[5])
        ]
        if suspect:
            norms = norms.double()
            for i in suspect:
                norms[i] = torch.linalg.vector_norm(gradients[i], dtype=torch.float64)
            if not bool(torch.isfinite(norms).all()):
                raise RuntimeError("LionPlus8bit encountered a non-finite AdaGC gradient norm")
            plan = prepare_clipping(norms, slots, self.clipping_config, state, max_norm)
            guard, nonzero = tiny_value_flag(
                params, self.state, (1e-20 / plan.scales).float(), return_nonzero=True
            )
            packed = state.pack(norms, plan, guard, nonzero)
        if any(not math.isfinite(row[2]) for row in packed):
            raise RuntimeError("Non-finite AdaGC history")
        for row in packed:
            n, scale = row[:2]
            if n > 0 and (scale <= 0 or (row[6] < n and row[6] < 2**-149)):
                raise ValueError("AdaGC clipping threshold is below FP32 gradient representability")
        plan.host = [row[1:4] for row in packed]
        self._adagc_observed = {}
        decisions = {id(p): (i, row) for i, (p, row) in enumerate(zip(params, packed))}
        try:
            if not self.initialized:
                self.check_overrides()
                self.to_gpu()
                self.initialized = True
            for gi, group in enumerate(self.param_groups):
                for pi, p in enumerate(group["params"]):
                    if p.grad is None:
                        continue
                    i, row = decisions[id(p)]
                    norm, scale, _, _, tiny = row[:5]
                    self._clip_scale = plan.scales[i]
                    self._needs_clip = scale < 1
                    self._wide_clip = scale < torch.finfo(torch.float32).tiny or norm < 1e-18
                    self._subnormal = bool(tiny) or (
                        0 < norm < torch.finfo(torch.float32).tiny * 1024
                    )
                    self._native_scale = scale
                    if not self.state[p]:
                        self.init_state(group, p, gi, pi)
                    self.prefetch_state(p)
                    self.update_step(group, p, gi, pi)
            if self.is_paged and params[0].device.type == "cuda":
                torch.cuda.synchronize(params[0].device)
            if self._adagc_observed:
                observed = norms.double() * plan.scales
                for i, p in enumerate(params):
                    if id(p) in self._adagc_observed:
                        observed[i] = self._adagc_observed[id(p)]
                corrected = prepare_clipping(
                    norms, slots, self.clipping_config, state, max_norm, observed=observed
                )
                plan.candidates, plan.ready, plan.host = corrected.candidates, corrected.ready, None
            commit_clipping(state, plan)
            self._clipping_state = state
            self.last_step_stats = dict(
                active_tensors=len(params),
                clipped_tensors=sum(row[1] < 1 for row in packed),
                strict_tensors=sum(bool(self.state[p].get("strict_math", False)) for p in params),
                tiny_tensors=sum(bool(row[4]) for row in packed),
                norm_recovery_tensors=len(suspect),
            )
            self.last_clipping = dict(
                norms=[row[0] for row in packed],
                scales=[row[1] for row in packed],
                histories=[row[1] for row in plan.host],
                steps=state.steps,
                warmup=state.steps <= self.clipping_config.warmup,
            )
        finally:
            self._clip_scale = None
            self._adagc_observed.clear()
        return loss

    def state_dict(self):
        for hook in self._optimizer_state_dict_pre_hooks.values():
            hook(self)
        with self._without_checkpoint_hooks("state_dict"):
            result = checkpoint_values(super().state_dict())
        result["lionplus_config"] = dict(
            version=3,
            state_order="logical-row-major-v1",
            parameter_shapes=[[tuple(p.shape) for p in g["params"]] for g in self.param_groups],
            clipping=self.clipping,
            adagc_beta=self.clipping_config.beta,
            adagc_factor=self.clipping_config.factor,
            adagc_warmup_steps=self.clipping_config.warmup,
            edge_policy="positive-bootstrap-v1",
            min_8bit_size=self.args.min_8bit_size,
            is_paged=self.is_paged,
            master_weights=self.master_weights,
            native_gradient_scale=self.native_gradient_scale,
            batched_norm=self.batched_norm,
            overrides=[
                [deepcopy(self._effective_override(gi, pi)) for pi in range(len(g["params"]))]
                for gi, g in enumerate(self.param_groups)
            ],
        )
        if self.clipping == "adagc" and self._clipping_state is not None:
            state = self._clipping_state
            result["lionplus_clipping"] = dict(
                history=state.history.clone(),
                initialized=state.initialized.clone(),
                exceptions=dict(state.exceptions),
                steps=state.steps,
                group_lengths=[len(g["params"]) for g in self.param_groups],
                shapes=[tuple(p.shape) for g in self.param_groups for p in g["params"]],
            )
        for hook in self._optimizer_state_dict_post_hooks.values():
            replacement = hook(self, result)
            if replacement is not None:
                result = replacement
        return result

    def load_state_dict(self, state_dict, move_to_device=True):
        state_dict = state_dict.copy()
        for hook in self._optimizer_load_state_dict_pre_hooks.values():
            replacement = hook(self, state_dict)
            if replacement is not None:
                state_dict = replacement
        config = state_dict.get("lionplus_config")
        if config is not None and config.get("version") == 3:
            if config.get("state_order") != "logical-row-major-v1":
                raise ValueError("Unsupported LionPlus checkpoint state order")
            shapes = [[tuple(p.shape) for p in g["params"]] for g in self.param_groups]
            if config.get("parameter_shapes") != shapes:
                raise ValueError("Incompatible checkpoint parameter shape mapping")
        validate_parameter_layouts([p for g in self.param_groups for p in g["params"]])
        # Validate incoming groups before changing overrides, allocation policy,
        # native state or clipping state. The base loader assigns groups eagerly.
        incoming = state_dict.get("param_groups", [])
        if len(incoming) != len(self.param_groups):
            raise ValueError("Incompatible checkpoint parameter groups")
        max_norm = None
        overrides = config.get("overrides", []) if config else []
        if overrides and len(overrides) != len(incoming):
            raise ValueError("Incompatible checkpoint override groups")
        for gi, (group, current) in enumerate(zip(incoming, self.param_groups)):
            if len(group.get("params", [])) != len(current["params"]):
                raise ValueError("Incompatible checkpoint parameter group size")
            _validate_hyperparameters(group)
            if max_norm is not None and group["max_grad_norm"] != max_norm:
                raise ValueError("All parameter groups must use the same max_grad_norm")
            max_norm = group["max_grad_norm"]
            if overrides:
                if len(overrides[gi]) != len(group["params"]):
                    raise ValueError("Incompatible checkpoint parameter overrides")
                for override in overrides[gi]:
                    _validate_hyperparameters(dict(group, **override))
        if (
            config is not None
            and config.get("version") in (2, 3)
            and config.get("edge_policy") != "positive-bootstrap-v1"
        ):
            raise ValueError("Unsupported AdaGC edge policy")
        mode = (
            config.get("clipping", "global")
            if config and config.get("version") in (2, 3)
            else "global"
        )
        clipping_config = (
            ClippingConfig(
                mode,
                config.get("adagc_beta", 0.99),
                config.get("adagc_factor", 1.04),
                config.get("adagc_warmup_steps", 100),
            )
            if config
            else ClippingConfig()
        )
        saved = state_dict.get("lionplus_clipping")
        restored = None
        if mode == "adagc" and saved is None and state_dict.get("state"):
            raise ValueError("AdaGC checkpoint is missing clipping history")
        if saved is not None:
            parameters = [p for g in self.param_groups for p in g["params"]]
            if (
                mode != "adagc"
                or saved["group_lengths"] != [len(g["params"]) for g in self.param_groups]
                or saved["shapes"] != [tuple(p.shape) for p in parameters]
            ):
                raise ValueError("Incompatible AdaGC parameter mapping")
            history, ready = saved["history"], saved["initialized"]
            if (
                history.dtype != torch.float32
                or ready.dtype != torch.bool
                or history.shape != (len(parameters),)
                or ready.shape != history.shape
            ):
                raise ValueError("Invalid AdaGC history layout")
            if (
                isinstance(saved["steps"], bool)
                or not isinstance(saved["steps"], int)
                or saved["steps"] < 0
            ):
                raise ValueError("Invalid AdaGC step counter")
            if not bool(torch.isfinite(history).all()) or bool((history < 0).any()):
                raise ValueError("Invalid AdaGC history")
            exceptions = saved["exceptions"]
            if any(
                isinstance(i, bool)
                or not isinstance(i, int)
                or not 0 <= i < len(parameters)
                or not math.isfinite(v)
                or v < 2**-149
                for i, v in exceptions.items()
            ):
                raise ValueError("Invalid AdaGC exceptional history")
            effective = history.double().cpu()
            for i, v in exceptions.items():
                effective[i] = v
            if bool((ready.cpu() & (effective <= 0)).any()):
                raise ValueError("Initialized AdaGC history must be positive")
            restored = ClippingState(len(parameters), parameters[0].device)
            restored.history.copy_(history)
            restored.initialized.copy_(ready)
            restored.exceptions = dict(exceptions)
            restored.steps = saved["steps"]
        if config is not None:
            if config.get("version") not in (1, 2, 3):
                raise ValueError("Unsupported LionPlus checkpoint configuration version")
            if self.state and (
                self.is_paged != config["is_paged"]
                or self.master_weights != config["master_weights"]
            ):
                raise ValueError("Load changed allocation policy into a fresh optimizer")
        if move_to_device:
            validate_paging_request(
                [p for g in self.param_groups for p in g["params"]],
                is_paged=config["is_paged"] if config else self.is_paged,
            )
        state_dict = checkpoint_values(state_dict)
        with self._without_checkpoint_hooks("load_state_dict"):
            super().load_state_dict(state_dict, move_to_device=move_to_device)
        if config is not None:
            self.args.min_8bit_size = config["min_8bit_size"]
            self.is_paged = config["is_paged"]
            self.master_weights = config["master_weights"]
            self.native_gradient_scale = config["native_gradient_scale"]
            self.batched_norm = config.get("batched_norm", False)
        self._parameter_overrides = {}
        self._manager_overrides = {}
        self._saved_override_params = set()
        overrides = config.get("overrides", []) if config else []
        for gi, group in enumerate(self.param_groups):
            for pi, p in enumerate(group["params"]):
                override = overrides[gi][pi] if overrides else {}
                self._parameter_overrides[p] = deepcopy(override)
                self._manager_overrides[p] = deepcopy(override)
                self._saved_override_params.add(p)
                # A separate owned dictionary lets the public manager update this
                # parameter after loading without aliasing a checkpoint or peer.
                self.mng.pid2config[id(p)] = deepcopy(override)
        self.clipping_config = clipping_config
        self._clipping_state = restored
        self.last_clipping = None
        self.last_step_stats = None
        self._validate_groups()
        self.initialized = False
        self._parameter_layouts = {}
        for hook in self._optimizer_load_state_dict_post_hooks.values():
            hook(self)

    def get_state_buffer(self, p, dtype=torch.float32):
        return allocate_state(p, dtype, is_paged=self.is_paged)

    def to_gpu(self):
        validate_paging_request(
            [p for g in self.param_groups for p in g["params"]], is_paged=self.is_paged
        )
        super().to_gpu()

    @torch.no_grad()
    def update_step(self, group, p, gindex, pindex):
        if self._clip_scale is None:
            raise RuntimeError("Call LionPlus8bit.step(), not update_step() directly")
        original_grad = p.grad
        state = self.state[p]
        config = self.get_config(gindex, pindex, group)
        use_master = self.master_weights and p.dtype != torch.float32
        if use_master and "master_weight" not in state:
            state["master_weight"] = p.detach().float().contiguous()
        target = state["master_weight"] if use_master else canonical_work(p)
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
        staged_weight = not p.is_contiguous()
        if use_master or use_scale or staged_weight:
            grad = grad.contiguous()
        if (
            self.clipping == "adagc"
            and self._needs_clip
            and not use_scale
            and (
                self._wide_clip
                or self._subnormal
                or (original_grad.dtype != torch.float32 and not use_master)
            )
        ):
            self._adagc_observed[id(p)] = torch.linalg.vector_norm(grad, dtype=torch.float64)
        # Once a FP32 state uses strict arithmetic, retain it across checkpoints.
        strict = self._subnormal or state.get("strict_math", False)
        current = torch.cuda.current_stream(p.device) if p.device.type == "cuda" else None
        default = torch.cuda.default_stream(p.device) if current is not None else None
        try:
            if strict:
                state["strict_math"] = True
                strict_update(target, grad, state, config, current, default)
            elif use_master or use_scale or staged_weight:
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
            if use_master or staged_weight:
                copy_back(p, target)
        finally:
            p.grad = original_grad
