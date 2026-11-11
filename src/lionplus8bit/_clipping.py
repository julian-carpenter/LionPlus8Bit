"""AdaGC preparation; history is independent of native momentum initialization."""

from collections import OrderedDict
from dataclasses import dataclass
import math
import struct

import torch


@dataclass(frozen=True)
class ClippingConfig:
    mode: str = "global"
    beta: float = 0.99
    factor: float = 1.04
    warmup: int = 100

    def __post_init__(self):
        if self.mode not in ("global", "adagc"):
            raise ValueError("clipping must be global or adagc")
        if not 0 <= self.beta < 1 or struct.unpack("f", struct.pack("f", self.beta))[0] == 1:
            raise ValueError("adagc_beta must be in [0,1) and not round to one in FP32")
        if not math.isfinite(self.factor) or self.factor < 1:
            raise ValueError("adagc_factor must be finite and >= 1")
        if isinstance(self.warmup, bool) or not isinstance(self.warmup, int) or self.warmup < 0:
            raise ValueError("adagc_warmup_steps must be a nonnegative integer")


class ClippingState:
    def __init__(self, size, device):
        self.history = torch.zeros(size, dtype=torch.float32, device=device)
        self.initialized = torch.zeros(size, dtype=torch.bool, device=device)
        self.exceptions = {}  # rare slot -> FP64 host value
        self.steps = 0
        self._index_cache = OrderedDict()
        self._index_device = self.history.device
        self._packed = None

    def indices(self, slots):
        if self._index_device != self.history.device:
            self._index_cache.clear()
            self._index_device = self.history.device
        key = tuple(slots)
        if key not in self._index_cache:
            self._index_cache[key] = torch.tensor(
                slots, device=self.history.device, dtype=torch.long
            )
            # Reuse alternating active sets without retaining every combination.
            if len(self._index_cache) > 8:
                self._index_cache.popitem(last=False)
        self._index_cache.move_to_end(key)
        return self._index_cache[key]

    def pack(self, norms, plan, guard, nonzero=None):
        count = norms.numel()
        if (
            self._packed is None
            or self._packed.shape[0] < count
            or self._packed.device != norms.device
        ):
            self._packed = torch.empty((count, 7), device=norms.device, dtype=torch.float64)
        values = (norms, plan.scales, plan.candidates, plan.ready, guard)
        if nonzero is not None:
            values += (nonzero, plan.bounds)
        packed = self._packed[:count, : len(values)]
        for column, value in enumerate(values):
            packed[:, column].copy_(value)
        return packed.cpu().tolist()

    def resize(self, size):
        if size > self.history.numel():
            extra = size - self.history.numel()
            self.history = torch.cat((self.history, self.history.new_zeros(extra)))
            self.initialized = torch.cat((self.initialized, self.initialized.new_zeros(extra)))


@dataclass
class ClipPlan:
    slots: list
    scales: torch.Tensor
    candidates: torch.Tensor
    ready: torch.Tensor
    bounds: torch.Tensor
    host: list | None = None


@torch.no_grad()
def prepare_clipping(norms, slots, config, state, max_norm, observed=None, *, validate_norms=True):
    """Prepare history; the optimizer may defer validation until FP64 norm recovery."""
    n = norms.double()
    if validate_norms and n.device.type == "cpu" and not bool(torch.isfinite(n).all()):
        raise RuntimeError("Non-finite AdaGC gradient norm")
    idx = state.indices(slots)
    old = state.history[idx].double()
    if state.exceptions:
        exceptional = torch.tensor(
            [state.exceptions.get(i, 0.0) for i in slots], device=n.device, dtype=torch.float64
        )
        old = torch.where(exceptional > 0, exceptional, old)
    ready = state.initialized[idx]
    if state.steps < config.warmup:
        total = torch.linalg.vector_norm(n)
        denominator = torch.where(total > 0, total, torch.ones_like(total))
        h = torch.where(
            total > 0,
            torch.minimum(torch.ones_like(n), max_norm / denominator),
            torch.ones_like(n),
        )
        bounds = torch.where(total > max_norm, max_norm * (n / denominator), n)
    else:
        ceiling = torch.where(ready, old * config.factor, torch.full_like(n, max_norm))
        bounds = torch.minimum(n, ceiling)
        h = torch.where(
            n > 0,
            torch.minimum(torch.ones_like(n), ceiling / torch.where(n > 0, n, torch.ones_like(n))),
            torch.ones_like(n),
        )
    # Native ordinary coefficients are FP32; preserve extreme ratios in FP64 work.
    h = torch.where(h >= torch.finfo(torch.float32).tiny, h.float().double(), h)
    observed = h * n if observed is None else observed
    if state.steps < config.warmup:
        candidate = torch.where(
            (observed > 0) & ready,
            torch.minimum(old, observed),
            torch.where(observed > 0, observed, old),
        )
    else:
        candidate = torch.where(ready, config.beta * old + (1 - config.beta) * observed, observed)
    active = candidate >= 2**-149
    return ClipPlan(slots, h, candidate, active, bounds)


@torch.no_grad()
def commit_clipping(state, plan):
    # One packed readback can be supplied by the optimizer alongside scale/guard metadata.
    rows = (
        plan.host
        if plan.host is not None
        else torch.stack((plan.scales, plan.candidates, plan.ready.double()), 1).cpu().tolist()
    )
    plan.host = rows
    for slot, (_, value, ready) in zip(plan.slots, rows):
        if not ready:
            state.exceptions.pop(slot, None)
        elif value < torch.finfo(torch.float32).tiny or value > torch.finfo(torch.float32).max:
            state.exceptions[slot] = value
        else:
            state.exceptions.pop(slot, None)
    idx = state.indices(plan.slots)
    ordinary = (
        plan.ready
        & (plan.candidates >= torch.finfo(torch.float32).tiny)
        & (plan.candidates <= torch.finfo(torch.float32).max)
    )
    state.history[idx] = torch.where(ordinary, plan.candidates, 0).float()
    state.initialized[idx] = plan.ready
    state.steps += 1
