"""AdaGC preparation; history is independent of native momentum initialization."""

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
        self._slot_key = None
        self._indices = None
        self._packed = None

    def indices(self, slots):
        key = tuple(slots)
        if key != self._slot_key:
            self._indices = torch.tensor(slots, device=self.history.device, dtype=torch.long)
            self._slot_key = key
        return self._indices

    def pack(self, norms, plan, guard):
        count = norms.numel()
        if self._packed is None or self._packed.shape[0] < count:
            self._packed = torch.empty((count, 5), device=norms.device, dtype=torch.float64)
        packed = self._packed[:count]
        for column, value in enumerate((norms, plan.scales, plan.candidates, plan.ready, guard)):
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
    host: list | None = None


@torch.no_grad()
def prepare_clipping(norms, slots, config, state, max_norm, observed=None):
    """Compute candidates without mutating state. Norm finiteness is validated by caller."""
    n = norms.double()
    if n.device.type == "cpu" and not bool(torch.isfinite(n).all()):
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
        h = torch.where(
            total > 0,
            torch.minimum(
                torch.ones_like(n), max_norm / torch.where(total > 0, total, torch.ones_like(total))
            ),
            torch.ones_like(n),
        )
    else:
        ceiling = torch.where(ready, old * config.factor, torch.full_like(n, max_norm))
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
    return ClipPlan(slots, h, candidate, active)


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
