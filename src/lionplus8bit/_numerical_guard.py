"""Per-tensor numerical guards, with optional fused CUDA coordinate scans."""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

if triton is not None:

    @triton.jit
    def _guard(
        G,
        S,
        FLAG,
        NONZERO,
        N: tl.constexpr,
        NS: tl.constexpr,
        GRAD_LIMIT,
        STATE_LIMIT,
        B: tl.constexpr,
        DEVICE_LIMIT: tl.constexpr = False,
        TRACK_NONZERO: tl.constexpr = False,
    ):
        i = tl.program_id(0) * B + tl.arange(0, B)
        # Integer magnitude comparisons preserve subnormals even under FTZ compilation.
        g = tl.load(G + i, i < N, other=0).to(tl.float32).to(tl.uint32, bitcast=True) & 0x7FFFFFFF
        s = tl.load(S + i, i < NS, other=0).to(tl.float32).to(tl.uint32, bitcast=True) & 0x7FFFFFFF
        limit = tl.load(GRAD_LIMIT) if DEVICE_LIMIT else tl.full((), GRAD_LIMIT, tl.float32)
        gl = limit.to(tl.uint32, bitcast=True)
        sl = tl.full((), STATE_LIMIT, tl.float32).to(tl.uint32, bitcast=True)
        found = tl.sum(((g > 0) & (g < gl)) | ((s > 0) & (s < sl)), 0) > 0
        tl.atomic_max(FLAG, found.to(tl.int32))
        if TRACK_NONZERO:
            tl.atomic_max(NONZERO, (tl.sum(g > 0, 0) > 0).to(tl.int32))


def _value_flags(params, states, limits, cutoff, return_nonzero):
    device_limits = isinstance(limits, torch.Tensor)
    device = params[0].device if params else limits.device if device_limits else "cpu"
    flags = torch.zeros(len(params), device=device, dtype=torch.int32)
    nonzero = torch.zeros_like(flags) if return_nonzero else None
    for i, p in enumerate(params):
        state = states.get(p, {})
        value = state.get("absmax1", state.get("state1"))
        count = value.numel() if value is not None else 0
        if not p.numel() and not count:
            continue
        state_limit = cutoff * (2**25 if "absmax1" in state else 1)
        if (
            triton is not None
            and p.device.type == "cuda"
            and p.grad.is_contiguous()
            and (value is None or value.is_contiguous())
        ):
            _guard[(triton.cdiv(max(p.numel(), count), 1024),)](
                p.grad,
                value if value is not None else p.grad,
                flags[i:],
                nonzero[i:] if return_nonzero else flags[i:],
                p.numel(),
                count,
                limits[i:] if device_limits else limits,
                state_limit,
                1024,
                device_limits,
                return_nonzero,
            )
        else:
            x = p.grad.detach().float().abs()
            positive = x > 0
            tiny = (positive & (x < (limits[i] if device_limits else limits))).any()
            if return_nonzero:
                # NaN/Inf must count as nonzero for the caller's norm validation.
                nonzero[i].copy_((x != 0).any())
            if value is not None:
                x = value.float().abs()
                tiny = tiny | ((x > 0) & (x < state_limit)).any()
            flags[i].copy_(tiny)
    return (flags, nonzero) if return_nonzero else flags


@torch.no_grad()
def has_tiny_values(params, states, gradient_limit, cutoff=1e-20):
    """Aggregate compatibility API; the scan still selects its path per tensor."""
    if not params:
        return False
    if gradient_limit > torch.finfo(torch.float32).max:
        gradient_limit = float("inf")
    return bool(_value_flags(params, states, gradient_limit, cutoff, False).any())


@torch.no_grad()
def tiny_value_flag(params, states, limits, cutoff=1e-20, *, return_nonzero=False):
    """Return one int32 tiny flag per tensor, optionally with gradient-nonzero flags.

    ``limits`` contains one FP32 device threshold per active tensor. A strided
    gradient/state uses the eager scan without changing the path of other tensors.
    The optional nonzero vector distinguishes exact zeros from norm underflow in
    the same scan and requires no host readback.
    """
    return _value_flags(params, states, limits, cutoff, return_nonzero)
