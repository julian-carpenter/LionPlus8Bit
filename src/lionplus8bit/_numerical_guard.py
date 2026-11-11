"""Optional fused coordinate guard. Eager reference remains available without Triton."""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

if triton is not None:

    @triton.jit
    def _guard(
        G, S, FLAG, N: tl.constexpr, NS: tl.constexpr, GRAD_LIMIT, STATE_LIMIT, B: tl.constexpr
    ):
        i = tl.program_id(0) * B + tl.arange(0, B)
        # Integer magnitude comparisons preserve subnormals even under FTZ compilation.
        g = tl.load(G + i, i < N, other=0).to(tl.float32).to(tl.uint32, bitcast=True) & 0x7FFFFFFF
        s = tl.load(S + i, i < NS, other=0).to(tl.float32).to(tl.uint32, bitcast=True) & 0x7FFFFFFF
        gl = tl.full((), GRAD_LIMIT, tl.float32).to(tl.uint32, bitcast=True)
        sl = tl.full((), STATE_LIMIT, tl.float32).to(tl.uint32, bitcast=True)
        found = tl.sum(((g > 0) & (g < gl)) | ((s > 0) & (s < sl)), 0) > 0
        tl.atomic_max(FLAG, found.to(tl.int32))


@torch.no_grad()
def has_tiny_values(params, states, gradient_limit, cutoff=1e-20):
    if not params:
        return False
    device = params[0].device
    if gradient_limit > torch.finfo(torch.float32).max:
        gradient_limit = float("inf")
    if triton is not None and device.type == "cuda" and all(p.grad.is_contiguous() for p in params):
        flag = torch.zeros((), device=device, dtype=torch.int32)
        for p in params:
            if not p.numel():
                continue
            state = states.get(p, {})
            value = state.get("absmax1", state.get("state1"))
            count = value.numel() if value is not None else 0
            limit = cutoff * (2**25 if "absmax1" in state else 1)
            _guard[(triton.cdiv(max(p.numel(), count), 1024),)](
                p.grad,
                value if value is not None else p.grad,
                flag,
                p.numel(),
                count,
                gradient_limit,
                limit,
                1024,
            )
        return bool(flag)
    flags = []
    for p in params:
        if not p.numel():
            continue
        x = p.grad.detach().float().abs()
        flags.append(((x > 0) & (x < gradient_limit)).any())
        state = states.get(p, {})
        if "state1" in state:
            x = state.get("absmax1", state["state1"]).float().abs()
            limit = cutoff * (2**25 if state["state1"].dtype == torch.uint8 else 1)
            flags.append(((x > 0) & (x < limit)).any())
    return bool(torch.stack(flags).any()) if flags else False
