"""Slow, read-only diagnostics. Run outside timed intervals on matched gradient replay."""

import torch


@torch.no_grad()
def decoded_momentum(optimizer, parameter):
    state = optimizer.state[parameter]
    value = state["state1"]
    if value.dtype == torch.uint8:
        return (
            state["qmap1"][value.long()].reshape(-1)
            * state["absmax1"].repeat_interleave(256)[: value.numel()]
        ).reshape_as(parameter)
    return value


@torch.no_grad()
def compare_signs(candidate, control, clipped_gradients):
    """Compare the next sign decisions given identical, already-clipped gradients.

    Optimizers must have matching group/parameter order, shapes, beta1 and replay
    history. Training runs with different gradients are not a quantization oracle.
    """
    a = [(p, g["betas"][0]) for g in candidate.param_groups for p in g["params"]]
    b = [(p, g["betas"][0]) for g in control.param_groups for p in g["params"]]
    gradients = list(clipped_gradients)
    if len(a) != len(b) or len(a) != len(gradients):
        raise ValueError("Parameter and gradient counts must match")
    rows = []
    for (p, beta), (q, other_beta), gradient in zip(a, b, gradients):
        if p.shape != q.shape or gradient.shape != p.shape or beta != other_beta:
            raise ValueError("Shapes and beta1 must match")
        m, reference = decoded_momentum(candidate, p), decoded_momentum(control, q)
        c = beta * m + (1 - beta) * gradient.float()
        oracle = beta * reference + (1 - beta) * gradient.float()
        rows.append(
            dict(
                elements=p.numel(),
                sign_disagreements=int((c.sign() != oracle.sign()).sum()),
                max_momentum_error=float((m - reference).abs().max()),
                minimum_reference_sign_margin=float(oracle.abs().min()),
            )
        )
    return rows
