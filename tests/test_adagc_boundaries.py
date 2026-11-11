"""Regression coverage for norm recovery and clipping representability."""

import copy
import math
import weakref

import pytest
import torch

from lionplus8bit import LionPlus8bit
from lionplus8bit._clipping import ClippingConfig, ClippingState, prepare_clipping


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batched_norm", [True, False])
@pytest.mark.parametrize("phase", ["bootstrap", "warmup", "adaptive"])
def test_finite_norm_overflow_recovers_before_adagc_validation(dtype, batched_norm, phase):
    p = torch.nn.Parameter(torch.ones(2, dtype=dtype))
    opt = LionPlus8bit(
        [p],
        clipping="adagc",
        adagc_warmup_steps=int(phase == "warmup"),
        adagc_factor=1,
        max_grad_norm=1,
        betas=(0, 0),
        weight_decay=0,
        batched_norm=batched_norm,
    )
    if phase == "adaptive":
        p.grad = torch.tensor([1, 0], dtype=dtype)
        opt.step()
    p.grad = torch.full_like(p, 1e20)
    original = p.grad
    values = original.clone()
    assert torch.isfinite(original).all()
    assert torch.isinf(original.float().norm())

    opt.step()

    torch.testing.assert_close(
        opt.state[p]["state1"], torch.full((2,), 1 / math.sqrt(2)), rtol=1e-6, atol=0
    )
    assert opt._clipping_state.history[0].item() == pytest.approx(1, rel=1e-6)
    assert opt._clipping_state.steps == (2 if phase == "adaptive" else 1)
    assert p.grad is original
    torch.testing.assert_close(p.grad, values, rtol=0, atol=0)


@pytest.mark.parametrize("phase", ["bootstrap", "warmup", "adaptive"])
def test_exact_minimum_fp32_bound_produces_nonzero_gradient_and_history(phase):
    minimum = 2**-149
    p = torch.nn.Parameter(torch.ones(1))
    opt = LionPlus8bit(
        [p],
        clipping="adagc",
        adagc_warmup_steps=int(phase == "warmup"),
        adagc_factor=1,
        max_grad_norm=minimum,
        betas=(0, 0),
        weight_decay=0,
    )
    if phase == "adaptive":
        p.grad = torch.tensor([minimum])
        opt.step()
    p.grad = torch.tensor([49.0])
    original = p.grad

    opt.step()

    torch.testing.assert_close(opt.state[p]["state1"], torch.tensor([minimum]), rtol=0, atol=0)
    assert opt._clipping_state.initialized[0]
    assert opt._clipping_state.exceptions[0] == minimum
    assert p.grad is original
    torch.testing.assert_close(p.grad, torch.tensor([49.0]), rtol=0, atol=0)


@pytest.mark.parametrize("warmup", [0, 1])
@pytest.mark.parametrize("threshold", [2**-150, math.nextafter(2**-149, 0)])
@pytest.mark.parametrize("gradient", [49.0, 2**-149])
def test_below_minimum_bound_still_rejects_without_state_changes(warmup, threshold, gradient):
    p = torch.nn.Parameter(torch.ones(1))
    opt = LionPlus8bit([p], clipping="adagc", adagc_warmup_steps=warmup, max_grad_norm=threshold)
    p.grad = torch.tensor([gradient])
    before = p.detach().clone()

    with pytest.raises(ValueError, match="representability"):
        opt.step()

    assert not opt.state
    assert opt._clipping_state is None
    torch.testing.assert_close(p, before, rtol=0, atol=0)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_gradient_recovery_preserves_existing_state(bad):
    p = torch.nn.Parameter(torch.ones(2))
    opt = LionPlus8bit([p], clipping="adagc", adagc_warmup_steps=1)
    p.grad = torch.ones_like(p)
    opt.step()
    before = copy.deepcopy(opt.state[p])
    history = opt._clipping_state.history.clone()
    clipping_steps = opt._clipping_state.steps
    weights = p.detach().clone()
    p.grad = torch.tensor([1.0, bad])

    with pytest.raises(RuntimeError, match="[Nn]on-finite"):
        opt.step()

    torch.testing.assert_close(p, weights, rtol=0, atol=0)
    torch.testing.assert_close(opt.state[p]["state1"], before["state1"], rtol=0, atol=0)
    torch.testing.assert_close(opt._clipping_state.history, history, rtol=0, atol=0)
    assert opt._clipping_state.steps == clipping_steps
    assert opt.state[p]["step"] == before["step"]


@pytest.mark.parametrize(
    "values,recovered",
    [((1.0, 0.0), False), ((0.0, 0.0), False), ((1.0, 1e-30), True), ((1.0, 1e20), True)],
    ids=["normal-and-zero", "all-zero", "underflow", "finite-overflow"],
)
def test_fp64_recovery_only_reduces_suspect_nonzero_tensors(monkeypatch, values, recovered):
    p = torch.nn.Parameter(torch.ones(2))
    q = torch.nn.Parameter(torch.ones(3))
    opt = LionPlus8bit(
        [p, q],
        clipping="adagc",
        adagc_warmup_steps=0,
        max_grad_norm=math.inf,
        betas=(0, 0),
        weight_decay=0,
    )
    p.grad = torch.full_like(p, values[0])
    q.grad = torch.full_like(q, values[1])
    original_norm = torch.linalg.vector_norm
    recovery_inputs = []

    def record_norm(tensor, *args, **kwargs):
        if kwargs.get("dtype") == torch.float64:
            recovery_inputs.append(tensor.data_ptr())
        return original_norm(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "vector_norm", record_norm)
    opt.step()

    assert recovery_inputs == ([q.grad.data_ptr()] if recovered else [])
    for param in (p, q):
        torch.testing.assert_close(opt.state[param]["state1"], param.grad, rtol=0, atol=0)
    assert opt._clipping_state.initialized.tolist() == [value != 0 for value in values]
    assert opt._clipping_state.steps == 1


@pytest.mark.parametrize("warmup", [0, 1])
def test_plan_retains_intended_bound_before_coefficient_rounding(warmup):
    state = ClippingState(1, torch.device("cpu"))
    plan = prepare_clipping(
        torch.tensor([49.0]), [0], ClippingConfig("adagc", warmup=warmup), state, 2**-149
    )

    assert plan.bounds.tolist() == [2**-149]
    rows = state.pack(torch.tensor([49.0]), plan, torch.tensor([1]), torch.tensor([1]))
    assert len(rows[0]) == 7
    assert rows[0][5:] == [1, 2**-149]
    assert len(state.pack(torch.tensor([49.0]), plan, torch.tensor([1]))[0]) == 5


def test_prepare_can_defer_norm_validation_for_optimizer_recovery():
    state = ClippingState(1, torch.device("cpu"))
    norms = torch.tensor([math.inf])
    config = ClippingConfig("adagc")
    with pytest.raises(RuntimeError, match="[Nn]on-finite"):
        prepare_clipping(norms, [0], config, state, 1)

    prepare_clipping(norms, [0], config, state, 1, validate_norms=False)

    assert state.steps == 0
    assert state.history.tolist() == [0]
    assert not state.initialized[0]


def test_alternating_active_sets_reuse_index_tensors_after_resize():
    state = ClippingState(3, torch.device("cpu"))
    first = state.indices([0, 2])
    second = state.indices([1])
    assert state.indices([0, 2]) is first
    assert state.indices([1]) is second
    state.resize(5)
    assert state.indices([0, 2]) is first
    assert state.indices([1, 4]).tolist() == [1, 4]


def test_index_cache_releases_old_active_sets():
    state = ClippingState(32, torch.device("cpu"))
    references = [weakref.ref(state.indices([slot])) for slot in range(32)]

    assert sum(reference() is not None for reference in references) <= 8
    assert state.indices([0]).tolist() == [0]
