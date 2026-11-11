import math

import pytest
import torch

from lionplus8bit._clipping import ClippingConfig, ClippingState, prepare_clipping, commit_clipping


def run(state, config, norms, slots=None):
    plan = prepare_clipping(
        torch.tensor(norms, dtype=torch.float64),
        slots or list(range(len(norms))),
        config,
        state,
        100.0,
    )
    commit_clipping(state, plan)
    return plan


def test_old_history_and_clipped_ema():
    state = ClippingState(2, torch.device("cpu"))
    config = ClippingConfig("adagc", 0.99, 1.04, 0)
    run(state, config, [2, 10])
    plan = run(state, config, [10, 5])
    assert plan.scales.tolist() == pytest.approx([0.208, 1])
    assert state.history.tolist() == pytest.approx([2.0008, 9.95])


def test_warmup_min_boundary_zero_recovery_and_none():
    state = ClippingState(2, torch.device("cpu"))
    config = ClippingConfig("adagc", 0.99, 1.04, 2)
    run(state, config, [0, 10])
    run(state, config, [2], [0])
    assert state.steps == 2
    assert state.history.tolist() == pytest.approx([2, 10])
    assert run(state, config, [10], [0]).scales.tolist() == pytest.approx([0.208])
    assert state.history[1] == 10


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(beta=1),
        dict(beta=1 - 1e-10),
        dict(factor=0.9),
        dict(factor=math.inf),
        dict(warmup=True),
        dict(warmup=-1),
        dict(mode="oops"),
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        ClippingConfig(**kwargs)


def test_nonfinite_rejected_without_mutation():
    state = ClippingState(1, torch.device("cpu"))
    with pytest.raises(RuntimeError):
        run(state, ClippingConfig("adagc"), [math.inf])
    assert state.steps == 0
    assert state.history.tolist() == [0]


def test_zero_warmup_coefficient_and_extreme_history_decay():
    state = ClippingState(1, torch.device("cpu"))
    config = ClippingConfig("adagc", beta=0, warmup=1)
    plan = prepare_clipping(torch.tensor([0.0]), [0], config, state, 0.1)
    assert plan.scales.tolist() == [1.0]
    commit_clipping(state, plan)
    plan = run(state, config, [1e-40])
    assert state.exceptions[0] > 0
    run(state, config, [0.0])
    assert not state.initialized[0]
    assert run(state, config, [10.0]).scales.tolist() == [1.0]
