import copy
import math

import pytest
import torch
from lionplus8bit import LionPlus8bit


def equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def test_cpu_lion_equations_and_local_spike():
    ps = [
        torch.nn.Parameter(torch.tensor([1.0, 2.0])),
        torch.nn.Parameter(torch.tensor([3.0, 4.0])),
    ]
    ref = [p.detach().clone() for p in ps]
    moments = [torch.zeros_like(p) for p in ps]
    history = [None, None]
    opt = LionPlus8bit(
        ps,
        clipping="adagc",
        adagc_warmup_steps=2,
        max_grad_norm=1.0,
        lr=0.1,
        betas=(0.7, 0.9),
        weight_decay=0.02,
    )
    for t, magnitudes in enumerate([(2.0, 4.0), (1.0, 3.0), (30.0, 0.2), (0.1, 0.1)]):
        gs = [torch.tensor([x, -x]) for x in magnitudes]
        ns = [float(g.double().norm()) for g in gs]
        if t < 2:
            scales = [min(1.0, 1 / math.sqrt(sum(n * n for n in ns)))] * 2
        else:
            scales = [min(1.0, 1.04 * h / n) for h, n in zip(history, ns)]
        for i, (p, g, n, h) in enumerate(zip(ps, gs, ns, scales)):
            clipped = g * h
            ref[i].mul_(1 - 0.1 * 0.02).add_((moments[i] * 0.7 + clipped * 0.3).sign(), alpha=-0.1)
            moments[i].mul_(0.9).add_(clipped, alpha=0.1)
            seen = n * h
            history[i] = (
                seen
                if history[i] is None
                else min(history[i], seen)
                if t < 2
                else 0.99 * history[i] + 0.01 * seen
            )
            p.grad = g
        opt.step()
        for i, p in enumerate(ps):
            torch.testing.assert_close(p, ref[i], rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(opt.state[p]["state1"], moments[i], rtol=1e-6, atol=1e-6)
        assert opt._clipping_state.history.tolist() == pytest.approx(history, rel=2e-6)


def test_none_zero_added_groups_and_nonfinite_atomicity():
    p = torch.nn.Parameter(torch.ones(3))
    q = torch.nn.Parameter(torch.ones(2))
    opt = LionPlus8bit([p], clipping="adagc", adagc_warmup_steps=1)
    opt.step()
    assert opt._clipping_state is None
    p.grad = torch.zeros_like(p)
    opt.step()
    assert not opt._clipping_state.initialized[0]
    opt.add_param_group({"params": [q]})
    p.grad = None
    q.grad = torch.ones_like(q)
    opt.step()
    assert opt._clipping_state.steps == 2
    assert opt.state[p]["step"] == 1
    before = copy.deepcopy(opt.state_dict())
    saved = p.clone()
    p.grad = torch.ones_like(p)
    q.grad = torch.full_like(q, math.nan)
    with pytest.raises(RuntimeError):
        opt.step()
    equal(before, opt.state_dict())
    torch.testing.assert_close(p, saved)
    with pytest.raises(ValueError):
        opt.add_param_group({"params": [torch.nn.Parameter(torch.ones(1))], "adagc_factor": 2})


@pytest.mark.parametrize("split", [1, 2, 3])
def test_checkpoint_roundtrip_and_saved_policy(split):
    p = torch.nn.Parameter(torch.ones(5))
    opt = LionPlus8bit([p], clipping="adagc", adagc_warmup_steps=2)
    for _ in range(split):
        p.grad = torch.ones_like(p)
        opt.step()
    saved = copy.deepcopy(opt.state_dict())
    q = torch.nn.Parameter(p.detach().clone())
    other = LionPlus8bit([q])
    other.load_state_dict(saved)
    assert other.clipping == "adagc"
    p.grad = torch.full_like(p, 10)
    q.grad = p.grad.clone()
    opt.step()
    other.step()
    equal(opt.state_dict(), other.state_dict())
    equal(p, q)
    bad = copy.deepcopy(saved)
    bad["lionplus_clipping"]["shapes"] = [(99,)]
    before = copy.deepcopy(other.state_dict())
    with pytest.raises(ValueError):
        other.load_state_dict(bad)
    equal(before, other.state_dict())


def test_legacy_loading_and_tiny_history():
    p = torch.nn.Parameter(torch.ones(1))
    opt = LionPlus8bit([p], clipping="adagc", adagc_warmup_steps=0)
    p.grad = torch.tensor([1e-40])
    opt.step()
    assert 0 in opt._clipping_state.exceptions
    saved = copy.deepcopy(opt.state_dict())
    other = LionPlus8bit([torch.nn.Parameter(torch.ones(1))])
    other.load_state_dict(saved)
    assert other._clipping_state.exceptions == opt._clipping_state.exceptions
    saved.pop("lionplus_clipping")
    saved["lionplus_config"]["version"] = 1
    for key in ["clipping", "adagc_beta", "adagc_factor", "adagc_warmup_steps", "edge_policy"]:
        saved["lionplus_config"].pop(key, None)
    other.load_state_dict(saved)
    assert other.clipping == "global"


def test_added_group_checkpoint_before_next_update_and_edge_policy_validation():
    p = torch.nn.Parameter(torch.ones(3))
    opt = LionPlus8bit([p], clipping="adagc")
    p.grad = torch.ones_like(p)
    opt.step()
    opt.add_param_group({"params": [torch.nn.Parameter(torch.ones(2))]})
    saved = copy.deepcopy(opt.state_dict())
    q = torch.nn.Parameter(torch.ones(3))
    other = LionPlus8bit([q])
    other.add_param_group({"params": [torch.nn.Parameter(torch.ones(2))]})
    other.load_state_dict(saved)
    assert other._clipping_state.history.numel() == 2
    before = copy.deepcopy(other.state_dict())
    bad = copy.deepcopy(saved)
    bad["lionplus_config"]["edge_policy"] = "unknown"
    with pytest.raises(ValueError):
        other.load_state_dict(bad)
    equal(before, other.state_dict())
    bad = copy.deepcopy(saved)
    bad.pop("lionplus_clipping")
    with pytest.raises(ValueError):
        other.load_state_dict(bad)
    bad = copy.deepcopy(saved)
    bad["lionplus_clipping"]["history"][0] = float("nan")
    with pytest.raises(ValueError):
        other.load_state_dict(bad)


@pytest.mark.parametrize(
    "field,value", [("max_grad_norm", float("nan")), ("lr", -1), ("betas", (0.9, 1))]
)
def test_checkpoint_invalid_group_rejected_before_mutation(field, value):
    p = torch.nn.Parameter(torch.ones(3))
    opt = LionPlus8bit([p], clipping="adagc")
    p.grad = torch.ones_like(p)
    opt.step()
    before = copy.deepcopy(opt.state_dict())
    saved = copy.deepcopy(before)
    saved["param_groups"][0][field] = value
    saved["lionplus_clipping"]["steps"] = 999
    with pytest.raises(ValueError):
        opt.load_state_dict(saved)
    equal(before, opt.state_dict())
