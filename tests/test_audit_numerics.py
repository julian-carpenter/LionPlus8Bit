import torch
from unittest.mock import patch
from lionplus8bit import LionPlus8bit

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_tiny_auxiliary_does_not_mark_normal_tensor_strict():
    p = torch.nn.Parameter(torch.ones(4096, device=DEVICE))
    q = torch.nn.Parameter(torch.ones(1, device=DEVICE))
    opt = LionPlus8bit([p, q], clipping="adagc", adagc_warmup_steps=0)
    p.grad = torch.ones_like(p)
    q.grad = torch.full_like(q, 1e-25)
    opt.step()
    assert not opt.state[p].get("strict_math", False)
    assert opt.state[q]["strict_math"]


def test_zero_auxiliary_does_not_recover_unrelated_norm():
    p = torch.nn.Parameter(torch.ones(4096, device=DEVICE))
    q = torch.nn.Parameter(torch.ones(1, device=DEVICE))
    opt = LionPlus8bit([p, q], clipping="adagc", adagc_warmup_steps=0)
    p.grad = torch.ones_like(p)
    q.grad = torch.zeros_like(q)
    original = torch.linalg.vector_norm
    recovered = []

    def norm(x, *args, **kwargs):
        if kwargs.get("dtype") == torch.float64:
            recovered.append(x.numel())
        return original(x, *args, **kwargs)

    with patch.object(torch.linalg, "vector_norm", norm):
        opt.step()
    assert 4096 not in recovered


def test_strict_blocks_preserve_independent_ranges():
    for mode in ("global", "adagc"):
        p = torch.nn.Parameter(torch.zeros(4097, device=DEVICE))
        opt = LionPlus8bit(
            [p],
            clipping=mode,
            adagc_warmup_steps=0,
            max_grad_norm=float("inf"),
            lr=0.1,
            betas=(0.9, 0.9),
            weight_decay=0,
        )
        for t in range(3):
            p.grad = torch.zeros_like(p)
            p.grad[:256] = 1e10
            p.grad[256:512] = 1e-39 if t == 0 else -1e-40 if t == 1 else 0
            opt.step()
            torch.testing.assert_close(
                p[256:512], torch.full_like(p[256:512], -0.1 * (t + 1)), rtol=0, atol=3e-8
            )
        assert opt.state[p]["absmax1"][1] > 0


def test_alias_and_live_dtype_changes_rejected():
    import pytest

    p = torch.nn.Parameter(torch.ones(2, device=DEVICE))
    opt = LionPlus8bit([p], clipping="adagc")
    p.grad = p.detach()
    with pytest.raises(RuntimeError, match="storage"):
        opt.step()
    assert not opt.state
    p.grad = torch.ones_like(p)
    opt.step()
    p.grad = None
    p.data = p.data.to(torch.bfloat16)
    p.grad = torch.ones_like(p)
    with pytest.raises(RuntimeError, match="dtype/device"):
        opt.step()


def test_distinct_overlapping_parameters_rejected_but_disjoint_views_work():
    import pytest

    base = torch.ones(8, device=DEVICE)
    p = torch.nn.Parameter(base[:4])
    q = torch.nn.Parameter(base[2:6])
    opt = LionPlus8bit([p, q])
    p.grad, q.grad = torch.ones_like(p), torch.ones_like(q)
    with pytest.raises(RuntimeError, match="storage ranges"):
        opt.step()
    assert not opt.state
    r = torch.nn.Parameter(base[4:])
    safe = LionPlus8bit([p, r])
    r.grad = torch.ones_like(r)
    safe.step()
