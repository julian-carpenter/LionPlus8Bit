import copy
import pytest
import torch
from lionplus8bit import LionPlus8bit
from test_adagc_optimizer import equal

pytestmark = pytest.mark.cuda


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("size", [257, 4096, 4353])
def test_native_replay_masters_and_gradient_preservation(dtype, size):
    torch.manual_seed(28)
    p = torch.nn.Parameter(torch.randn(size, device="cuda", dtype=dtype))
    q = torch.nn.Parameter(p.detach().clone())
    a = LionPlus8bit([p], clipping="adagc", adagc_warmup_steps=1, max_grad_norm=0.4)
    b = LionPlus8bit([q], max_grad_norm=float("inf"), native_gradient_scale=False)
    hist = None
    for t in range(4):
        grad = torch.randn_like(p) * (10 if t == 2 else 0.1)
        norm = float(grad.double().norm())
        scale = min(1.0, (0.4 if t == 0 else 1.04 * hist) / norm)
        scale = float(torch.tensor(scale, dtype=torch.float32))
        seen = norm * scale
        hist = seen if hist is None else 0.99 * hist + 0.01 * seen
        p.grad = grad
        original = grad.clone()
        # masters perform FP32 gradient clipping before update
        q.grad = grad.clone()
        # Inject the independently determined coefficient at the native update boundary.
        original_update = b.update_step

        def update(group, param, gi, pi):
            b._clip_scale = torch.tensor(scale, device="cuda", dtype=torch.float64)
            b._needs_clip = scale < 1
            b._wide_clip = False
            return original_update(group, param, gi, pi)

        b.update_step = update
        a.step()
        b.step()
        b.update_step = original_update
        assert p.grad is grad
        torch.testing.assert_close(grad, original, rtol=0, atol=0)
        torch.testing.assert_close(p, q, rtol=0, atol=2e-6 if dtype == torch.float32 else 0)


def test_amp_overflow_boundary_and_accumulation():
    torch.manual_seed(77)
    a = torch.nn.Linear(8, 4).cuda()
    b = copy.deepcopy(a)
    oa = LionPlus8bit(a.parameters(), clipping="adagc", adagc_warmup_steps=1)
    ob = LionPlus8bit(b.parameters(), clipping="adagc", adagc_warmup_steps=1)
    sa = torch.amp.GradScaler("cuda", init_scale=16)
    sb = torch.amp.GradScaler("cuda", init_scale=16)
    for t in range(4):
        x = torch.randn(8, 8, device="cuda")
        y = torch.randn(8, 4, device="cuda")
        oa.zero_grad()
        ob.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = (a(x) - y).square().mean()
        sa.scale(loss).backward()
        for xx, yy in zip(x.chunk(3 if t == 2 else 2), y.chunk(3 if t == 2 else 2)):
            with torch.autocast("cuda", dtype=torch.float16):
                micro = (b(xx) - yy).square().mean() * (len(xx) / len(x))
            sb.scale(micro).backward()
        if t in (1, 3):
            next(a.parameters()).grad.fill_(float("inf"))
            next(b.parameters()).grad.fill_(float("inf"))
        before = copy.deepcopy(oa.state_dict())
        weights = copy.deepcopy(a.state_dict())
        sa.step(oa)
        sb.step(ob)
        sa.update()
        sb.update()
        if t in (1, 3):
            equal(before, oa.state_dict())
            equal(weights, a.state_dict())
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-5)
    assert oa._clipping_state.steps == 2


def test_nondefault_stream_noncontiguous_grad_and_tiny_guard():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        p = torch.nn.Parameter(torch.ones(4096, device="cuda"))
        opt = LionPlus8bit([p], clipping="adagc", adagc_warmup_steps=0)
        for t in range(3):
            g = torch.randn(4096, 2, device="cuda")[:, 0]
            g[0] = 1e-30
            p.grad = g
            saved = g.clone()
            opt.step()
            torch.testing.assert_close(g, saved, rtol=0, atol=0)
        assert opt.state[p]["strict_math"]
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.isfinite(p).all()


def test_actual_fp16_gradscaler_restriction():
    p = torch.nn.Parameter(torch.ones(2, device="cuda", dtype=torch.float16))
    opt = LionPlus8bit([p], clipping="adagc")
    scaler = torch.amp.GradScaler("cuda")
    scaler.scale(p.float().sum()).backward()
    with pytest.raises(ValueError, match="unscale FP16"):
        scaler.step(opt)
    assert opt._clipping_state is None


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_checkpoint_history_precision_with_half_parameters(dtype):
    p = torch.nn.Parameter(torch.ones(257, device="cuda", dtype=dtype))
    a = LionPlus8bit([p], clipping="adagc", adagc_warmup_steps=1)
    p.grad = torch.ones_like(p)
    a.step()
    q = torch.nn.Parameter(p.detach().clone())
    b = LionPlus8bit([q])
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    assert b._clipping_state.history.dtype == torch.float32
    for _ in range(2):
        p.grad = torch.full_like(p, 0.01)
        q.grad = p.grad.clone()
        a.step()
        b.step()
        equal(p, q)
        equal(a.state_dict(), b.state_dict())
