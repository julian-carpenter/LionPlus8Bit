import pytest
import torch
from lionplus8bit import LionPlus8bit

pytestmark = pytest.mark.cuda


@pytest.mark.parametrize("mode", ["global", "adagc"])
@pytest.mark.parametrize("path", ["native", "no-scale", "master", "strict", "strided"])
def test_stream_allocator_churn_matches_serialized(mode, path):
    dtype = torch.float16 if path == "master" else torch.float32
    p = torch.nn.Parameter(torch.ones(65537, device="cuda", dtype=dtype))
    q = torch.nn.Parameter(p.detach().clone())
    kwargs = dict(
        clipping=mode,
        adagc_warmup_steps=0,
        max_grad_norm=0.5,
        native_gradient_scale=path != "no-scale",
    )
    a, b = LionPlus8bit([p], **kwargs), LionPlus8bit([q], **kwargs)
    stream = torch.cuda.Stream()
    for step in range(4):
        g = torch.full_like(p, 0.1 * (step + 1))
        if path == "strict":
            g[0] = 1e-30
        if path == "strided":
            g = torch.stack([g, g], 1)[:, 0]
        q.grad = g.clone()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            p.grad = g
            a.step()
            p.grad = None
            del g
            # Recycle short-lived allocations while native work runs on default.
            for _ in range(8):
                torch.empty(65536, device="cuda").fill_(17)
        b.step()
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        for key in ("state1", "absmax1", "master_weight"):
            if key in a.state[p]:
                torch.testing.assert_close(a.state[p][key], b.state[q][key], rtol=0, atol=0)


def test_inactive_other_device_is_rejected_before_history_allocation():
    p = torch.nn.Parameter(torch.ones(2, device="cuda"))
    q = torch.nn.Parameter(torch.ones(2, device="cpu"))
    opt = LionPlus8bit([p, q], clipping="adagc")
    p.grad = torch.ones_like(p)
    with pytest.raises(RuntimeError, match="per device"):
        opt.step()
    assert opt._clipping_state is None
    assert not opt.state


@pytest.mark.parametrize("mode", ["global", "adagc"])
@pytest.mark.parametrize("size", [17, 4097])
def test_deferred_checkpoint_state_migrates_before_guard(mode, size):
    import copy

    p = torch.nn.Parameter(torch.ones(size, device="cuda"))
    a = LionPlus8bit([p], clipping=mode, adagc_warmup_steps=0)
    p.grad = torch.ones_like(p)
    a.step()
    saved = copy.deepcopy(a.state_dict())

    # Recreate a CPU-loaded native checkpoint, including bnb's nested state wrapper.
    def cpu(value):
        if isinstance(value, torch.Tensor):
            return value.cpu()
        if isinstance(value, dict):
            return {k: cpu(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cpu(v) for v in value]
        return value

    q = torch.nn.Parameter(p.detach().clone())
    b = LionPlus8bit([q])
    b.load_state_dict(cpu(saved), move_to_device=False)
    assert b.state[q]["state1"].device.type == "cpu"
    q.grad = p.grad.clone()
    a.step()
    b.step()
    torch.testing.assert_close(p, q, rtol=0, atol=0)
    torch.testing.assert_close(a.state[p]["state1"], b.state[q]["state1"], rtol=0, atol=0)


@pytest.mark.parametrize("phase", ["bootstrap", "warmup", "adaptive"])
def test_cuda_minimum_fp32_clipping_bound(phase):
    minimum = 2**-149
    p = torch.nn.Parameter(torch.ones(1, device="cuda"))
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
        p.grad = torch.full_like(p, minimum)
        opt.step()
    p.grad = torch.full_like(p, 49)
    opt.step()
    torch.testing.assert_close(opt.state[p]["state1"], torch.full_like(p, minimum), rtol=0, atol=0)
    assert opt._clipping_state.exceptions[0] == minimum
