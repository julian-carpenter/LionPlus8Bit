"""Logical state ordering and original storage ownership for supported weight layouts."""

import copy
import os

import pytest
import torch

from lionplus8bit import LionPlus8bit


@pytest.fixture(autouse=True)
def layout_test_device():
    # The coordinator opts into GPU validation explicitly; CPU CI stays portable.
    with torch.device(os.environ.get("LIONPLUS_LAYOUT_DEVICE", "cpu")):
        yield


def layout_value(kind, dtype=torch.float32):
    if kind == "channels_last":
        return (
            torch.linspace(-1, 1, 17 * 31 * 3 * 3, dtype=dtype)
            .reshape(17, 31, 3, 3)
            .to(memory_format=torch.channels_last)
        )
    if kind == "channels_last_3d":
        return (
            torch.linspace(-1, 1, 7 * 23 * 27, dtype=dtype)
            .reshape(7, 23, 3, 3, 3)
            .to(memory_format=torch.channels_last_3d)
        )
    if kind == "transpose":
        return torch.linspace(-1, 1, 4097, dtype=dtype).reshape(17, 241).t()
    if kind == "permutation":
        return torch.linspace(-1, 1, 4097, dtype=dtype).reshape(17, 1, 241).permute(2, 0, 1)
    if kind == "size_one":
        return torch.linspace(-1, 1, 4097, dtype=dtype).reshape(1, 17, 1, 241).permute(0, 3, 2, 1)
    raise AssertionError(kind)


def assert_state_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], torch.Tensor):
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        else:
            assert left[key] == right[key]


@pytest.mark.parametrize(
    "kind", ["channels_last", "channels_last_3d", "transpose", "permutation", "size_one"]
)
@pytest.mark.parametrize("mode", ["global", "adagc"])
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_dense_layout_matches_logical_contiguous_updates(kind, mode, quantized, dtype):
    p = torch.nn.Parameter(layout_value(kind, dtype))
    q = torch.nn.Parameter(p.detach().contiguous())
    options = dict(
        lr=0.01,
        betas=(0.6, 0.8),
        clipping=mode,
        adagc_warmup_steps=0,
        max_grad_norm=0.5,
        min_8bit_size=0 if quantized else p.numel() + 1,
    )
    actual, reference = LionPlus8bit([p], **options), LionPlus8bit([q], **options)
    storage = (p.data_ptr(), p.stride(), p.storage_offset(), p.shape)
    for scale in (0.001, 10.0, 1e-35, 0.02):
        # The tiny case must enter strict arithmetic even for low precision weights.
        if dtype == torch.float16 and scale == 1e-35:
            scale = 0.0
        g = torch.linspace(-scale, scale, p.numel(), dtype=dtype).reshape(p.shape)
        p.grad = g.clone(memory_format=torch.preserve_format)
        q.grad = g.clone()
        original, snapshot = p.grad, p.grad.clone()
        actual.step()
        reference.step()
        assert (p.data_ptr(), p.stride(), p.storage_offset(), p.shape) == storage
        assert p.grad is original
        torch.testing.assert_close(p.grad, snapshot, rtol=0, atol=0)
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        assert_state_equal(actual.state[p], reference.state[q])
    state = actual.state[p]
    assert state["state1"].is_contiguous()
    assert state["state1"].dtype == (torch.uint8 if quantized else torch.float32)
    if dtype == torch.float32:
        assert "master_weight" not in state
    else:
        assert state["master_weight"].dtype == torch.float32
        assert state["master_weight"].is_contiguous()
    if mode == "adagc":
        torch.testing.assert_close(
            actual._clipping_state.history, reference._clipping_state.history, rtol=0, atol=0
        )


@pytest.mark.parametrize("kind", ["expanded", "overlapping", "stepped", "shared"])
def test_unsafe_layouts_fail_before_any_weight_or_state_update(kind):
    base = torch.arange(24, dtype=torch.float32)
    if kind == "expanded":
        values = [base[:1].expand(4)]
    elif kind == "overlapping":
        values = [base.as_strided((3, 3), (1, 1))]
    elif kind == "stepped":
        values = [base[::2]]
    else:
        values = [base[:12].view(3, 4).t(), base[8:20].view(3, 4).t()]
    parameters = [torch.nn.Parameter(v) for v in values]
    snapshots = [p.detach().clone() for p in parameters]
    optimizer = LionPlus8bit(parameters)
    for p in parameters:
        p.grad = torch.ones_like(p)
    with pytest.raises(RuntimeError, match="overlap|dense|contiguous|slice"):
        optimizer.step()
    assert not optimizer.state
    for p, before in zip(parameters, snapshots):
        torch.testing.assert_close(p, before, rtol=0, atol=0)


def test_disjoint_dense_views_preserve_storage_offset_and_tied_identity():
    base = torch.arange(24, dtype=torch.float32)
    p, q = torch.nn.Parameter(base[:12].view(3, 4).t()), torch.nn.Parameter(base[12:])
    tied = p
    pointers = [(v.data_ptr(), v.stride(), v.storage_offset()) for v in (p, q)]
    optimizer = LionPlus8bit([p, q], lr=0.125, weight_decay=0)
    for v in (p, q):
        v.grad = torch.ones_like(v)
    optimizer.step()
    assert p is tied
    assert [(v.data_ptr(), v.stride(), v.storage_offset()) for v in (p, q)] == pointers
    torch.testing.assert_close(base, torch.arange(24, dtype=torch.float32) - 0.125)


@pytest.mark.parametrize("change", ["stride", "shape", "offset"])
def test_initialized_layout_changes_are_rejected_before_update(change):
    p = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).view(3, 4))
    optimizer = LionPlus8bit([p])
    p.grad = torch.ones_like(p)
    optimizer.step()
    if change == "stride":
        p.data = p.detach().t().contiguous().t()
    elif change == "shape":
        p.data = p.detach().view(2, 6)
    else:
        p.data = torch.cat([torch.zeros(1), p.detach().flatten()])[1:].view(3, 4)
    p.grad = torch.ones_like(p)
    before = p.detach().clone()
    step = optimizer.state[p]["step"]
    with pytest.raises(RuntimeError, match="layout|fresh optimizer"):
        optimizer.step()
    assert optimizer.state[p]["step"] == step
    torch.testing.assert_close(p, before, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["global", "adagc"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_checkpoint_v3_continues_across_weight_layouts(mode, reverse, deferred, quantized, dtype):
    value = layout_value("transpose", dtype)
    p = torch.nn.Parameter(value.contiguous() if reverse else value)
    source = LionPlus8bit(
        [p], clipping=mode, adagc_warmup_steps=0, min_8bit_size=0 if quantized else 10000
    )
    p.grad = torch.linspace(-2, 2, p.numel(), dtype=p.dtype).reshape(p.shape)
    source.step()
    saved = copy.deepcopy(source.state_dict())
    assert saved["lionplus_config"]["version"] == 3
    assert saved["lionplus_config"]["state_order"] == "logical-row-major-v1"
    q_value = p.detach().t().contiguous().t() if reverse else p.detach().contiguous()
    q = torch.nn.Parameter(q_value)
    resumed = LionPlus8bit([q])
    resumed.load_state_dict(saved, move_to_device=not deferred)
    for scale in (0.1, 1e-35, 2.0):
        g = torch.linspace(scale, -scale, p.numel(), dtype=p.dtype).reshape(p.shape)
        p.grad, q.grad = g.clone(), g.clone()
        source.step()
        resumed.step()
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        assert_state_equal(source.state[p], resumed.state[q])


@pytest.mark.parametrize("mode", ["global", "adagc"])
def test_checkpoint_shape_mapping_is_validated_before_mutation(mode):
    p = torch.nn.Parameter(torch.ones(3, 4))
    source = LionPlus8bit([p], clipping=mode)
    p.grad = torch.ones_like(p)
    source.step()
    saved = copy.deepcopy(source.state_dict())
    q = torch.nn.Parameter(torch.ones(2, 6))
    target = LionPlus8bit([q])
    with pytest.raises(ValueError, match="mapping"):
        target.load_state_dict(saved)
    assert not target.state
    assert target.clipping == "global"


@pytest.mark.parametrize("version,mode", [(1, "global"), (2, "global"), (2, "adagc")])
def test_legacy_contiguous_checkpoint_can_resume_into_dense_layout(version, mode):
    p = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4))
    source = LionPlus8bit([p], clipping=mode)
    p.grad = torch.ones_like(p)
    source.step()
    saved = copy.deepcopy(source.state_dict())
    saved["lionplus_config"]["version"] = version
    saved["lionplus_config"].pop("state_order", None)
    saved["lionplus_config"].pop("parameter_shapes", None)
    q = torch.nn.Parameter(p.detach().t().contiguous().t())
    target = LionPlus8bit([q])
    target.load_state_dict(saved)
    q.grad = torch.ones_like(q)
    source.step()
    target.step()
    torch.testing.assert_close(p, q, rtol=0, atol=0)
    assert_state_equal(source.state[p], target.state[q])


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("mode", ["global", "adagc"])
def test_channels_last_model_forward_backward_and_step_match_contiguous(dimension, mode):
    conv = torch.nn.Conv2d if dimension == 2 else torch.nn.Conv3d
    memory_format = torch.channels_last if dimension == 2 else torch.channels_last_3d
    model = conv(2, 3, 3, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.125)
    control = copy.deepcopy(model)
    model.to(memory_format=memory_format)
    original_weight = model.weight
    storage = (original_weight.data_ptr(), original_weight.stride())
    actual = LionPlus8bit(model.parameters(), clipping=mode, adagc_warmup_steps=0, min_8bit_size=0)
    reference = LionPlus8bit(
        control.parameters(), clipping=mode, adagc_warmup_steps=0, min_8bit_size=0
    )
    inputs = torch.ones((2, 2, *([5] * dimension)))
    for _ in range(3):
        actual.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        output, expected = model(inputs.to(memory_format=memory_format)), control(inputs)
        torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-7)
        output.square().mean().backward()
        expected.square().mean().backward()
        original_grad, before = model.weight.grad, model.weight.grad.clone()
        torch.testing.assert_close(original_grad, control.weight.grad, rtol=1e-6, atol=1e-7)
        actual.step()
        reference.step()
        assert model.weight is original_weight
        assert (model.weight.data_ptr(), model.weight.stride()) == storage
        assert model.weight.grad is original_grad
        torch.testing.assert_close(model.weight.grad, before, rtol=0, atol=0)
        torch.testing.assert_close(model.weight, control.weight, rtol=0, atol=0)


@pytest.mark.parametrize(
    "field,value", [("state_order", "physical-strides"), ("parameter_shapes", [])]
)
def test_checkpoint_v3_rejects_invalid_order_and_mapping_atomically(field, value):
    source = LionPlus8bit([torch.nn.Parameter(torch.ones(3, 4))])
    saved = copy.deepcopy(source.state_dict())
    saved["lionplus_config"][field] = value
    target = LionPlus8bit([torch.nn.Parameter(torch.ones(3, 4))])
    with pytest.raises(ValueError, match="order|mapping"):
        target.load_state_dict(saved)
    assert not target.state
    assert not target.initialized


@pytest.mark.cuda
@pytest.mark.parametrize("quantized", [False, True])
def test_nondefault_stream_staging_survives_allocator_churn(quantized):
    device = torch.device("cuda")
    stream = torch.cuda.Stream(device=device)
    p = torch.nn.Parameter(layout_value("transpose").to(device))
    q = torch.nn.Parameter(p.detach().contiguous())
    options = dict(lr=0.01, max_grad_norm=0.5, min_8bit_size=0 if quantized else 10000)
    actual, reference = LionPlus8bit([p], **options), LionPlus8bit([q], **options)
    stream.wait_stream(torch.cuda.current_stream(device))
    pointer = p.data_ptr()
    for scale in (0.001, 10.0, 1e-35, 0.2):
        with torch.cuda.stream(stream):
            g = (
                torch.linspace(-scale, scale, p.numel(), device=device)
                .reshape(tuple(reversed(p.shape)))
                .t()
            )
            p.grad = g
            before = g.clone()
            actual.step()
            # Reuse temporary allocation sizes immediately while the default
            # stream's native work may still be outstanding.
            churn = [torch.empty_like(p, memory_format=torch.contiguous_format) for _ in range(8)]
            for buffer in churn:
                buffer.fill_(123)
            torch.testing.assert_close(p.grad, before, rtol=0, atol=0)
        torch.cuda.current_stream(device).wait_stream(stream)
        q.grad = g.contiguous()
        reference.step()
        torch.cuda.synchronize(device)
        assert p.data_ptr() == pointer
        assert p.grad is g
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        assert_state_equal(actual.state[p], reference.state[q])


@pytest.mark.cuda
def test_fp32_layout_scratch_is_released_and_bounded_per_tensor():
    device = torch.device("cuda")
    parameters = [torch.nn.Parameter(torch.ones((1024, 512), device=device).t()) for _ in range(4)]
    optimizer = LionPlus8bit(parameters, max_grad_norm=float("inf"))
    for p in parameters:
        p.grad = torch.ones_like(p)
    optimizer.step()
    torch.cuda.synchronize(device)
    before = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.step()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) - before
    after = torch.cuda.memory_allocated(device)
    one_weight = parameters[0].numel() * parameters[0].element_size()
    # Four staged weights retained together would require at least 4x bytes;
    # per-weight work plus a contiguous gradient requires 2x, with small norms.
    assert peak < 3 * one_weight
    assert after - before < one_weight // 16
    assert all("master_weight" not in optimizer.state[p] for p in parameters)


def test_native_override_and_checkpoint_hooks_preserve_dense_layout_trajectory():
    import bitsandbytes as bnb

    p = torch.nn.Parameter(layout_value("transpose"))
    manager = bnb.optim.GlobalOptimManager.get_instance()
    manager.override_config(p, key_value_dict={"optim_bits": 32, "lr": 0.125})
    source = LionPlus8bit([p], weight_decay=0)
    p.grad = torch.ones_like(p)
    source.step()
    assert source.state[p]["state1"].dtype == torch.float32
    expected = layout_value("transpose") - 0.125
    torch.testing.assert_close(p, expected, rtol=0, atol=0)
    events = []
    source.register_state_dict_pre_hook(lambda opt: events.append("save-pre"))

    def saved(opt, state):
        assert state["lionplus_config"]["state_order"] == "logical-row-major-v1"
        events.append("save-post")

    source.register_state_dict_post_hook(saved)
    checkpoint = copy.deepcopy(source.state_dict())
    q = torch.nn.Parameter(p.detach().contiguous())
    resumed = LionPlus8bit([q])
    resumed.register_load_state_dict_pre_hook(lambda opt, state: events.append("load-pre"))

    def loaded(opt):
        assert opt.state[q]["state1"].is_contiguous()
        events.append("load-post")

    resumed.register_load_state_dict_post_hook(loaded)
    resumed.load_state_dict(checkpoint)
    assert events == ["save-pre", "save-post", "load-pre", "load-post"]
    q.grad = torch.ones_like(q)
    source.step()
    resumed.step()
    torch.testing.assert_close(p, q, rtol=0, atol=0)
    assert_state_equal(source.state[p], resumed.state[q])
