"""Paging safety contract: ordinary storage is never relabelled as managed memory."""

import copy
import io
from types import SimpleNamespace

import pytest
import torch

from lionplus8bit import LionPlus8bit


def saved_momentum(checkpoint):
    state = checkpoint["state"][0]
    return state.get("__bnb_optimizer_quant_state__", state)["state1"]


@pytest.mark.parametrize("count", [99_999, 100_000, 100_001, 100_257])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
def test_cpu_paging_policy_has_plain_contiguous_storage(count, dtype):
    from lionplus8bit._paging import allocate_state

    template = torch.empty((count, 1)).t()
    state = allocate_state(template, dtype, is_paged=True)
    assert state.shape == template.shape
    assert state.dtype == dtype
    assert state.device.type == "cpu"
    assert state.is_contiguous()
    assert not getattr(state, "is_paged", False)
    assert torch.count_nonzero(state) == 0


@pytest.mark.parametrize("count", [100_000, 100_001, 100_257])
@pytest.mark.parametrize("ordinal", [0, 1, 2, 3])
def test_managed_eligible_request_fails_without_touching_cuda(count, ordinal):
    from lionplus8bit._paging import PagingUnsupportedError, validate_paging_request

    parameter = SimpleNamespace(device=torch.device("cuda", ordinal), numel=lambda: count)
    with pytest.raises(PagingUnsupportedError, match="storage deleter"):
        validate_paging_request([parameter], is_paged=True)


def test_ineligible_and_nonpaged_requests_do_not_require_cuda():
    from lionplus8bit._paging import validate_paging_request

    small = SimpleNamespace(device=torch.device("cuda", 3), numel=lambda: 99_999)
    large = SimpleNamespace(device=torch.device("cuda", 3), numel=lambda: 100_001)
    validate_paging_request([small], is_paged=True)
    validate_paging_request([large], is_paged=False)


def test_paging_capability_explicitly_reports_ownership_blocker():
    from lionplus8bit._paging import paging_capability

    capability = paging_capability()
    assert capability["managed_supported"] is False
    assert capability["min_managed_elements"] == 100_000
    assert "storage deleter" in capability["reason"]
    assert capability["reclamation_supported"] is False
    assert capability["managed_allocation_attempted"] is False


def test_checkpoint_values_remove_legacy_metadata_without_mutating_input():
    from lionplus8bit._paging import checkpoint_values

    source = torch.arange(12, dtype=torch.float32)
    source.is_paged = True
    source.page_deviceid = 3
    source.is_managed = True
    checkpoint = {"state": {0: {"state1": source}}, "nested": [(source,)]}
    saved = checkpoint_values(checkpoint)
    result = saved["state"][0]["state1"]
    assert not hasattr(result, "is_paged")
    assert not hasattr(result, "page_deviceid")
    assert not hasattr(result, "is_managed")
    assert result.data_ptr() != source.data_ptr()
    assert source.is_paged and source.page_deviceid == 3
    torch.testing.assert_close(result, source)
    result.zero_()
    assert source[-1] == 11
    assert not hasattr(saved["nested"][0][0], "is_paged")


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("version", [2, 3])
def test_legacy_paging_metadata_is_not_restored_as_managed_storage(deferred, version):
    parameter = torch.nn.Parameter(torch.arange(16, dtype=torch.float32))
    optimizer = LionPlus8bit([parameter], is_paged=True)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    checkpoint = copy.deepcopy(optimizer.state_dict())
    checkpoint["lionplus_config"]["version"] = version
    # Layout v3 will be supplied by Task 2. A legacy v2 has no state-order contract.
    if version == 2:
        checkpoint["lionplus_config"].pop("state_order", None)
        checkpoint["lionplus_config"].pop("parameter_shapes", None)
    saved = saved_momentum(checkpoint)
    saved.is_paged, saved.page_deviceid = True, 3
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=False)
    target = torch.nn.Parameter(parameter.detach().clone())
    resumed = LionPlus8bit([target])
    resumed.load_state_dict(checkpoint, move_to_device=not deferred)
    restored = resumed.state[target]["state1"]
    assert not hasattr(restored, "is_paged")
    assert not hasattr(restored, "page_deviceid")
    assert resumed.is_paged is True
    for _ in range(3):
        parameter.grad = torch.full_like(parameter, -0.2)
        target.grad = parameter.grad.clone()
        optimizer.step()
        resumed.step()
    torch.testing.assert_close(target, parameter, atol=0, rtol=0)
    torch.testing.assert_close(
        resumed.state[target]["state1"], optimizer.state[parameter]["state1"]
    )


def test_optimizer_state_dict_takes_value_snapshot_of_legacy_paging_state():
    parameter = torch.nn.Parameter(torch.ones(16))
    optimizer = LionPlus8bit([parameter], is_paged=True)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    momentum = optimizer.state[parameter]["state1"]
    momentum.is_paged, momentum.page_deviceid = True, 0
    checkpoint = optimizer.state_dict()
    saved = saved_momentum(checkpoint)
    assert not hasattr(saved, "is_paged")
    assert saved.data_ptr() != momentum.data_ptr()
    assert momentum.is_paged


@pytest.mark.parametrize("count", [99_999, 100_000, 100_001, 100_257])
@pytest.mark.parametrize("bits", [8, 32])
@pytest.mark.parametrize("clipping", ["global", "adagc"])
def test_cpu_policy_continuation_at_native_threshold(count, bits, clipping):
    parameter = torch.nn.Parameter(torch.linspace(-1, 1, count))
    control = torch.nn.Parameter(parameter.detach().clone())
    options = dict(
        clipping=clipping,
        min_8bit_size=0 if bits == 8 else count + 1,
        adagc_warmup_steps=0,
        lr=0.01,
    )
    paged = LionPlus8bit([parameter], is_paged=True, **options)
    ordinary = LionPlus8bit([control], **options)
    for gradient in [0.2, 1e-35, -0.2]:
        parameter.grad = torch.full_like(parameter, gradient)
        control.grad = parameter.grad.clone()
        paged.step()
        ordinary.step()
        torch.testing.assert_close(parameter, control, atol=0, rtol=0)
        torch.testing.assert_close(
            paged.state[parameter]["state1"], ordinary.state[control]["state1"]
        )
        assert not getattr(paged.state[parameter]["state1"], "is_paged", False)
    assert paged.state[parameter]["state1"].dtype == (torch.uint8 if bits == 8 else torch.float32)
    resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
    resumed = LionPlus8bit([resumed_parameter])
    resumed.load_state_dict(copy.deepcopy(paged.state_dict()), move_to_device=False)
    parameter.grad = torch.full_like(parameter, -0.3)
    resumed_parameter.grad = parameter.grad.clone()
    paged.step()
    resumed.step()
    torch.testing.assert_close(parameter, resumed_parameter, atol=0, rtol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("count", [100_000, 100_001, 100_257])
def test_cuda_managed_gate_prevents_partial_update_and_manager_growth(count):
    import bitsandbytes as bnb
    from lionplus8bit._paging import PagingUnsupportedError

    small = torch.nn.Parameter(torch.ones(8, device="cuda"))
    large = torch.nn.Parameter(torch.ones(count, device="cuda"))
    optimizer = LionPlus8bit([small, large], is_paged=True)
    for parameter in [small, large]:
        parameter.grad = torch.ones_like(parameter)
    manager = bnb.functional.GlobalPageManager.get_instance()
    owned = len(manager.paged_tensors)
    with pytest.raises(PagingUnsupportedError, match="storage deleter"):
        optimizer.step()
    assert not optimizer.state
    assert len(manager.paged_tensors) == owned
    torch.testing.assert_close(small, torch.ones_like(small), atol=0, rtol=0)
    torch.testing.assert_close(large, torch.ones_like(large), atol=0, rtol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("deferred", [False, True])
def test_cuda_checkpoint_cannot_restore_paging_flag_as_allocation(deferred):
    from lionplus8bit._paging import PagingUnsupportedError

    source = torch.nn.Parameter(torch.ones(100_001))
    source_optimizer = LionPlus8bit([source], is_paged=True)
    source.grad = torch.ones_like(source)
    source_optimizer.step()
    checkpoint = copy.deepcopy(source_optimizer.state_dict())
    saved_momentum(checkpoint).is_paged = True
    saved_momentum(checkpoint).page_deviceid = 0
    target = torch.nn.Parameter(source.detach().cuda())
    target_optimizer = LionPlus8bit([target])
    if deferred:
        target_optimizer.load_state_dict(checkpoint, move_to_device=False)
        assert not hasattr(target_optimizer.state[target]["state1"], "is_paged")
        target.grad = torch.ones_like(target)
        with pytest.raises(PagingUnsupportedError, match="storage deleter"):
            target_optimizer.step()
        assert target_optimizer.state[target]["step"] == 1
    else:
        with pytest.raises(PagingUnsupportedError, match="storage deleter"):
            target_optimizer.load_state_dict(checkpoint)
        assert not target_optimizer.state
        assert target_optimizer.is_paged is False
    torch.testing.assert_close(target.cpu(), source, atol=0, rtol=0)


def test_paging_integration_report_labels_unrun_managed_cases(tmp_path):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from _paging_integration import run_paging_suite

    result = run_paging_suite(tmp_path)
    assert result["status"] == "unsupported"
    assert result["ordinary_storage_controls"]["status"] == "passed"
    assert result["managed_numerics"]["status"] == "unsupported"
    assert result["pressure_stress"]["status"] == "skipped"
    assert result["managed_allocation_attempted"] is False
    assert (tmp_path / "paging_native_evidence.json").exists()
