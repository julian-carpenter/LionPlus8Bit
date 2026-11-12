"""CPU contracts for the standalone find_unused_parameters integration workload."""

import importlib.util
from pathlib import Path
import sys

import torch


def runner():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "tools"))
    spec = importlib.util.spec_from_file_location(
        "ddp_unused_integration", root / "tools/ddp_unused_integration.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_potential_branch_parameters_start_trainable_and_unused_have_no_gradient():
    module = runner()
    model = module.UnusedBranchProbe()
    assert all(parameter.requires_grad for parameter in model.parameters())
    output = model(torch.ones(4, 64), local_branch=False, shared_branch=False)
    output.square().mean().backward()
    for name, parameter in model.named_parameters():
        assert (parameter.grad is not None) == name.startswith("common.")


def test_shared_branch_activates_together_after_warmup_and_local_branch_can_be_unused():
    module = runner()
    for step in range(12):
        choices = [module.branch_schedule(step, rank) for rank in range(4)]
        assert len({choice["shared_branch"] for choice in choices}) == 1
        if step < 5:
            assert not any(choice["shared_branch"] for choice in choices)
        assert not any(choice["local_branch"] for choice in choices[1:])
    assert module.branch_schedule(5, 0)["shared_branch"]
    assert not module.branch_schedule(7, 0)["shared_branch"]
    assert module.branch_schedule(1, 0)["local_branch"]
    assert not module.branch_schedule(2, 0)["local_branch"]


def test_cpu_late_branch_keeps_history_unseeded_until_first_activation():
    from lionplus8bit import LionPlus8bit

    module = runner()
    torch.manual_seed(91)
    model = module.UnusedBranchProbe()
    optimizer = LionPlus8bit(model.parameters(), clipping="adagc", adagc_warmup_steps=3)
    slots = {name: index for index, (name, _) in enumerate(model.named_parameters())}
    for step in range(6):
        optimizer.zero_grad(set_to_none=True)
        output = model(torch.ones(4, 64) * 0.2, **module.branch_schedule(step, 0))
        output.square().mean().backward()
        optimizer.step()
        for name, parameter in model.named_parameters():
            if name == "dormant" or (name.startswith("shared.") and step < 5):
                assert parameter.grad is None
                assert not optimizer.state.get(parameter)
                assert not bool(optimizer._clipping_state.initialized[slots[name]])
                assert optimizer._clipping_state.history[slots[name]] == 0
    for name, parameter in model.named_parameters():
        if name.startswith("shared."):
            assert optimizer.state[parameter]["step"] == 1
            assert bool(optimizer._clipping_state.initialized[slots[name]])
