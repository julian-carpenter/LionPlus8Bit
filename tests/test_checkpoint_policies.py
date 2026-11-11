"""Regression coverage for effective native policies and complete checkpoint hooks."""

import copy

import bitsandbytes as bnb
import pytest
import torch

from lionplus8bit import LionPlus8bit


def parameter(shape=(1,)):
    return torch.nn.Parameter(torch.ones(shape))


def test_loaded_override_ownership_and_explicit_later_changes():
    manager = bnb.optim.GlobalOptimManager.get_instance()
    p = parameter()
    manager.override_config(p, "lr", 0.125)
    source = LionPlus8bit([p], weight_decay=0)
    checkpoint = copy.deepcopy(source.state_dict())
    q, r = parameter(), parameter()
    a, b = LionPlus8bit([q]), LionPlus8bit([r])
    a.load_state_dict(checkpoint)
    b.load_state_dict(checkpoint)
    manager.override_config(q, "lr", 0.5)
    for optimizer, value in ((a, q), (b, r), (source, p)):
        value.grad = torch.ones_like(value)
        optimizer.step()
    assert checkpoint["lionplus_config"]["overrides"][0][0]["lr"] == 0.125
    torch.testing.assert_close(q, torch.tensor([0.5]), rtol=0, atol=0)
    torch.testing.assert_close(r, torch.tensor([0.875]), rtol=0, atol=0)
    torch.testing.assert_close(p, torch.tensor([0.875]), rtol=0, atol=0)


@pytest.mark.parametrize("destination", ["index", "module"])
@pytest.mark.parametrize("saved_lr", [None, 0.125])
def test_saved_overrides_win_until_an_explicit_later_change(destination, saved_lr):
    manager = bnb.optim.GlobalOptimManager.get_instance()
    p = parameter((1, 1))
    if saved_lr is not None:
        manager.override_config(p, "lr", saved_lr)
    source = LionPlus8bit([p], lr=0.1, weight_decay=0, clipping="adagc")
    checkpoint = copy.deepcopy(source.state_dict())
    module = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        module.weight.fill_(1)
    if destination == "index":
        manager.override_config(module.weight, "lr", 0.7)
        manager.register_parameters([module.weight])
    else:
        manager.register_module_override(module, "weight", {"lr": 0.7})
    resumed = LionPlus8bit([module.weight])
    resumed.load_state_dict(checkpoint)
    expected_lr = 0.1 if saved_lr is None else saved_lr
    assert resumed.get_config(0, 0, resumed.param_groups[0])["lr"] == expected_lr
    module.weight.grad = torch.ones_like(module.weight)
    resumed.step()
    torch.testing.assert_close(module.weight, torch.tensor([[1 - expected_lr]]))
    # Repeating the destination's pre-load value must still be a deliberate new override.
    manager.override_config(module.weight, "lr", 0.7)
    resumed.step()
    torch.testing.assert_close(module.weight, torch.tensor([[0.3 - expected_lr]]))


def test_checkpoint_serializes_the_merged_effective_native_override():
    manager = bnb.optim.GlobalOptimManager.get_instance()
    registered, p = parameter(), parameter()
    manager.override_config(registered, "lr", 0.7)
    manager.register_parameters([registered])
    manager.override_config(p, "optim_bits", 32)
    source = LionPlus8bit([p], lr=0.1, weight_decay=0)
    saved = copy.deepcopy(source.state_dict())
    assert saved["lionplus_config"]["overrides"] == [[{"lr": 0.7, "optim_bits": 32}]]
    # The restored policy must not depend on a process's unrelated index registrations.
    manager.index2config.clear()
    q = parameter()
    resumed = LionPlus8bit([q])
    resumed.load_state_dict(saved)
    for optimizer, value in ((source, p), (resumed, q)):
        value.grad = torch.ones_like(value)
        optimizer.step()
        torch.testing.assert_close(value, torch.tensor([0.3]))


@pytest.mark.parametrize("mode", ["global", "adagc"])
def test_added_stable_embedding_policy_survives_lazy_checkpoint(mode):
    p = parameter()
    source = LionPlus8bit([p], clipping=mode, weight_decay=0)
    p.grad = torch.ones_like(p)
    source.step()
    embedding = bnb.nn.StableEmbedding(64, 128)
    source.add_param_group({"params": [embedding.weight]})
    saved = copy.deepcopy(source.state_dict())
    assert saved["lionplus_config"]["overrides"][1][0]["optim_bits"] == 32
    if mode == "adagc":
        assert source._clipping_state.initialized.tolist() == [True, False]
    q = torch.nn.Parameter(p.detach().clone())
    r = torch.nn.Parameter(embedding.weight.detach().clone())
    resumed = LionPlus8bit([{"params": [q]}, {"params": [r]}])
    resumed.load_state_dict(saved)
    p.grad = q.grad = None
    for optimizer, value in ((source, embedding.weight), (resumed, r)):
        value.grad = torch.ones_like(value)
        optimizer.step()
        assert optimizer.state[value]["state1"].dtype == torch.float32
    torch.testing.assert_close(embedding.weight, r, rtol=0, atol=0)
    if mode == "adagc":
        torch.testing.assert_close(source._clipping_state.history, resumed._clipping_state.history)


def test_save_hooks_run_once_in_order_around_complete_wrapper_metadata():
    p = parameter()
    optimizer = LionPlus8bit([p], clipping="adagc")
    p.grad = torch.ones_like(p)
    optimizer.step()
    events = []
    optimizer.register_state_dict_pre_hook(lambda opt: events.append("pre-last"))
    optimizer.register_state_dict_pre_hook(lambda opt: events.append("pre-first"), prepend=True)

    def post_first(opt, state):
        events.append("post-first")
        assert state["lionplus_config"]["clipping"] == "adagc"
        assert state["lionplus_clipping"]["steps"] == 1
        return dict(state, complete=True)

    def post_last(opt, state):
        events.append("post-last")
        assert state["complete"]

    optimizer.register_state_dict_post_hook(post_last)
    optimizer.register_state_dict_post_hook(post_first, prepend=True)
    assert optimizer.state_dict()["complete"]
    assert events == ["pre-first", "pre-last", "post-first", "post-last"]


def test_load_hooks_can_replace_input_and_observe_complete_restored_state():
    p = parameter()
    source = LionPlus8bit([p], clipping="adagc")
    p.grad = torch.ones_like(p)
    source.step()
    saved = copy.deepcopy(source.state_dict())
    target = LionPlus8bit([parameter()])
    events = []

    def pre_first(opt, state):
        events.append("pre-first")
        assert state == {"wrapped": True}
        return copy.deepcopy(saved)

    def pre_last(opt, state):
        events.append("pre-last")
        assert state["lionplus_clipping"]["steps"] == 1

    def post_first(opt):
        events.append("post-first")
        assert opt.clipping == "adagc"
        assert opt._clipping_state.steps == 1
        assert opt.state[opt.param_groups[0]["params"][0]]["step"] == 1

    target.register_load_state_dict_pre_hook(pre_last)
    target.register_load_state_dict_pre_hook(pre_first, prepend=True)
    target.register_load_state_dict_post_hook(lambda opt: events.append("post-last"))
    target.register_load_state_dict_post_hook(post_first, prepend=True)
    target.load_state_dict({"wrapped": True})
    assert events == ["pre-first", "pre-last", "post-first", "post-last"]


def test_loading_checkpoint_clears_previous_step_diagnostics():
    p, q = parameter(), parameter()
    source = LionPlus8bit([p], clipping="adagc")
    target = LionPlus8bit([q], clipping="adagc")
    for optimizer, value in ((source, p), (target, q)):
        value.grad = torch.ones_like(value)
        optimizer.step()
    assert target.last_step_stats is not None
    assert target.last_clipping is not None
    target.load_state_dict(copy.deepcopy(source.state_dict()))
    assert target.last_step_stats is None
    assert target.last_clipping is None


def test_existing_optimizer_keeps_its_index_policy_when_another_registers():
    manager = bnb.optim.GlobalOptimManager.get_instance()
    registered, p, other = parameter(), parameter(), parameter()
    manager.override_config(registered, "lr", 0.25)
    manager.register_parameters([registered])
    source = LionPlus8bit([p], lr=0.1, weight_decay=0)
    manager.override_config(other, "lr", 0.9)
    manager.register_parameters([other])
    unrelated = LionPlus8bit([other], weight_decay=0)
    for optimizer, value in ((source, p), (unrelated, other)):
        value.grad = torch.ones_like(value)
        optimizer.step()
    torch.testing.assert_close(p, torch.tensor([0.75]))
    torch.testing.assert_close(other, torch.tensor([0.1]))


@pytest.mark.parametrize("mode", ["global", "adagc"])
def test_saved_lazy_allocation_policy_wins_over_destination_module(mode):
    manager = bnb.optim.GlobalOptimManager.get_instance()
    p, lazy = parameter(), parameter((64, 64))
    manager.override_config(lazy, "optim_bits", 32)
    source = LionPlus8bit([p, lazy], clipping=mode, weight_decay=0)
    p.grad = torch.ones_like(p)
    source.step()
    saved = copy.deepcopy(source.state_dict())
    module = torch.nn.Linear(64, 64, bias=False)
    manager.register_module_override(module, "weight", {"optim_bits": 8})
    resumed = LionPlus8bit([parameter(), module.weight])
    resumed.load_state_dict(saved)
    module.weight.grad = torch.ones_like(module.weight)
    resumed.step()
    assert resumed.state[module.weight]["state1"].dtype == torch.float32
    if mode == "adagc":
        assert resumed._clipping_state.initialized.tolist() == [True, True]
        assert resumed._clipping_state.history[0] == source._clipping_state.history[0]
