"""Trainer checkpoint integrity and real subprocess rollback controls."""

import importlib.util
from pathlib import Path
import random

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def training_state():
    spec = importlib.util.spec_from_file_location(
        "training_state", ROOT / "examples/_training_state.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def objects():
    from lionplus8bit import LionPlus8bit

    model = torch.nn.Linear(3, 2)
    optimizer = LionPlus8bit(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    scheduler.step()
    return model, optimizer, scheduler


def test_checkpoint_roundtrip_preserves_rng_counters_and_base_model(tmp_path):
    api = training_state()
    model, opt, scheduler = objects()
    api.save_training_checkpoint(
        tmp_path,
        model=model,
        optimizer=opt,
        scaler=None,
        scheduler=scheduler,
        data_state={"consumed_batches": 3, "successful_updates": 2, "world_size": 1},
    )
    expected_torch = torch.rand(5)
    expected_python = random.random()
    target, new_opt, new_scheduler = objects()
    result = api.load_training_checkpoint(
        tmp_path, model=target, optimizer=new_opt, scaler=None, scheduler=new_scheduler
    )
    assert result["data_state"]["consumed_batches"] == 3
    assert result["data_state"]["successful_updates"] == 2
    assert torch.equal(torch.rand(5), expected_torch)
    assert random.random() == expected_python
    for left, right in zip(model.parameters(), target.parameters()):
        assert torch.equal(left, right)
    assert new_scheduler.state_dict() == scheduler.state_dict()


def test_corrupt_checkpoint_rejected_before_model_or_optimizer_mutation(tmp_path):
    api = training_state()
    model, opt, scheduler = objects()
    api.save_training_checkpoint(
        tmp_path,
        model=model,
        optimizer=opt,
        scaler=None,
        scheduler=scheduler,
        data_state={"consumed_batches": 2},
    )
    payload = next(tmp_path.glob("generation-*/rank-0.pt"))
    payload.write_bytes(b"truncated")
    before = [p.detach().clone() for p in model.parameters()]
    with pytest.raises(RuntimeError, match="checksum"):
        api.load_training_checkpoint(
            tmp_path, model=model, optimizer=opt, scaler=None, scheduler=scheduler
        )
    assert all(torch.equal(p, saved) for p, saved in zip(model.parameters(), before))


def test_uncommitted_generation_ignored_and_changed_world_rejected(tmp_path):
    api = training_state()
    model, opt, scheduler = objects()
    (tmp_path / "generation-incomplete").mkdir()
    with pytest.raises(FileNotFoundError, match="committed"):
        api.load_training_checkpoint(
            tmp_path, model=model, optimizer=opt, scaler=None, scheduler=scheduler
        )
    api.save_training_checkpoint(
        tmp_path,
        model=model,
        optimizer=opt,
        scaler=None,
        scheduler=scheduler,
        data_state={"world_size": 2},
    )
    with pytest.raises(RuntimeError, match="world size"):
        api.load_training_checkpoint(
            tmp_path,
            model=model,
            optimizer=opt,
            scaler=None,
            scheduler=scheduler,
            expected_world_size=2,
        )


def test_rng_restored_after_loader_reconstruction(tmp_path):
    api = training_state()
    model, opt, scheduler = objects()
    api.save_training_checkpoint(
        tmp_path,
        model=model,
        optimizer=opt,
        scaler=None,
        scheduler=scheduler,
        data_state={"consumed_batches": 2},
    )
    expected = torch.rand(3)
    result = api.load_training_checkpoint(
        tmp_path, model=model, optimizer=opt, scaler=None, scheduler=scheduler, restore_rng=False
    )
    loader_rng = torch.Generator().manual_seed(91)
    loader = torch.utils.data.DataLoader(torch.arange(12), batch_size=2, generator=loader_rng)
    next(iter(loader))
    api.restore_training_rng(result["rng_state"])
    assert torch.equal(torch.rand(3), expected)


def test_controlled_trainer_crash_relaunches_and_replays_bitwise(tmp_path):
    import subprocess
    import sys

    command = [
        sys.executable,
        str(ROOT / "tools/restart_integration.py"),
        "--world-size",
        "1",
        "--loader-workers",
        "0",
        "--case",
        "trainer",
        "--fault-point",
        "inside-accumulation",
        "--device",
        "cpu",
        "--output",
        str(tmp_path / "run"),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    import json

    report = json.loads((tmp_path / "run/summary.json").read_text())
    assert report["status"] == "passed"
    assert report["cases"][0]["injected_exit_code"] != 0
    assert report["cases"][0]["recovery_exit_code"] == 0
    assert report["cases"][0]["bitwise_equal"] is True


def test_replay_contract_rejects_external_streams_and_worker_count_change():
    api = training_state()
    with pytest.raises(RuntimeError, match="checkpoint/replay"):
        api.validate_replay_contract(saved_world_size=1, world_size=1, checkpointable=False)
    with pytest.raises(RuntimeError, match="worker count"):
        api.validate_replay_contract(
            saved_world_size=1,
            world_size=1,
            checkpointable=True,
            saved_workers=2,
            loader_workers=4,
            stateful_iterable=True,
        )


class StatefulCounter(torch.utils.data.IterableDataset):
    def __init__(self):
        self.cursor = 0

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        while self.cursor < 48:
            value = self.cursor
            self.cursor += 1
            if worker is None or value % worker.num_workers == worker.id:
                yield value

    def state_dict(self):
        return {"cursor": self.cursor}

    def load_state_dict(self, state):
        self.cursor = state["cursor"]


@pytest.mark.parametrize("workers", [0, 2, 4])
def test_stateful_loader_snapshot_replays_iterable_when_optional_backend_present(workers):
    pytest.importorskip("torchdata.stateful_dataloader")
    api = training_state()
    kwargs = {"prefetch_factor": 1} if workers else {}
    first = api.make_stateful_loader(StatefulCounter(), num_workers=workers, batch_size=2, **kwargs)
    iterator = iter(first)
    next(iterator)
    snapshot = first.state_dict()
    expected = [next(iterator) for _ in range(3)]
    restored = api.make_stateful_loader(
        StatefulCounter(),
        num_workers=workers,
        batch_size=2,
        saved_state=snapshot,
        saved_workers=workers,
        **kwargs,
    )
    resumed_iterator = iter(restored)
    assert all(torch.equal(next(resumed_iterator), batch) for batch in expected)


def test_generation_directory_is_durable_before_manifest_publication(tmp_path, monkeypatch):
    import os

    api = training_state()
    events = []
    original_fsync = api.os.fsync
    original_replace = api.os.replace

    def fsync(descriptor):
        events.append(("fsync", os.readlink(f"/proc/self/fd/{descriptor}")))
        return original_fsync(descriptor)

    def replace(source, destination):
        events.append(("replace", str(destination)))
        return original_replace(source, destination)

    monkeypatch.setattr(api.os, "fsync", fsync)
    monkeypatch.setattr(api.os, "replace", replace)
    model, optimizer, scheduler = objects()
    generation = api.save_training_checkpoint(
        tmp_path,
        model=model,
        optimizer=optimizer,
        scaler=None,
        scheduler=scheduler,
        data_state={"consumed_batches": 2},
    )
    component_publication = events.index(("replace", str(tmp_path / generation / "rank-0.pt")))
    directory_sync = events.index(("fsync", str(tmp_path / generation)))
    manifest_publication = events.index(("replace", str(tmp_path / "manifest.json")))
    assert component_publication < directory_sync < manifest_publication


def test_invalid_optimizer_payload_does_not_partially_restore_model(tmp_path):
    import hashlib
    import json

    api = training_state()
    saved, optimizer, scheduler = objects()
    api.save_training_checkpoint(
        tmp_path,
        model=saved,
        optimizer=optimizer,
        scaler=None,
        scheduler=scheduler,
        data_state={"consumed_batches": 2},
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    component = tmp_path / manifest["components"][0]["file"]
    payload = torch.load(component, weights_only=False)
    payload["optimizer"]["param_groups"] = []
    torch.save(payload, component)
    manifest["components"][0]["sha256"] = hashlib.sha256(component.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    target, target_optimizer, target_scheduler = objects()
    with torch.no_grad():
        for parameter in target.parameters():
            parameter.add_(7)
    before = [p.detach().clone() for p in target.parameters()]
    with pytest.raises((RuntimeError, ValueError), match="parameter|restore|group"):
        api.load_training_checkpoint(
            tmp_path,
            model=target,
            optimizer=target_optimizer,
            scaler=None,
            scheduler=target_scheduler,
        )
    assert all(
        torch.equal(parameter, value) for parameter, value in zip(target.parameters(), before)
    )


def test_normal_child_named_module_keeps_outer_parameters_and_buffers(tmp_path):
    from lionplus8bit import LionPlus8bit

    api = training_state()
    model = torch.nn.Module()
    model.add_module("module", torch.nn.Linear(3, 2))
    model.register_parameter("extra", torch.nn.Parameter(torch.tensor(7.0)))
    model.register_buffer("outer_buffer", torch.tensor(13))
    optimizer = LionPlus8bit(model.parameters())
    api.save_training_checkpoint(
        tmp_path,
        model=model,
        optimizer=optimizer,
        scaler=None,
        scheduler=None,
        data_state={"consumed_batches": 0},
    )
    with torch.no_grad():
        model.extra.fill_(1)
        model.outer_buffer.fill_(2)
    api.load_training_checkpoint(
        tmp_path, model=model, optimizer=optimizer, scaler=None, scheduler=None
    )
    assert model.extra.item() == 7
    assert model.outer_buffer.item() == 13
