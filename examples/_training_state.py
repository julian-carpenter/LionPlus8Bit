"""Quiescent, fixed-world-size trainer checkpoints, separate from optimizer state.

Recreate model, optimizer, loader and iterator before restoring trainer RNG. Use
``restore_rng=False`` on load followed by ``restore_training_rng`` after iterator
construction. Data state must describe consumed samples, never prefetched work.
Only load checkpoints produced by trusted training jobs (the payload uses pickle).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import random
import uuid

import numpy as np
import torch
import torch.distributed as dist


def _base_model(model):
    from torch._dynamo.eval_frame import OptimizedModule

    while True:
        if isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)):
            model = model.module
        elif isinstance(model, OptimizedModule):
            model = model._orig_mod
        else:
            return model


def _rank_world(group):
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(group), dist.get_world_size(group)
    return 0, 1


def capture_training_rng():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_training_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"].cpu())
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA RNG checkpoint requires a CUDA trainer")
        torch.cuda.set_rng_state(state["cuda"].cpu())


def _checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, content):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        json.dump(content, handle, sort_keys=True, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def save_training_checkpoint(
    path, *, model, optimizer, scaler, scheduler, data_state, process_group=None
):
    """All ranks participate; rank zero publishes after every component is durable.

    Call only between accumulation groups, with no pending model/data mutations.
    A crash before publication leaves the prior manifest and an ignored generation.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    rank, world = _rank_world(process_group)
    generation = ["generation-" + uuid.uuid4().hex if rank == 0 else None]
    if world > 1:
        source = dist.get_global_rank(process_group, 0) if process_group is not None else 0
        dist.broadcast_object_list(generation, src=source, group=process_group)
    generation = generation[0]
    directory = path / generation
    directory.mkdir(exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    payload = {
        "schema": 1,
        "rank": rank,
        "world_size": world,
        "model": _base_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "data_state": data_state,
        "rng_state": capture_training_rng(),
    }
    component = directory / f"rank-{rank}.pt"
    temporary = component.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, component)
    directory_descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    metadata = {
        "rank": rank,
        "file": f"{generation}/{component.name}",
        "sha256": _checksum(component),
    }
    components = [None] * world
    if world > 1:
        dist.all_gather_object(components, metadata, group=process_group)
    else:
        components[0] = metadata
    if rank == 0:
        _atomic_json(
            path / "manifest.json",
            {"schema": 1, "generation": generation, "world_size": world, "components": components},
        )
    if world > 1:
        dist.barrier(group=process_group)
    return generation


def _read_training_checkpoint(path, rank, world, expected):
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No committed training checkpoint in {path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != 1:
        raise RuntimeError("Unsupported training checkpoint schema")
    if manifest["world_size"] != expected or expected != world:
        raise RuntimeError("Changed world size is unsupported for exact training replay")
    components = manifest["components"]
    if len(components) != world or {item["rank"] for item in components} != set(range(world)):
        raise RuntimeError("Incomplete committed checkpoint rank components")
    for item in components:
        component = path / item["file"]
        if component.resolve().parent.parent != path.resolve():
            raise RuntimeError("Checkpoint component escaped generation directory")
        if not component.is_file() or _checksum(component) != item["sha256"]:
            raise RuntimeError(f"Checkpoint checksum mismatch: rank {item['rank']}")
    item = next(item for item in components if item["rank"] == rank)
    payload = torch.load(path / item["file"], map_location="cpu", weights_only=False)
    if (
        payload.get("schema") != 1
        or payload.get("rank") != rank
        or payload.get("world_size") != world
    ):
        raise RuntimeError("Checkpoint payload rank/world/schema mismatch")
    return manifest, payload


def _cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        snapshot = copy.copy(value)
        for key, item in value.items():
            snapshot[key] = _cpu_snapshot(item)
        if hasattr(value, "_metadata"):
            snapshot._metadata = copy.deepcopy(value._metadata)
        return snapshot
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item) for item in value)
    return copy.deepcopy(value)


def _agreed_errors(error, world, group):
    message = None if error is None else f"{type(error).__name__}: {error}"
    messages = [None] * world
    if world > 1:
        dist.all_gather_object(messages, message, group=group)
    else:
        messages[0] = message
    return [message for message in messages if message is not None]


def load_training_checkpoint(
    path,
    *,
    model,
    optimizer,
    scaler,
    scheduler,
    process_group=None,
    restore_rng=True,
    expected_world_size=None,
):
    """Validate the common generation, then restore transactionally on every rank.

    Public component loaders can reject semantic state after making mutations.
    CPU value snapshots permit rollback on all ranks before any update is allowed.
    """
    path = Path(path)
    rank, world = _rank_world(process_group)
    expected = world if expected_world_size is None else expected_world_size
    base = _base_model(model)
    error = None
    try:
        manifest, payload = _read_training_checkpoint(path, rank, world, expected)
        current, saved = base.state_dict(), payload["model"]
        if current.keys() != saved.keys() or any(
            isinstance(current[key], torch.Tensor)
            and (not isinstance(saved[key], torch.Tensor) or current[key].shape != saved[key].shape)
            for key in current
        ):
            raise RuntimeError("Checkpoint model mapping does not match reconstructed trainer")
        if (scaler is None) != (payload["scaler"] is None):
            raise RuntimeError("Checkpoint scaler configuration differs")
        if (scheduler is None) != (payload["scheduler"] is None):
            raise RuntimeError("Checkpoint scheduler configuration differs")
        if (
            restore_rng
            and payload["rng_state"]["cuda"] is not None
            and not torch.cuda.is_available()
        ):
            raise RuntimeError("CUDA RNG checkpoint requires a CUDA trainer")
    except Exception as caught:
        error = caught
    errors = _agreed_errors(error, world, process_group)
    if errors:
        if world == 1:
            raise error
        raise RuntimeError("Checkpoint validation rejected on worker group: " + "; ".join(errors))
    components = {"model": base, "optimizer": optimizer, "scaler": scaler, "scheduler": scheduler}
    backups = {
        name: _cpu_snapshot(component.state_dict())
        for name, component in components.items()
        if component is not None
    }
    error = None
    try:
        for name, component in components.items():
            if component is not None:
                component.load_state_dict(payload[name])
    except Exception as caught:
        error = caught
    errors = _agreed_errors(error, world, process_group)
    if errors:
        for name, component in components.items():
            if component is not None:
                component.load_state_dict(backups[name])
        raise RuntimeError(
            "Checkpoint restore rolled back on worker group: " + "; ".join(errors)
        ) from error
    if restore_rng:
        restore_training_rng(payload["rng_state"])
    return {
        "data_state": payload["data_state"],
        "rng_state": payload["rng_state"],
        "generation": manifest["generation"],
        "world_size": world,
    }


def validate_replay_contract(
    *,
    saved_world_size,
    world_size,
    checkpointable,
    saved_workers=None,
    loader_workers=None,
    stateful_iterable=False,
):
    if saved_world_size != world_size:
        raise RuntimeError("Changed world size is unsupported for exact replay")
    if not checkpointable:
        raise RuntimeError("External streams require checkpoint/replay support")
    if stateful_iterable and saved_workers != loader_workers:
        raise RuntimeError("Stateful iterable replay requires unchanged loader worker count")


def make_stateful_loader(
    dataset,
    *,
    num_workers,
    batch_size=1,
    saved_state=None,
    saved_workers=None,
    generator=None,
    **kwargs,
):
    """Lazy optional torchdata adapter; caller saves each rank's loader snapshot.

    Iterable sources must expose replayable dataset state. Worker-count changes
    are rejected because worker snapshots cannot be repartitioned exactly.
    """
    iterable = isinstance(dataset, torch.utils.data.IterableDataset)
    if iterable and not (
        callable(getattr(dataset, "state_dict", None))
        and callable(getattr(dataset, "load_state_dict", None))
    ):
        raise RuntimeError("External streams require checkpoint/replay support")
    if saved_state is not None:
        validate_replay_contract(
            saved_world_size=1,
            world_size=1,
            checkpointable=True,
            saved_workers=saved_workers,
            loader_workers=num_workers,
            stateful_iterable=iterable,
        )
    try:
        from torchdata.stateful_dataloader import StatefulDataLoader
    except ImportError as error:
        raise RuntimeError(
            "Stateful recovery requires optional torchdata StatefulDataLoader"
        ) from error
    # A dedicated worker generator keeps iterator construction out of trainer RNG.
    if generator is None:
        generator = torch.Generator().manual_seed(0)
    loader = StatefulDataLoader(
        dataset, num_workers=num_workers, batch_size=batch_size, generator=generator, **kwargs
    )
    if saved_state is not None:
        loader.load_state_dict(saved_state)
    return loader
