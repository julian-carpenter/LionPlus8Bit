"""Shared immutable evidence and capability discovery for integration runners."""

import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import torch


def output_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def capabilities():
    import bitsandbytes as bnb

    root = Path(__file__).resolve().parents[1]
    optional = {}
    for name in ("torchao", "transformer_engine", "torchdata", "triton"):
        optional[name] = {"present": importlib.util.find_spec(name) is not None}
        try:
            optional[name]["version"] = importlib.metadata.version(name.replace("_", "-"))
        except importlib.metadata.PackageNotFoundError:
            pass
    devices = [
        dict(
            index=i,
            name=torch.cuda.get_device_name(i),
            capability=torch.cuda.get_device_capability(i),
            memory=torch.cuda.get_device_properties(i).total_memory,
        )
        for i in range(torch.cuda.device_count())
    ]
    try:
        topology = subprocess.run(
            ["nvidia-smi", "topo", "-m"], capture_output=True, text=True
        ).stdout
    except FileNotFoundError:
        topology = "nvidia-smi unavailable"
    try:
        inventory = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,driver_version,memory.total",
                "--format=csv",
            ],
            capture_output=True,
            text=True,
        ).stdout
    except FileNotFoundError:
        inventory = "nvidia-smi unavailable"
    return dict(
        python=platform.python_version(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        bitsandbytes=bnb.__version__,
        devices=devices,
        optional=optional,
        nccl_available=torch.distributed.is_nccl_available(),
        source_hashes={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ("src/lionplus8bit", "tools", "examples")
            for p in (root / folder).glob("*.py")
        },
        topology=topology,
        inventory=inventory,
        environment={
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "NCCL_P2P_DISABLE",
                "NCCL_ALGO",
                "NCCL_PROTO",
                "CUBLAS_WORKSPACE_CONFIG",
            )
        },
        native_fp8=any(tuple(d["capability"]) >= (8, 9) for d in devices),
    )
