"""Paging evidence helper: ordinary controls and explicit native ownership blocker."""

import copy
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess

import bitsandbytes as bnb
import torch

from lionplus8bit import LionPlus8bit
from lionplus8bit._paging import allocate_state, paging_capability


def _native_evidence():
    source = inspect.getsource(bnb.functional.get_paged)
    manager = inspect.getsource(bnb.functional.GlobalPageManager)
    libraries = []
    if shutil.which("nm"):
        for library in sorted(Path(bnb.__file__).parent.glob("libbitsandbytes_cuda*.so")):
            result = subprocess.run(
                ["nm", "-D", str(library)],
                capture_output=True,
                text=True,
                check=True,
            )
            symbols = result.stdout
            libraries.append(
                dict(
                    path=str(library),
                    managed_symbols=[
                        line.strip()
                        for line in symbols.splitlines()
                        if any(
                            key in line
                            for key in (
                                "cget_managed_ptr",
                                "cprefetch",
                                "cudaMallocManaged",
                                "cudaMemPrefetchAsync",
                                "cudaFree",
                                "cfree",
                            )
                        )
                    ],
                )
            )
    return dict(
        bitsandbytes=bnb.__version__,
        functional_path=inspect.getsourcefile(bnb.functional),
        get_paged_source=source,
        get_paged_sha256=hashlib.sha256(source.encode()).hexdigest(),
        manager_source=manager,
        libraries=libraries,
        observations=[
            "get_paged wraps cget_managed_ptr with numpy.ctypeslib and torch.frombuffer",
            "No tensor storage deleter or allocator release operation is attached",
            "GlobalPageManager.paged_tensors retains strong tensor references",
            "CPU-labelled tensors cannot represent CUDA operations through metadata alone",
        ],
    )


def _ordinary_controls(device):
    controls = []
    counts = (99_999, 100_000, 100_001, 100_257) if device.type == "cpu" else (99_999,)
    before = len(bnb.functional.GlobalPageManager.get_instance().paged_tensors)
    for count in counts:
        for bits in (8, 32):
            template = torch.empty(count, device=device)
            state = allocate_state(
                template, torch.uint8 if bits == 8 else torch.float32, is_paged=True
            )
            assert not getattr(state, "is_paged", False)
            assert state.is_contiguous() and state.device == device
            assert not bool(state.any())
            # CPU fallback and subthreshold CUDA are ordinary-storage controls;
            # none is a positive native managed-paging qualification.
            parameter = torch.nn.Parameter(torch.linspace(-1, 1, count, device=device))
            reference = torch.nn.Parameter(parameter.detach().clone())
            options = dict(lr=0.01, min_8bit_size=0 if bits == 8 else count + 1)
            optimizer = LionPlus8bit([parameter], is_paged=True, **options)
            control = LionPlus8bit([reference], **options)
            for gradient in (0.2, 1e-35, -0.2):
                parameter.grad = torch.full_like(parameter, gradient)
                reference.grad = parameter.grad.clone()
                optimizer.step()
                control.step()
                torch.testing.assert_close(parameter, reference, atol=0, rtol=0)
                torch.testing.assert_close(
                    optimizer.state[parameter]["state1"],
                    control.state[reference]["state1"],
                    atol=0,
                    rtol=0,
                )
            resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
            resumed = LionPlus8bit([resumed_parameter])
            resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()), move_to_device=False)
            parameter.grad = torch.full_like(parameter, -0.3)
            resumed_parameter.grad = parameter.grad.clone()
            optimizer.step()
            resumed.step()
            torch.testing.assert_close(parameter, resumed_parameter, atol=0, rtol=0)
            controls.append(
                dict(
                    elements=count,
                    momentum_bits=bits,
                    status="passed",
                    strict_math=True,
                    deferred_checkpoint=True,
                    managed_storage=False,
                )
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    assert len(bnb.functional.GlobalPageManager.get_instance().paged_tensors) == before
    return dict(status="passed", device=str(device), cases=controls, global_page_manager_growth=0)


def run_paging_suite(output_dir, *, device=None):
    """Use an existing evidence directory; fail rather than overwrite our files.

    ``device=None`` is CPU-only. An explicit CUDA ordinal is reserved for the
    coordinator's serialized GPU run, and never falls back to CPU.
    """
    output_dir = Path(output_dir)
    native_path = output_dir / "paging_native_evidence.json"
    if native_path.exists():
        raise FileExistsError(native_path)
    if device is None:
        target = torch.device("cpu")
    else:
        if not torch.cuda.is_available() or not 0 <= device < torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {device} required; CPU fallback forbidden")
        torch.cuda.set_device(device)
        target = torch.device("cuda", device)
    evidence = _native_evidence()
    with native_path.open("x") as stream:
        json.dump(evidence, stream, indent=2)
        stream.write("\n")
    capability = paging_capability()
    return dict(
        status="unsupported",
        suite="paging",
        capability=capability,
        managed_allocation_attempted=False,
        ordinary_storage_controls=_ordinary_controls(target),
        managed_numerics=dict(status="unsupported", reason=capability["reason"]),
        managed_restore=dict(status="unsupported", reason="No owned native allocation to restore"),
        managed_lifetime=dict(status="unsupported", reason="No native reclamation contract"),
        managed_ddp=dict(
            status="unsupported", reason="Ownership gate applies independently per rank"
        ),
        pressure_stress=dict(
            status="skipped", reason="Managed allocator ownership prerequisite failed"
        ),
        migration_performance=dict(status="skipped", reason="No qualified managed storage"),
        native_evidence=str(native_path),
    )
