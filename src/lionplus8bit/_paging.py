"""Conservative native paging boundary for pinned bitsandbytes 0.50.2.

Its ``get_paged`` returns a CPU-labelled ``torch.frombuffer`` of a raw
``cudaMallocManaged`` pointer. There is no storage deleter, and the global page
manager retains a strong reference without a release API. Python attributes are
not evidence of CUDA-managed storage. Until an owned allocation/alias contract
exists, reject eligible requests before allocation instead of leaking memory or
freeing pointers still referenced by tensors/kernels.
"""

from __future__ import annotations

import torch

MANAGED_MIN_NUMEL = 100_000
_REASON = (
    "Native managed paging is unsupported: bitsandbytes 0.50.2 get_paged has no "
    "storage deleter or allocator release contract; CPU-labelled managed state "
    "also cannot safely enter PyTorch CUDA guard/strict arithmetic. "
    "Use is_paged=False for parameters with at least 100000 elements on CUDA."
)


class PagingUnsupportedError(RuntimeError):
    """Raised before entering an allocator with unqualified storage ownership."""


def paging_capability():
    """Report an explicit unsupported result without initializing CUDA."""
    return dict(
        managed_supported=False,
        reclamation_supported=False,
        managed_allocation_attempted=False,
        min_managed_elements=MANAGED_MIN_NUMEL,
        allocator="bitsandbytes.functional.get_paged",
        reason=_REASON,
        ordinary_storage="CPU parameters and CUDA parameters below 100000 elements",
    )


def validate_paging_request(parameters, *, is_paged):
    """Fail before any native state allocation, update, or checkpoint mutation."""
    if is_paged:
        for parameter in parameters:
            if parameter.device.type != "cpu" and parameter.numel() >= MANAGED_MIN_NUMEL:
                raise PagingUnsupportedError(_REASON)


def allocate_state(parameter, dtype=torch.float32, *, is_paged=False):
    """Allocate canonical ordinary state; never relabel it as managed storage."""
    validate_paging_request([parameter], is_paged=is_paged)
    return torch.zeros(parameter.shape, dtype=dtype, device=parameter.device)


def checkpoint_values(value):
    """Copy legacy paged values, removing pointer-like Python metadata recursively.

    This operation does not certify managed allocation or reconstruct it. Loading
    onto an eligible CUDA parameter must pass ``validate_paging_request`` first.
    Plain tensors retain the usual shallow ``Optimizer.state_dict`` semantics.
    """
    if isinstance(value, torch.Tensor):
        if any(hasattr(value, key) for key in ("is_paged", "page_deviceid", "is_managed")):
            # Native kernels may use the default stream for this CPU-labelled
            # storage. Complete outstanding device work before a host snapshot.
            ordinal = getattr(value, "page_deviceid", None)
            if getattr(value, "is_paged", False) and torch.cuda.is_initialized():
                if isinstance(ordinal, int) and 0 <= ordinal < torch.cuda.device_count():
                    torch.cuda.synchronize(ordinal)
                else:
                    raise PagingUnsupportedError(
                        "Legacy paged state has no valid CUDA device ordinal"
                    )
            return value.detach().clone()
        return value
    if isinstance(value, dict):
        return {key: checkpoint_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(checkpoint_values(item) for item in value)
    return value
