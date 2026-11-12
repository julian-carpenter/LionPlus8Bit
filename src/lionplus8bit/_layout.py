"""Dense parameter layouts with canonical logical row-major kernel work."""

import torch


def validate_parameter_layouts(params):
    """Reject internal overlap, holes and shared overlapping byte ranges.

    A permutation of contiguous dimensions is dense even when PyTorch's
    conservative internal-overlap probe returns ``too hard``. Ignoring size-one
    dimensions, increasing strides must exactly cover the preceding dimensions.
    """
    ranges = {}
    for parameter in params:
        if parameter.layout != torch.strided:
            raise RuntimeError("LionPlus8bit supports only dense parameters")
        if not parameter.numel():
            continue
        expected = 1
        for stride, size in sorted(
            (stride, size) for stride, size in zip(parameter.stride(), parameter.shape) if size > 1
        ):
            if stride != expected:
                raise RuntimeError(
                    "Parameters must be dense nonoverlapping contiguous permutations; "
                    "overlapping views and stepped slices are unsupported"
                )
            expected *= size
        key = (parameter.device, parameter.untyped_storage().data_ptr())
        start = parameter.storage_offset() * parameter.element_size()
        ranges.setdefault(key, []).append(
            (start, start + parameter.numel() * parameter.element_size())
        )
    for intervals in ranges.values():
        intervals.sort()
        if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
            raise RuntimeError("Parameter storage ranges must not overlap")


def parameter_layout(parameter):
    """Capture live layout identity; a load boundary resets this contract."""
    return (
        parameter.dtype,
        parameter.device,
        tuple(parameter.shape),
        parameter.stride(),
        parameter.storage_offset(),
        parameter.data_ptr(),
    )


def canonical_work(tensor):
    """Return logical row-major work, borrowing contiguous input storage."""
    return tensor.detach().contiguous()


def copy_back(parameter, work):
    """Write values without replacing the original storage or view metadata."""
    parameter.copy_(work)
