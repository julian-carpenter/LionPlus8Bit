import inspect
from importlib.metadata import version

import torch

from lionplus8bit import LionPlus8bit, __version__


def test_version_matches_distribution():
    assert version("lionplus8bit") == __version__


def test_optimized_defaults():
    parameters = inspect.signature(LionPlus8bit).parameters
    assert parameters["native_gradient_scale"].default is True
    assert parameters["batched_norm"].default is True
    p = torch.nn.Parameter(torch.ones(4))
    optimizer = LionPlus8bit([p])
    assert optimizer.native_gradient_scale is True
    assert optimizer.batched_norm is True
    assert optimizer.master_weights is True
