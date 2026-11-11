import importlib.util
from pathlib import Path

import pytest


def test_spikes_use_only_previous_observations():
    path = Path(__file__).resolve().parents[1] / "tools" / "study_clipping.py"
    spec = importlib.util.spec_from_file_location("study", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.spike_score([1.0] * 1000 + [2.0]) == pytest.approx(1.0)
    assert module.spike_score([1.0] * 1001) == 0
    assert module.spike_score([1.0] * 10) is None
