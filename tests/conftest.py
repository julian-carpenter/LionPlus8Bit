import pytest
import torch


def pytest_addoption(parser):
    parser.addoption("--require-cuda", action="store_true", help="Fail if CUDA is unavailable")


def pytest_sessionstart(session):
    if session.config.getoption("--require-cuda") and not torch.cuda.is_available():
        raise pytest.UsageError("CUDA is required; refusing a CPU fallback")


@pytest.fixture(autouse=True)
def require_cuda_for_marked_tests(request):
    if request.node.get_closest_marker("cuda") and not torch.cuda.is_available():
        pytest.skip("Requires CUDA; use --require-cuda to prohibit a CPU fallback")


@pytest.fixture(autouse=True)
def isolate_native_overrides():
    import bitsandbytes as bnb

    manager = bnb.optim.GlobalOptimManager.get_instance()
    saved = (
        manager.pid2config.copy(),
        manager.index2config.copy(),
        list(manager.module_weight_config_triple),
    )
    manager.pid2config.clear()
    manager.index2config.clear()
    manager.module_weight_config_triple.clear()
    yield
    manager.pid2config, manager.index2config, manager.module_weight_config_triple = saved
