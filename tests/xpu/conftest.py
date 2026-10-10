"""The XPU engines' tests: collected only where PyTorch sees an Intel GPU (the B70 boxes)."""

import gc
import importlib.util

import pytest


def _xpu() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    return torch.xpu.is_available()


collect_ignore_glob = [] if _xpu() else ["test_*.py"]


@pytest.fixture(autouse=True, scope="module")
def _give_back_gpu_memory():
    """After each module: free its tensors and torch's cached blocks, so the next engine admits as a fresh process."""

    yield
    if not _xpu():
        return
    import torch

    if torch.xpu.is_initialized():
        gc.collect()
        torch.xpu.empty_cache()
