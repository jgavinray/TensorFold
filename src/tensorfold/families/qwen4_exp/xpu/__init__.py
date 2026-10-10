"""Flash Next's XPU weights: the ModelOpt (NVFP4) checkpoint path of the CUDA engine, on Intel GPUs."""

from .weights import load

__all__ = ["load"]


def xpu_engine():
    """The XPU engine wiring (streams, kernels, graphs): a later slice."""
    raise NotImplementedError("engine wiring is a later slice")
