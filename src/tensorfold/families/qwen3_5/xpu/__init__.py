"""XPU verify rows match serial decoding on these kernels bit for bit, independently of window width."""

from .engine import xpu_engine

__all__ = ["xpu_engine"]
