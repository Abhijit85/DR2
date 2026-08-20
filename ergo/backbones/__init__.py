from .base import DenoiseResult, DiffusionBackbone, Snapshot  # noqa: F401
from .mock import MockBackbone  # noqa: F401

__all__ = ["DenoiseResult", "DiffusionBackbone", "Snapshot", "MockBackbone"]


def load_llada(**kw):
    from .hf_adapter import LLaDABackbone
    return LLaDABackbone(**kw)


def load_dream(**kw):
    from .hf_adapter import DreamBackbone
    return DreamBackbone(**kw)
