from .base import DenoiseResult, DiffusionBackbone, Snapshot  # noqa: F401
from .mock import MockBackbone  # noqa: F401
from .dream_native import NativeDreamBackbone, SARDIDreamBackbone  # noqa: F401

__all__ = ["DenoiseResult", "DiffusionBackbone", "Snapshot", "MockBackbone", "NativeDreamBackbone", "SARDIDreamBackbone"]


def load_llada(**kw):
    from .hf_adapter import LLaDABackbone
    return LLaDABackbone(**kw)


def load_dream(**kw):
    from .hf_adapter import DreamBackbone
    return DreamBackbone(**kw)


def load_dream_native(**kw):
    from .dream_native import NativeDreamBackbone
    return NativeDreamBackbone(**kw)



def load_sardi_dream(**kw):
    from .dream_native import SARDIDreamBackbone
    return SARDIDreamBackbone(**kw)
