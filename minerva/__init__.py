"""Minerva package.

The heavy model classes are imported lazily so lightweight submodules
(``minerva.visualization``, ``minerva.data``) can be used without pulling in
``modeling_minerva`` and its flash-attn dependency.
"""

__all__ = [
    "MinervaConfig",
    "MinervaForMaskedLM",
    "MinervaForMaskedLMOutput",
    "MinervaModel",
    "MinervaPreTrainedModel",
]


def __getattr__(name):
    if name in __all__:
        from . import modeling_minerva
        return getattr(modeling_minerva, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
