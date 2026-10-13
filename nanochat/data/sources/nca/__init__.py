"""NCA procedural data source. Requires the optional `nca` extra (JAX/Flax).

The submodule import is deferred (PEP 562), so importing this package does
not pull in JAX.
"""

__all__ = ["nca_paper_source"]


def __getattr__(name):
    if name == "nca_paper_source":
        from .source import nca_paper_source
        return nca_paper_source
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
