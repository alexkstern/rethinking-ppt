"""Model package. Callers import a concrete arch (and version) by its submodule
path, e.g. `nanochat.models.nanochat_gpt.vanilla_3a5e0bc`; only `BaseModel`,
the cross-arch contract, is re-exported here.
"""

from nanochat.models.base import BaseModel

__all__ = ["BaseModel"]
