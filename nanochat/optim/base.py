"""Contract between per-arch grouping helpers and the optimizer factory.

Each `nanochat/models/<arch>.py` ships `build_param_groups(model, cfg) ->
list[ParamGroupSpec]`; the factory routes each spec by `spec.optim_name`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from torch.nn import Parameter


@dataclass
class ParamGroupSpec:
    """A single optimizer param group, fully resolved for the factory.

    `lr` is the BASE learning rate the schedule multiplies against; any
    arch-specific LR scaling belongs in `build_param_groups`, as the factory
    does none.

    `optim_kwargs` is unpacked into the optimizer constructor / param-group
    dict. Keys omitted here fall back to the optimizer class's own defaults, so
    the per-arch helper sets them explicitly.
    """

    name: str
    params: list[Parameter]
    # Routing tag the factory branches on, not an optimizer object.
    optim_name: Literal["muon", "adamw"]
    lr: float
    weight_decay: float = 0.0
    optim_kwargs: dict[str, Any] = field(default_factory=dict)
