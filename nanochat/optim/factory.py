"""Optimizer factory, generic over ParamGroupSpec.

Routing by `spec.optim_name`; DDP vs single-process is a class swap per
algorithm:
  - "adamw": all adamw specs collapse into one (Dist)AdamW with one param
             group per spec.
  - "muon":  each muon spec becomes its own (Dist)Muon, since Muon takes a
             single LR per instance.

`initial_lr` is set on every returned param group; the LR schedule multiplies
against it.
"""

from __future__ import annotations

from functools import partial

import torch

from nanochat.optim.adamw import DistAdamW
from nanochat.optim.base import ParamGroupSpec
from nanochat.optim.muon import DistMuon, Muon


def build_optimizers(
    specs: list[ParamGroupSpec],
    *,
    ddp: bool,
) -> list[torch.optim.Optimizer]:
    AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
    MuonFactory = DistMuon if ddp else Muon

    # These literals are coupled to ParamGroupSpec.optim_name's Literal in base.py.
    adamw_specs = [s for s in specs if s.optim_name == "adamw"]
    muon_specs = [s for s in specs if s.optim_name == "muon"]

    optimizers: list[torch.optim.Optimizer] = []

    if adamw_specs:
        adam_groups = [
            dict(
                params=s.params,
                lr=s.lr,
                weight_decay=s.weight_decay,
                name=s.name,
                **s.optim_kwargs,
            )
            for s in adamw_specs
        ]
        optimizers.append(AdamWFactory(adam_groups))

    for s in muon_specs:
        muon = MuonFactory(s.params, lr=s.lr, **s.optim_kwargs)
        for group in muon.param_groups:
            group["name"] = s.name
        optimizers.append(muon)

    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    return optimizers
