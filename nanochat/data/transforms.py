"""Stateless per-batch target transforms.

`mask_token_ids` is the plain function; the `with_*` wrappers compose onto a
whole loader at train-assembly time.
"""

from typing import Iterable

import torch

from .base import DataIterator


def mask_token_ids(targets: torch.Tensor, ids: Iterable[int]) -> torch.Tensor:
    """Set positions in `targets` whose value is in `ids` to -1 (train-loop ignore_index)."""
    ids_t = torch.tensor(list(ids), dtype=torch.long, device=targets.device)
    return targets.masked_fill(torch.isin(targets, ids_t), -1)


def with_id_offset(loader: DataIterator, offset: int) -> DataIterator:
    """Wrap a loader so every token id is shifted up by `offset`.

    Maps a narrow-vocab corpus into a disjoint high slice of a larger model
    vocab (e.g. dyck ids 0..255 -> 65536..65791 above the 65536 text ids), so
    a synthetic phase can share the model's embedding table with the text
    phase. Negative target sentinels (the ignore_index) pass through
    unshifted.
    """
    for inputs, targets, state in loader:
        yield inputs + offset, torch.where(targets >= 0, targets + offset, targets), state


def with_token_id_masking(loader: DataIterator, ids: Iterable[int]) -> DataIterator:
    """Wrap a loader so each batch's `targets` have `ids` masked to -1.

    `inputs` and the `state` slot pass through untouched. The NCA recipe wraps
    its loader with `ids=(START, END)` to drop the per-grid delimiter positions
    from the loss.
    """
    ids = list(ids)
    for inputs, targets, state in loader:
        yield inputs, mask_token_ids(targets, ids), state
