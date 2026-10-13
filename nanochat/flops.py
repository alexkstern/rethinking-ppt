"""FLOPs measurement and compute-budget -> iteration arithmetic.

The budget split between phases is the caller's concern: experiment scripts
call `iterations_for_budget` once per phase with that phase's share of
`target_flops`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def flops_per_token(
    model: nn.Module,
    seq_len: int,
    vocab_size: int,
    *,
    use_measured: bool = True,
    device: torch.device | None = None,
) -> float:
    """FLOPs per token for one fwd+bwd step at the given sequence length.

    `use_measured=True` counts a single fwd+bwd over a random `(1, seq_len)`
    batch under `FlopCounterMode`. The model must already be on the target
    device and must be UNCOMPILED — the count is identical either way, but
    torch.compile under FlopCounterMode is very slow, so the order is
    build -> measure -> compile -> train.

    `use_measured=False` is the closed form `6 * (N - N_embed)`, which skips
    the attention correction and ignores `vocab_size`. Use the measured path
    for compute-matched claims.
    """
    if not use_measured:
        return _analytical_flops_per_token(model)
    return _measure_via_flop_counter(model, seq_len, vocab_size, device)


def _analytical_flops_per_token(model: nn.Module) -> float:
    total = sum(p.numel() for p in model.parameters())
    embed = sum(
        p.numel()
        for m in model.modules()
        if isinstance(m, nn.Embedding)
        for p in m.parameters(recurse=False)
    )
    return 6.0 * (total - embed)


def _measure_via_flop_counter(
    model: nn.Module,
    seq_len: int,
    vocab_size: int,
    device: torch.device | None,
) -> float:
    from torch.utils.flop_counter import FlopCounterMode

    if device is None:
        device = next(model.parameters()).device
    assert device.type != "meta", "flops_per_token() requires a real device, not meta"

    # Local generator: keeps global + CUDA RNG state untouched, so measurement
    # cannot perturb downstream model init or sampling.
    gen = torch.Generator(device=device).manual_seed(0)
    x = torch.randint(0, vocab_size, (1, seq_len), generator=gen, device=device)
    y = torch.randint(0, vocab_size, (1, seq_len), generator=gen, device=device)

    model.zero_grad()
    with FlopCounterMode(display=False) as fc:
        loss = model(x, y)
        loss.backward()
        total_flops = fc.get_total_flops()
    model.zero_grad()

    return total_flops / seq_len


def tokens_per_step(device_batch_size: int, seq_len: int, world_size: int, grad_accum_steps: int) -> int:
    """Tokens consumed per optimizer step; also the per-step multiplier for a bin
    loader's `expected_tokens`."""
    return device_batch_size * seq_len * world_size * grad_accum_steps


def iterations_for_budget(
    *,
    flops_budget: float,
    flops_per_token: float,
    tokens_per_step: int,
) -> int:
    """Number of training steps that fit in `flops_budget` (floor)."""
    assert flops_budget >= 0, f"flops_budget must be non-negative, got {flops_budget}"
    assert flops_per_token > 0, f"flops_per_token must be positive, got {flops_per_token}"
    assert tokens_per_step > 0, f"tokens_per_step must be positive, got {tokens_per_step}"
    return int(flops_budget // (flops_per_token * tokens_per_step))


def iterations_for_tokens(*, tokens: int, tokens_per_step: int) -> int:
    """Number of optimizer steps to consume `tokens` (floor)."""
    assert tokens > 0, f"tokens must be positive, got {tokens}"
    assert tokens_per_step > 0, f"tokens_per_step must be positive, got {tokens_per_step}"
    return tokens // tokens_per_step
