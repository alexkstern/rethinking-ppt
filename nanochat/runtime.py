"""Process lifecycle helpers.

`setup()` / `cleanup()` bracket an entire experiment, however many training
phases happen in between; `release_phase()` cleans up between multiple
`torch.compile()` invocations in the same process.
"""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass

import torch
import torch._dynamo as dynamo
import torch.distributed as dist


@dataclass(frozen=True)
class RuntimeCtx:
    """Immutable record of the per-process runtime configuration."""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    # Sampled once on rank 0 and broadcast, so every rank composes the same
    # per-run paths even if their local clocks straddle a boundary at startup.
    run_timestamp: str


def _timestamp_string() -> str:
    """Microsecond resolution, so two same-second relaunches of one `run_name`
    do not collide on the per-run dir."""
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")


def _broadcast_run_timestamp(rank: int) -> str:
    """Sample on rank 0 and broadcast; falls back to a local sample when no
    process group is initialized."""
    if not dist.is_initialized():
        return _timestamp_string()
    payload = [_timestamp_string() if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def is_ddp() -> bool:
    # `torchrun` sets RANK; its absence means single-process.
    return "RANK" in os.environ


def get_dist_info() -> tuple[bool, int, int, int]:
    """(ddp, rank, local_rank, world_size). Single-process => (False, 0, 0, 1)."""
    if is_ddp():
        # Partial env is a launcher bug; fail loud.
        assert all(var in os.environ for var in ["RANK", "LOCAL_RANK", "WORLD_SIZE"])
        return (
            True,
            int(os.environ["RANK"]),
            int(os.environ["LOCAL_RANK"]),
            int(os.environ["WORLD_SIZE"]),
        )
    return False, 0, 0, 1


def autodetect_device_type() -> str:
    """Prefer CUDA, then MPS, otherwise CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def print0(msg: object = "", **kwargs) -> None:
    """Rank-0 print, reading RANK from env so callers need no RuntimeCtx."""
    rank = int(os.environ["RANK"]) if "RANK" in os.environ else 0
    if rank == 0:
        print(msg, **kwargs)


def setup(seed: int = 0) -> RuntimeCtx:
    """Initialize the process: device, seeds, TF32, and (if DDP on CUDA) NCCL."""
    device_type = autodetect_device_type()
    if device_type == "cuda":
        assert torch.cuda.is_available(), "CUDA requested but unavailable"
    if device_type == "mps":
        assert torch.backends.mps.is_available(), "MPS requested but unavailable"

    # These mostly govern nn.Module weight init; the rest of the code uses
    # explicit torch.Generator objects.
    torch.manual_seed(seed)
    if device_type == "cuda":
        torch.cuda.manual_seed(seed)
        # "high" (vs the "highest" default) enables TF32 on Ampere and newer:
        # 10-bit mantissa with fp32-range exponent, 2-3x faster at negligible
        # accuracy cost. Matters where ops fall back to fp32 (some RoPE / norm
        # internals); BF16 autocast covers the rest.
        torch.set_float32_matmul_precision("high")

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)  # make `cuda` default to this rank's GPU
        # `device_id` eagerly binds the NCCL comm to this GPU.
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()  # surface NCCL setup failures here rather than mid-training
    else:
        device = torch.device(device_type)

    # After the process group is up, so the broadcast can use it.
    run_timestamp = _broadcast_run_timestamp(ddp_rank)

    return RuntimeCtx(
        rank=ddp_rank,
        local_rank=ddp_local_rank,
        world_size=ddp_world_size,
        device=device,
        run_timestamp=run_timestamp,
    )


def cleanup() -> None:
    """Tear down the process group if one exists. Safe to call unconditionally."""
    if dist.is_initialized():
        dist.destroy_process_group()


def release_phase(*objs) -> None:
    """Drop refs, reset Dynamo, free GPU cache, between `torch.compile()`
    invocations in the same process.

    The compiled-model wrapper holds a ref to the original model, so a plain
    `del model` at the call site does not free GPU memory; Dynamo must be reset
    too. `*objs` is documentation of what the caller intends to drop — the
    bindings themselves only go away when they fall out of scope there.
    """
    del objs
    dynamo.reset()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
