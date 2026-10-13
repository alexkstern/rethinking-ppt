"""The single-phase training loop.

`train()` is handed already-constructed objects and runs the loop on them. It
builds nothing, loads no checkpoints, and knows nothing about multi-stage
training: an experiment script that wants two stages calls `train()` twice, and
each call returns a metrics dict for HPO and reporting.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Callable

import torch
import torch.distributed as dist
import torch.nn as nn

from nanochat.checkpoint import CheckpointMeta, get_rng_state, save_checkpoint
from nanochat.config import TrainConfig
from nanochat.data.base import DataIterator
from nanochat.eval.base import EvalFn
from nanochat.runtime import RuntimeCtx, print0
from nanochat.wandb_log import build_train_metrics, build_val_metrics

if TYPE_CHECKING:
    # Annotation only; the loop never imports wandb at runtime.
    import wandb


# Throughput timing starts only after this many steps, so one-off warmup costs
# (notably the first compiled-graph executions) do not skew tok/sec and MFU.
_TIMING_WARMUP_STEPS = 10


def _synchronize(device: torch.device) -> None:
    """Block until queued device work finishes, so wall-clock timing is honest."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def _peak_memory_mib(device: torch.device) -> float:
    """Peak allocated device memory in MiB, or 0.0 when there is no device cache."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


# Metric-name globs whose x-axis is the global training `step`, which carries
# the cross-phase offset. Everything a builder emits falls under one of these.
_GLOBAL_METRIC_GLOBS = ("train/*", "val/*", "flops/*", "data/*")
# Envelope fields describing the global axis; a mirrored payload replaces them
# with phase-local equivalents rather than copying them.
_GLOBAL_AXIS_FIELDS = ("step", "total_training_flops", "total_training_time")


def _define_wandb_axes(wandb_run, metric_prefix: str | None) -> None:
    """Declare custom x-axes so charts plot against the training step rather than
    wandb's internal log counter. The global `step` carries the cross-phase
    offset, so calls sharing a run concatenate on one axis; `metric_prefix` adds
    a phase-local axis reading from 0. Idempotent."""
    wandb_run.define_metric("step")
    wandb_run.define_metric("total_training_flops")
    for glob in _GLOBAL_METRIC_GLOBS:
        wandb_run.define_metric(glob, step_metric="step")
    if metric_prefix:
        wandb_run.define_metric(f"{metric_prefix}/step")
        wandb_run.define_metric(f"{metric_prefix}/total_training_flops")
        wandb_run.define_metric(f"{metric_prefix}/*", step_metric=f"{metric_prefix}/step")


def _mirror_under_prefix(
    payload: dict, metric_prefix: str, *, local_step: int, local_flops: float
) -> dict:
    """Re-key a global metric payload under `{metric_prefix}/…` on a phase-local
    axis."""
    mirrored = {
        f"{metric_prefix}/step": local_step,
        f"{metric_prefix}/total_training_flops": local_flops,
    }
    for key, value in payload.items():
        if key in _GLOBAL_AXIS_FIELDS:
            continue
        mirrored[f"{metric_prefix}/{key}"] = value
    return mirrored


def train(
    train_config: TrainConfig,
    *,
    model: nn.Module,
    optimizers: list[torch.optim.Optimizer],
    train_loader: DataIterator,
    eval_fns: list[EvalFn],
    objective_key: str = "val/bpb",        # which eval metric to minimize + return
    lr_schedule: Callable[[int], float],   # maps step -> lr multiplier
    ctx: RuntimeCtx,                        # rank / local_rank / world_size / device
    flops_per_token: float,
    tokens_per_step: int,
    grad_accum_steps: int,
    peak_tflops: float,
    flops_budget: float | None = None,
    wandb_run: wandb.Run | None = None,
    starting_step: int = 0,
    starting_min_objective: float | None = float("inf"),
    starting_smooth_train_loss: float = 0.0,
    starting_total_training_time: float = 0.0,
    step_offset: int = 0,
    flops_offset: float = 0.0,
    metric_prefix: str | None = None,
    checkpoint_dir: str | None = None,
    run_config: object | None = None,
) -> dict:
    """Run the loop for `train_config.num_iterations` optimizer steps and return
    `{min_objective, objective, total_training_flops, total_training_time,
    final_step, peak_memory_mib}`.

    `model(inputs, targets)` must return a scalar loss. `train_loader` yields
    `(inputs, targets, state)`, where `state` is the loader's resume token or
    `None`. The `eval_fns` dicts are merged and logged verbatim, so the loop
    never needs to know their metric names; the one requirement is that some
    eval fn emits `objective_key`.

    Resume contract:
      - `starting_step` is the absolute step counter to continue from. `train()`
        does NOT reposition `train_loader`; the caller must hand it a loader
        already advanced to that step. The bin/dyck loaders derive their read
        position from an internal counter starting at 0, so a run using them
        re-reads from the start unless the loader is made resume-aware.
      - `starting_min_objective` may be `None` (no eval yet) and is then treated
        as +inf.
      - `starting_smooth_train_loss` is valid only when `starting_step` also
        continues, i.e. a same-phase crash resume, because the debias assumes
        the EMA is `starting_step + 1` updates old. At a phase boundary, where
        `starting_step` restarts at 0, leave it 0.0.

    `step_offset` and `flops_offset` are added to the LOGGED `step` and
    `total_training_flops` and to nothing else, so phases sharing one W&B run
    get a continuous axis while the loop's own counters stay phase-local. With
    `metric_prefix` set, every payload is additionally mirrored under
    `{metric_prefix}/…` on a phase-local axis, giving each phase a section that
    reads from step 0.

    `checkpoint_dir` and `run_config` (a dataclass exposing `seed`) are required
    only if a save cadence fires.
    """
    # The loop stops on `step == num_iterations`, so a starting_step past that
    # would never be reached and the loop would hang silently.
    if not 0 <= starting_step <= train_config.num_iterations:
        raise ValueError(
            f"starting_step must be in [0, num_iterations={train_config.num_iterations}], "
            f"got {starting_step}"
        )
    if train_config.save_every_n_steps is not None or train_config.save_at_end:
        if checkpoint_dir is None:
            raise ValueError("checkpoint_dir is required when a save cadence is configured")
        if run_config is None:
            raise ValueError("run_config is required when a save cadence is configured")

    # Carried across a resume so the EMA, wall-clock total and running-min
    # objective continue rather than restart.
    smooth_train_loss = starting_smooth_train_loss
    total_training_time = starting_total_training_time
    min_objective = starting_min_objective if starting_min_objective is not None else float("inf")
    last_objective = float("inf")

    if wandb_run is not None:
        _define_wandb_axes(wandb_run, metric_prefix)

    step = starting_step
    inputs, targets, dataloader_state = next(train_loader)

    # The loop body runs num_iterations + 1 times: the trailing pass does only
    # the end-of-training eval/save before breaking, so those opt-ins see the
    # model in its final state.
    while True:
        if step == train_config.num_iterations:
            last_step = True
        else:
            last_step = False
        # FLOPs spent so far in THIS call; cumulative compute across phases is
        # the experiment script's concern.
        flops_so_far = flops_per_token * tokens_per_step * step

        is_periodic_eval = (
            train_config.eval_every_n_steps is not None and step % train_config.eval_every_n_steps == 0
        )
        is_eval_at_steps_eval = (
            train_config.eval_at_steps is not None and step in train_config.eval_at_steps
        )
        is_final_eval = last_step and train_config.eval_at_end
        do_eval = is_periodic_eval or is_eval_at_steps_eval or is_final_eval
        if do_eval:
            model.eval()
            eval_results = {}  # merged across eval fns
            for eval_fn in eval_fns:
                eval_results.update(eval_fn(model))
            if objective_key not in eval_results:
                raise KeyError(
                    f"objective_key {objective_key!r} not emitted by any eval fn; "
                    f"available keys: {sorted(eval_results)}"
                )
            objective = eval_results[objective_key]
            last_objective = objective
            min_objective = min(min_objective, objective)
            print0(f"step {step:05d} | {objective_key}: {objective:.4f}")
            if wandb_run is not None:
                val_payload = build_val_metrics(
                    step + step_offset,
                    flops_so_far + flops_offset,
                    total_training_time,
                    eval_results,
                )
                if metric_prefix:
                    val_payload.update(_mirror_under_prefix(
                        val_payload, metric_prefix, local_step=step, local_flops=flops_so_far,
                    ))
                wandb_run.log(val_payload)
            model.train()

        # `step != starting_step` keeps a resumed run from immediately re-saving
        # the checkpoint it just loaded.
        is_periodic_save = (
            train_config.save_every_n_steps is not None
            and step > 0
            and step != starting_step
            and step % train_config.save_every_n_steps == 0
        )
        is_save_at_steps_save = (
            train_config.save_at_steps is not None
            and step > 0
            and step != starting_step
            and step in train_config.save_at_steps
        )
        is_final_save = last_step and train_config.save_at_end
        do_save = is_periodic_save or is_save_at_steps_save or is_final_save
        if do_save:
            meta = CheckpointMeta(
                step=step,
                phase=None,
                seed=run_config.seed,
                flops_used=flops_so_far,
                flops_per_token=flops_per_token,
                iterations_planned=train_config.num_iterations,
                dataloader_state=dataloader_state,
                wandb_run_id=wandb_run.id if wandb_run is not None else None,
                # inf would serialize as non-strict-JSON `Infinity`.
                min_objective=min_objective if math.isfinite(min_objective) else None,
                smooth_train_loss=smooth_train_loss,
                total_training_time=total_training_time,
            )
            save_checkpoint(
                checkpoint_dir,
                step,
                model=model,
                optimizers=optimizers,
                meta=meta,
                rng_state=get_rng_state(),
                config=run_config,
                rank=ctx.rank,
            )
            if ctx.world_size > 1:
                dist.barrier()

        if last_step:
            break

        _synchronize(ctx.device)
        t0 = time.time()
        for micro_step in range(grad_accum_steps):
            loss = model(inputs, targets)
            train_loss = loss.detach()
            # Backwards sum into .grad, so normalize to keep the effective loss
            # a mean over the full optimizer-step batch.
            (loss / grad_accum_steps).backward()
            # Prefetch the next batch while the backward is still in flight.
            inputs, targets, dataloader_state = next(train_loader)

        if train_config.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_config.grad_clip
            ).item()
        else:
            grad_norm = None

        lrm = lr_schedule(step)
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * lrm
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
        _synchronize(ctx.device)
        # Floored: a step faster than the clock's resolution gives dt == 0.0,
        # which would divide by zero in tok/sec and MFU below.
        dt = max(time.time() - t0, 1e-9)

        # EMA of the LOGGED train loss only, debiased for its cold start. This
        # does not affect optimization.
        ema_beta = train_config.ema_beta
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
        debiased_train_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
        if step > _TIMING_WARMUP_STEPS:
            total_training_time += dt

        tok_per_sec = int(tokens_per_step / dt)
        mfu = (flops_per_token * tokens_per_step / dt) / (peak_tflops * 1e12 * ctx.world_size)
        total_flops = flops_per_token * tokens_per_step * (step + 1)

        if step % train_config.log_every_n_steps == 0:
            pct_done = 100 * step / train_config.num_iterations
            grad_norm_str = f" grad norm: {grad_norm:.4f} |" if grad_norm is not None else ""
            print0(
                f"step {step:05d}/{train_config.num_iterations:05d} ({pct_done:.2f}%) | "
                f"loss: {debiased_train_loss:.6f} |{grad_norm_str} lrm: {lrm:.2f} | "
                f"dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | "
                f"mfu: {mfu:.4f} | total time: {total_training_time / 60:.2f}m"
            )
            flops_budget_remaining = (
                flops_budget - total_flops if flops_budget is not None else None
            )
            if wandb_run is not None:
                train_payload = build_train_metrics(
                    step + step_offset, total_flops + flops_offset,
                    total_training_time, debiased_train_loss,
                    lrm, dt, tok_per_sec, mfu,
                    grad_norm=grad_norm,
                    flops_budget_remaining=flops_budget_remaining,
                )
                if metric_prefix:
                    train_payload.update(_mirror_under_prefix(
                        train_payload, metric_prefix, local_step=step, local_flops=total_flops,
                    ))
                wandb_run.log(train_payload)

        step += 1

    peak_memory_mib = _peak_memory_mib(ctx.device)
    print0(f"peak memory: {peak_memory_mib:.2f}MiB")
    print0(f"total training time: {total_training_time / 60:.2f}m")
    print0(f"min {objective_key}: {min_objective:.4f}")

    return {
        "min_objective": min_objective,
        "objective": last_objective,
        "total_training_flops": flops_per_token * tokens_per_step * step,
        "total_training_time": total_training_time,
        "final_step": step,
        "peak_memory_mib": peak_memory_mib,
    }
