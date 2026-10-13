"""The paper's experiment manifest, and how one record becomes a training config.

``cells.json`` holds one record per reported experiment. Three parts build
the run:

``config``
    Dotted field paths into the training config (``depth``,
    ``optim.matrix_lr``, ``model_pt.sequence_len``, ...), assigned over the
    config dataclass's own defaults. A path naming no field raises.

``selectors``
    The three things that are not settable fields:

    * ``entry`` — which training program runs the cell.
    * ``init_variant`` / ``ppt_opt`` — the model spec the phases build under,
      and whether the pre-pretraining (PPT) phase optimizes every parameter
      group with stock AdamW instead of the pretraining recipe.
    * ``pt_batch`` / ``ppt_batch`` — effective tokens per optimizer step. The
      per-device split (``device_batch_size`` x ``grad_accum_steps``) follows
      from the GPU count the cell is launched on. ``ppt_batch`` is null when
      the PPT phase trains at the PT phase's batch.

``micro_batch``
    The per-GPU micro-batch (sequences per forward pass) each phase of the
    reported runs used, and the GPU count they ran on. The split follows from
    it: the largest divisor of the per-GPU sequences that is at most this, with
    gradient accumulation making up the rest.

``checkpoints`` (the published model per seed) and ``fineweb_bpb_scored_with``
are read by ``reproduce.evaluate``.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from experiments.training.config import ExperimentConfig
from experiments.training.interleave import InterleaveConfig
from experiments.training.interleave import main as run_interleaved
from experiments.training.two_phase import main as run_two_phase
from experiments.training.two_phase import resolve_model_specs
from experiments.tuner.batch_plan import plan_microbatch
from nanochat.models.nanochat_gpt.vanilla_3a5e0bc import MODEL_SPEC

CELLS_PATH = Path(__file__).with_name("cells.json")

# W&B project the replicas are logged to.
DEFAULT_DEST = "rethinking_ppt"

# Every name a record's ``selectors`` block carries; a record carries all of them.
SELECTORS = ("entry", "init_variant", "ppt_opt", "pt_batch", "ppt_batch")


@dataclass(frozen=True)
class Cell:
    """One manifest record, built into the config that trains it."""

    cell_id: str
    setting: str
    run_name: str
    config: ExperimentConfig
    train: Callable[[ExperimentConfig], None]
    # Effective tokens per optimizer step, per phase, as the manifest records
    # them; ``pt_batch`` is the one the config's ``hardware`` block carries.
    pt_batch: int
    ppt_batch: int | None

    def summary(self) -> str:
        """The values this cell trains at, read off the built config."""
        c = self.config
        parts = [f"depth={c.depth}", f"matrix_lr={c.optim.matrix_lr}",
                 f"warmdown={c.lr_warmdown_ratio}", f"warmup={c.lr_warmup_ratio}"]
        if isinstance(c, InterleaveConfig):
            parts.append(f"pure_code_frac={c.pure_code_frac}")
        elif c.data_ppt is not None:
            parts += [f"ppt_lr={c.ppt_lr}", f"ppt_warmdown={c.ppt_lr_warmdown_ratio}",
                      f"alpha_ppt={c.alpha_ppt}"]
        parts.append(f"eff_batch={self.pt_batch}")
        return "  ".join(parts)


def load_records() -> dict[str, dict]:
    """cell_id -> record, for every experiment the paper reports."""
    cells = json.loads(CELLS_PATH.read_text())["cells"]
    return {record["cell_id"]: record for record in cells}


def run_name(cell_id: str, seed: int) -> str:
    """The name one cell/seed trains under: its W&B run name and its done
    marker. The checkpoint directory is this plus a timestamp (see
    `nanochat.paths.resolve_checkpoint_dir`)."""
    return f"{cell_id.replace('__', '_')}_s{seed}"


def _set_path(cfg: Any, path: str, value: Any, *, cell_id: str) -> None:
    """Assign ``value`` at the dotted field ``path`` of the config ``cfg``.

    Every component must name a field of the dataclass it is read from; one
    that does not is a manifest error and raises with the cell named.
    """
    obj = cfg
    parts = path.split(".")
    for i, name in enumerate(parts):
        fields = ({f.name for f in dataclasses.fields(obj)}
                  if dataclasses.is_dataclass(obj) else set())
        if name not in fields:
            raise ValueError(
                f"{cell_id}: config path {path!r} — {'.'.join(parts[:i + 1])} is not "
                f"a field of {type(obj).__name__}"
            )
        if i == len(parts) - 1:
            setattr(obj, name, value)
        else:
            obj = getattr(obj, name)


def _split_batch(total_batch_tokens: int, *, world_size: int, seq_len: int,
                 micro_batch: int, cell_id: str) -> tuple[int, int]:
    """Effective tokens per step -> the (device_batch_size, grad_accum_steps)
    that deliver exactly that many tokens across ``world_size`` GPUs, with the
    micro-batch at most ``micro_batch`` sequences."""
    seqs, rem = divmod(total_batch_tokens, world_size * seq_len)
    if rem:
        raise ValueError(
            f"{cell_id}: {total_batch_tokens} tokens per step do not split into "
            f"whole {seq_len}-token sequences across {world_size} GPUs"
        )
    dbs, accum = plan_microbatch(0, seqs, schedule=[(0, micro_batch)])
    got = dbs * accum * seq_len * world_size
    if got != total_batch_tokens:
        raise ValueError(
            f"{cell_id}: batch split gives {got} tokens per step, not the "
            f"recorded {total_batch_tokens}"
        )
    return dbs, accum


def build_cell(record: dict, *, seed: int, dest: str, world_size: int,
               max_device_batch: int | None = None, wandb: bool = True) -> Cell:
    """One manifest record, built into the config and program that train it.

    ``max_device_batch`` lowers the recorded micro-batch; the effective batch
    is kept.
    ``wandb=False`` trains without logging, for a machine with no W&B login.
    """
    cell_id = record["cell_id"]
    selectors = record["selectors"]
    if set(selectors) != set(SELECTORS):
        raise ValueError(
            f"{cell_id}: selectors are {sorted(selectors)}, expected exactly "
            f"{sorted(SELECTORS)} — a name nothing reads would be silently dropped"
        )

    entry = selectors["entry"]
    if entry == "two_phase":
        cfg: ExperimentConfig = ExperimentConfig()
    elif entry == "interleaved":
        cfg = InterleaveConfig()
    else:
        raise ValueError(
            f"{cell_id}: unknown entry {entry!r} (expected 'two_phase' or 'interleaved')"
        )

    for path, value in record["config"].items():
        _set_path(cfg, path, value, cell_id=cell_id)

    variant = selectors["init_variant"]
    ppt_opt = selectors["ppt_opt"]
    if (ppt_opt is None) != (cfg.data_ppt is None):
        raise ValueError(
            f"{cell_id}: ppt_opt {ppt_opt!r} and data_ppt {cfg.data_ppt!r} disagree "
            f"about whether this cell has a PPT phase"
        )
    try:
        model_spec, ppt_model_spec = resolve_model_specs(variant, ppt_opt)
    except ValueError as e:
        raise ValueError(f"{cell_id}: {e}") from e

    if entry == "interleaved":
        # The interleaved program serves both corpora from one model built by
        # the pretraining spec, and has no separate PPT-phase optimizer.
        if model_spec is not MODEL_SPEC or ppt_model_spec is not None:
            raise ValueError(
                f"{cell_id}: the interleaved program builds one model under the "
                f"pretraining spec, so init_variant must be 'spectral' and ppt_opt "
                f"'muon'; got {variant!r} / {ppt_opt!r}"
            )
        train = run_interleaved
    else:
        train = partial(run_two_phase, model_spec=model_spec, ppt_model_spec=ppt_model_spec)

    if cfg.depth is None:
        raise ValueError(f"{cell_id}: the record sets no depth")
    # The sharded AdamW splits embedding rows across ranks, so every vocab
    # width the cell trains must divide by the GPU count.
    for label, width in (("pad_vocab", cfg.pad_vocab), ("ppt_vocab_size", cfg.ppt_vocab_size)):
        if width is not None and width % world_size:
            raise ValueError(
                f"{cell_id}: {label} {width} is not divisible by {world_size} GPUs; "
                f"this cell trains on a GPU count that divides it (1, 2 or 4)"
            )
    seq_len = cfg.model_pt.sequence_len
    pt_batch = selectors["pt_batch"]
    ppt_batch = selectors["ppt_batch"]
    micro = record["micro_batch"]
    cap = (lambda n: n if max_device_batch is None else min(n, max_device_batch))
    cfg.hardware.device_batch_size, cfg.hardware.grad_accum_steps = _split_batch(
        pt_batch, world_size=world_size, seq_len=seq_len,
        micro_batch=cap(micro["pt"]), cell_id=cell_id)
    if ppt_batch is not None:
        cfg.ppt_device_batch_size, cfg.ppt_grad_accum_steps = _split_batch(
            ppt_batch, world_size=world_size, seq_len=seq_len,
            micro_batch=cap(micro["ppt"]), cell_id=cell_id)

    name = run_name(cell_id, seed)
    cfg.seed = seed
    cfg.run_name = name
    cfg.model_project = dest
    cfg.wandb.enabled = wandb

    return Cell(cell_id=cell_id, setting=record["setting"], run_name=name, config=cfg,
                train=train, pt_batch=pt_batch, ppt_batch=ppt_batch)
