"""Train one experiment from ``reproduce/cells.json``, at one seed.

The record's ``config`` paths are applied over the training config's defaults.
The effective batch the record names is split into a per-device micro-batch and
a gradient-accumulation count for this launch's GPU count.

    uv run torchrun --standalone --nproc_per_node=4 -m reproduce.train \\
        --cell-id scaling__c4_1_1e18 --seed 0

``reproduce/launch.py`` drives this for a sweep; ``scripts/train.sh`` is the
launcher a reproduction runs.
"""

from __future__ import annotations

import argparse
import sys

from nanochat.runtime import get_dist_info

from reproduce.cells import DEFAULT_DEST, build_cell, load_records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell-id", required=True,
                    help="a cell_id from reproduce/cells.json")
    ap.add_argument("--seed", type=int, required=True,
                    help="weight-init seed; the paper reports three per cell")
    ap.add_argument("--dest", default=DEFAULT_DEST,
                    help="W&B project to log this run to (default: %(default)s)")
    ap.add_argument("--keep-checkpoints", action="store_true",
                    help="leave the trained weights on disk, for reproduce.evaluate "
                         "to score (training removes them by default)")
    ap.add_argument("--max-device-batch", type=int, default=None, metavar="N",
                    help="cap the per-GPU micro-batch below the recorded one")
    ap.add_argument("--no-wandb", action="store_true",
                    help="train without logging to W&B")
    args = ap.parse_args()

    records = load_records()
    if args.cell_id not in records:
        sys.exit(f"unknown cell_id {args.cell_id!r} "
                 f"(see reproduce/cells.json for valid ids)")

    _, _, _, world_size = get_dist_info()
    cell = build_cell(records[args.cell_id], seed=args.seed, dest=args.dest,
                      world_size=world_size, max_device_batch=args.max_device_batch,
                      wandb=not args.no_wandb)
    if args.keep_checkpoints:
        cell.config.cleanup_checkpoints = False
    cell.train(cell.config)


if __name__ == "__main__":
    main()
