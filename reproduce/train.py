"""Train one experiment from ``reproduce/cells.json``, at one seed.

The record's ``config`` paths are applied over the training config's defaults
and the result is handed straight to the training program, never rendered to a
command line in between. The effective batch the record names is split into a
per-device micro-batch and a gradient-accumulation count for this launch's GPU
count, so the tokens per optimizer step are the same at any GPU count. Training
is not bitwise deterministic, so a reproduction lands within seed noise of the
reported number rather than exactly on it.

    uv run torchrun --standalone --nproc_per_node=4 -m reproduce.train \\
        --cell-id scaling__c4_1_1e18 --seed 0

``reproduce/run_cell.py`` drives this for a whole sweep.
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
    args = ap.parse_args()

    records = load_records()
    if args.cell_id not in records:
        sys.exit(f"unknown cell_id {args.cell_id!r} "
                 f"(see reproduce/cells.json for valid ids)")

    _, _, _, world_size = get_dist_info()
    cell = build_cell(records[args.cell_id], seed=args.seed, dest=args.dest,
                      world_size=world_size)
    if args.keep_checkpoints:
        cell.config.cleanup_checkpoints = False
    cell.train(cell.config)


if __name__ == "__main__":
    main()
