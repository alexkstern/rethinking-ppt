"""Queue paper experiments from ``reproduce/cells.json`` and train them.

Builds one cell per record (or every record, with ``--all``) and runs each at
every requested seed through ``reproduce.train``. ``--validate`` does the build
alone — every record produces a config, and the recorded effective batch splits
into a per-device micro-batch on ``--nproc`` GPUs — with no W&B, network or GPU.

Every selected cell is built before the first one trains, so a record that
cannot build fails the sweep up front. A marker file per run under
``outputs/<dest>/done/`` makes a relaunch skip runs that already finished;
``--fresh`` clears the selected runs' markers first.

    uv run python -m reproduce.launch scaling__c4_1_1e18 --dry-run
    uv run python -m reproduce.launch --all --validate

``scripts/train.sh`` sets up the environment and runs this.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from experiments.tuner.training import torchrun_cmd

from reproduce.cells import DEFAULT_DEST, Cell, build_cell, load_records, run_name

TRAIN_ENTRY = "reproduce.train"


def command(cell_id: str, *, seed: int, dest: str, nproc: int,
            keep_checkpoints: bool = False,
            max_device_batch: int | None = None,
            wandb: bool = True) -> list[str]:
    """The ``torchrun`` command that trains one cell at one seed."""
    argv = ["--cell-id", cell_id, "--seed", str(seed), "--dest", dest]
    if keep_checkpoints:
        argv.append("--keep-checkpoints")
    if max_device_batch is not None:
        argv += ["--max-device-batch", str(max_device_batch)]
    if not wandb:
        argv.append("--no-wandb")
    return ["uv", "run"] + torchrun_cmd(
        nproc=nproc, entry_module=TRAIN_ENTRY, trial_argv=argv,
    )


def build_cells(records: list[dict], *, dest: str, nproc: int,
                max_device_batch: int | None = None
                ) -> tuple[list[Cell], list[tuple[str, str]]]:
    """Every record built into a cell (at seed 0), and the ones that would not build."""
    cells, failures = [], []
    for record in records:
        try:
            cells.append(build_cell(record, seed=0, dest=dest, world_size=nproc,
                                    max_device_batch=max_device_batch))
        except (ValueError, KeyError, TypeError) as e:
            failures.append((record["cell_id"], f"{type(e).__name__}: {e}"))
    return cells, failures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cell_id", nargs="?", help="a cell_id from reproduce/cells.json")
    ap.add_argument("--all", action="store_true", help="every cell in cells.json")
    ap.add_argument("--setting", metavar="NAME",
                    help="every cell of one setting, e.g. additive (see the "
                         "`setting` field in cells.json)")
    ap.add_argument("--dest", default=DEFAULT_DEST,
                    help="destination W&B project for the replicas (default: %(default)s)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--nproc", type=int, default=4,
                    help="GPU count: 1, 2 or 4 (default: %(default)s)")
    ap.add_argument("--max-device-batch", type=int, default=None, metavar="N",
                    help="cap the per-GPU micro-batch below the recorded one; the "
                         "effective batch is kept by gradient accumulation")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-checkpoints", action="store_true",
                    help="leave each run's weights on disk, for reproduce.evaluate")
    ap.add_argument("--validate", action="store_true",
                    help="check that every planned cell builds its config, then exit")
    ap.add_argument("--fresh", action="store_true",
                    help="clear the selected runs' done markers and train them again")
    ap.add_argument("--no-wandb", action="store_true",
                    help="train without logging to W&B")
    args = ap.parse_args()

    if sum(map(bool, (args.cell_id, args.all, args.setting))) != 1:
        sys.exit("pass exactly one of a cell_id, --all, or --setting")

    records = load_records()
    if args.setting:
        selected = [r for r in records.values() if r["setting"] == args.setting]
        if not selected:
            settings = sorted({r["setting"] for r in records.values()})
            sys.exit(f"no cells with setting {args.setting!r} "
                     f"(known settings: {', '.join(settings)})")
        print(f"--setting {args.setting} covers {len(selected)} cells")
    elif args.all:
        selected = list(records.values())
    else:
        if args.cell_id not in records:
            sys.exit(f"unknown cell_id {args.cell_id!r} (see reproduce/cells.json for valid ids)")
        selected = [records[args.cell_id]]

    cells, failures = build_cells(selected, dest=args.dest, nproc=args.nproc,
                                  max_device_batch=args.max_device_batch)

    for cell in cells:
        print(f"OK  {cell.cell_id}")
        print(f"      {cell.summary()}")
    for cell_id, reason in failures:
        print(f"BAD {cell_id}: {reason}")
    if args.all:
        print(f"\n{len(cells)} ok, {len(failures)} bad, {len(selected)} total")

    if failures:
        sys.exit(1)
    if args.validate:
        sys.exit(0)

    done_dir = Path("outputs") / args.dest / "done"
    if not args.dry_run:
        done_dir.mkdir(parents=True, exist_ok=True)
    if args.fresh and not args.dry_run:
        for cell in cells:
            for seed in args.seeds:
                (done_dir / run_name(cell.cell_id, seed)).unlink(missing_ok=True)
    launch_failures = 0
    for cell in cells:
        for seed in args.seeds:
            name = run_name(cell.cell_id, seed)
            cmd = command(cell.cell_id, seed=seed, dest=args.dest, nproc=args.nproc,
                          keep_checkpoints=args.keep_checkpoints,
                          max_device_batch=args.max_device_batch,
                          wandb=not args.no_wandb)
            marker = done_dir / name
            if marker.exists():
                print(f"== skip {name} (done)")
                continue
            print(f"== {name}")
            if args.dry_run:
                print("   " + " ".join(cmd))
                continue
            if subprocess.run(cmd).returncode == 0:
                marker.touch()
            else:
                launch_failures += 1
                print(f"!! {name} FAILED — continuing", file=sys.stderr)
    if launch_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
