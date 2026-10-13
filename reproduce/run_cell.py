"""Queue paper experiments from ``reproduce/cells.json`` and train them.

Builds one cell per record (or every record, with ``--all``) and runs each at
every requested seed through ``reproduce.train``. ``--validate`` does the build
alone — every record produces a config, and the recorded effective batch splits
into a per-device micro-batch on ``--nproc`` GPUs — with no W&B, network or GPU.

A marker file per run under ``outputs/<dest>/done/`` makes a relaunch skip runs
that already finished.

    uv run python -m reproduce.run_cell scaling__c4_1_1e18 --dry-run
    uv run python -m reproduce.run_cell --all --validate
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from experiments.tuner.training import torchrun_cmd

from reproduce.cells import DEFAULT_DEST, Cell, build_cell, load_records, run_name

TRAIN_ENTRY = "reproduce.train"

# Settings `--all` does not queue: an over-Chinchilla cell trains 50B tokens, an
# order of magnitude above everything else, so it is asked for explicitly —
# by cell id, or by `--setting over_chinchilla` for all three.
EXCLUDED_FROM_ALL = ("over_chinchilla",)


def command(cell_id: str, *, seed: int, dest: str, nproc: int,
            keep_checkpoints: bool = False) -> list[str]:
    """The ``torchrun`` command that trains one cell at one seed."""
    argv = ["--cell-id", cell_id, "--seed", str(seed), "--dest", dest]
    if keep_checkpoints:
        argv.append("--keep-checkpoints")
    return ["uv", "run"] + torchrun_cmd(
        nproc=nproc, entry_module=TRAIN_ENTRY, trial_argv=argv,
    )


def build_cells(records: list[dict], *, dest: str, nproc: int
                ) -> tuple[list[Cell], list[tuple[str, str]]]:
    """Every record built into a cell, and the ones that would not build.

    Seed 0 stands for the whole cell: the seed sets weight init and the run
    name, nothing else a build can get wrong.
    """
    cells, failures = [], []
    for record in records:
        try:
            cells.append(build_cell(record, seed=0, dest=dest, world_size=nproc))
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
    ap.add_argument("--max-target-flops", type=float, default=None, metavar="F",
                    help="skip selected cells above this compute budget. The "
                         "over-Chinchilla cells are budgeted in tokens rather "
                         "than FLOPs and carry 0 here, so this never filters "
                         "them — they are the most expensive cells there are")
    ap.add_argument("--nproc", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-checkpoints", action="store_true",
                    help="leave each run's weights on disk, for reproduce.evaluate")
    ap.add_argument("--validate", action="store_true",
                    help="check that every planned cell builds its config, then exit")
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
        selected = [r for r in records.values()
                    if r["setting"] not in EXCLUDED_FROM_ALL]
        held = len(records) - len(selected)
        if held:
            print(f"--all covers {len(selected)} cells; {held} held back "
                  f"({', '.join(EXCLUDED_FROM_ALL)} — ask for these explicitly)")
    else:
        if args.cell_id not in records:
            sys.exit(f"unknown cell_id {args.cell_id!r} (see reproduce/cells.json for valid ids)")
        selected = [records[args.cell_id]]

    if args.max_target_flops is not None:
        kept = [r for r in selected
                if r["config"]["target_flops"] <= args.max_target_flops]
        print(f"--max-target-flops {args.max_target_flops:.3g} keeps "
              f"{len(kept)} of {len(selected)}")
        selected = kept

    cells, failures = build_cells(selected, dest=args.dest, nproc=args.nproc)

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
    done_dir.mkdir(parents=True, exist_ok=True)
    launch_failures = 0
    for cell in cells:
        for seed in args.seeds:
            name = run_name(cell.cell_id, seed)
            cmd = command(cell.cell_id, seed=seed, dest=args.dest, nproc=args.nproc,
                          keep_checkpoints=args.keep_checkpoints)
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
