# Rethinking Pre-pretraining

This repository is the training factory used for the paper *Rethinking
Pre-pretraining*, which studies pre-pretraining a language model on synthetic
non-natural-language corpora — neural cellular automata, k-Dyck, a
declarative-reference language, plus code and C4 controls — before
pretraining on natural text, under compute-matched comparison. It is a fork
of and built on [nanochat](https://github.com/karpathy/nanochat).

## Layout

- `nanochat/` — the library: models, data sources and loaders, optimizers, eval
- `setup/` — per-corpus data preparation scripts, one directory per corpus
- `experiments/` — the training entry points and the hyperparameter-search harness
- `reproduce/` — `cells.json`, one record per experiment the paper reports; `train.py`, which trains one record; `run_cell.py`, which queues a sweep of them; and `evaluate.py`, which scores a trained checkpoint

## Data

Every corpus is pre-tokenized and hosted on Hugging Face; `nanochat/data/registry.py`
maps each registry key to its dataset and downloads it on first use. The
`setup/` scripts hold the generation and tokenization code behind them.

| Corpus | Dataset |
|---|---|
| FineWeb-Edu, 20B tokens (data-unconstrained, additive and interleave PT) | [`alexkstern/fineweb-nanochatbpe-20B`](https://huggingface.co/datasets/alexkstern/fineweb-nanochatbpe-20B) |
| FineWeb-Edu, 100M tokens (data-constrained and additive PT; prefix of the 20B) | [`alexkstern/fineweb-nanochatbpe-100M`](https://huggingface.co/datasets/alexkstern/fineweb-nanochatbpe-100M) |
| FineWeb-Edu, 55B tokens (over-Chinchilla PT) | [`alexkstern/fineweb-nanochatbpe-55B`](https://huggingface.co/datasets/alexkstern/fineweb-nanochatbpe-55B) |
| NCA (20 epochs over one 16k-rule pool) | [`alexkstern/nca-paper-share20-seq_len_2048-657M`](https://huggingface.co/datasets/alexkstern/nca-paper-share20-seq_len_2048-657M) |
| NCA (200 epochs over the same pool; large additive doses) | [`alexkstern/nca-paper-share200-seq_len_2048-6.5B`](https://huggingface.co/datasets/alexkstern/nca-paper-share200-seq_len_2048-6.5B) |
| k-Dyck, k = 128 | [`alexkstern/dyck-k128-seq_len_2048-1B`](https://huggingface.co/datasets/alexkstern/dyck-k128-seq_len_2048-1B) |
| DeclRef | [`alexkstern/declref-01-declref_01_symbols-2B`](https://huggingface.co/datasets/alexkstern/declref-01-declref_01_symbols-2B) |
| C4 | [`alexkstern/c4-nanochatbpe-10B`](https://huggingface.co/datasets/alexkstern/c4-nanochatbpe-10B) |
| GitHub code | [`alexkstern/github-code-nanochatbpe-1B`](https://huggingface.co/datasets/alexkstern/github-code-nanochatbpe-1B) |

## Reproducing an experiment

`reproduce/cells.json` holds one record per reported experiment — its full
training configuration. `reproduce/train.py` builds that configuration and
trains it:

```bash
bash reproduce/run_one_cell.sh          # one cheap cell, one seed
bash reproduce/run_all_cells.sh         # the cells up to 9e18, one seed each
bash reproduce/run_interleave_cells.sh  # the interleave cells, which all sit above that cap
```

All three set up the environment first and accept `NPROC` for a GPU count
other than four, and `SEEDS="0 1 2"` for the three seeds the paper reports.
The over-Chinchilla cells train 50B tokens and are never in a sweep, so they
are asked for by name through `run_one_cell.sh`'s `CELL`. A finished run drops a
marker under `outputs/`, so relaunching after an interruption resumes rather
than starting over.

The runner underneath launches one `torchrun` per cell and seed, and can show
or check them without training:

```bash
uv run python -m reproduce.run_cell <cell_id> --dry-run       # print the command for one cell
uv run python -m reproduce.run_cell --all --validate          # build every record's training config
```

`cell_id` is a key from `reproduce/cells.json` (for example
`scaling__c4_1_1e18`). Run `uv run python -m reproduce.run_cell --help` for
the full option list.

### Which cells belong to which result

Each record's `setting` names the comparison it belongs to. The three
compute-matched configurations are defined in §3 *Experimental Setup*:

| `setting` | in the paper |
| --- | --- |
| `additive` | §4 *Additional Compute* — PPT tokens added on top of a fixed natural-language budget, so the two arms are not compute-matched |
| `constrained` | §5 *Compute Matched*, the **data-constrained** configuration — natural language fixed at 100M tokens, total budget 1e18 FLOPs |
| `scaling` | §5 *Compute Matched*, the **data-unconstrained** configuration — natural language abundant, swept over 1e18 / 3e18 / 9e18 FLOPs with the model size searched at each budget |
| `over_chinchilla` | §5 *Compute Matched*, the **over-Chinchilla** configuration — 881M parameters on 50B tokens, ~57 tokens per parameter |
| `interleave` | §6 *Retention Across the PPT-to-PT Transition* — the same token budgets, with the PPT corpus mixed into pretraining rather than trained as a separate phase |

A record's `selectors.init_variant` names one of the three weight
initializations ablated in §3, defined in `experiments/training/run.py`:

| `init_variant` | initialization |
| --- | --- |
| `spectral` | the nanochat default: linear weights from `N(0, sigma)` with `sigma = min(1, sqrt(d_out/d_in))/sqrt(d_in)`, token embeddings from `N(0, 1)`, lm_head and residual projections zeroed |
| `n01_embed` | token embeddings from `N(0, 1)`, every other parameter from `N(0, 0.02)` |
| `n002_all` | every parameter from `N(0, 0.02)` |

`selectors.ppt_opt` names the optimizer the PPT phase runs under (`muon` or
`adamw`); the PT phase always uses Muon on the body and AdamW on the
embeddings.

### Corpora

Every corpus a cell references downloads from the Hub on first use, so nothing
needs preparing before a run. The `setup/` scripts regenerate each corpus from
source and publish it; they exist so the provenance of every corpus is
inspectable, not as a prerequisite.

The hyperparameter-search harness (`llmtuna`, used by `experiments/tuner/` and
`experiments/tasks/`) is an optional extra, since it is a git dependency
rather than a published package:

```bash
uv sync --extra optimize
```

## Scoring a checkpoint

Training reports bits-per-byte on held-out text. The benchmark tables report
BLiMP and DCLM CORE, and `reproduce/evaluate.py` produces all three from a
trained checkpoint:

```bash
bash reproduce/run_eval.sh                             # a published checkpoint, all three suites
CHECKPOINT=<dir> bash reproduce/run_eval.sh            # weights you trained
SUITES="--suite bpb" bash reproduce/run_eval.sh        # one suite
```

With no arguments it downloads a published checkpoint, so the path can be
exercised without training anything first. The suites are `bpb` (the held-out
window a run reports), `blimp` (67 minimal-pair grammar paradigms) and `core`
(22 DCLM tasks, scored as centered accuracy).

Training removes its weights when it finishes, so a run you intend to score
has to be asked for:

```bash
uv run python -m reproduce.run_cell <cell_id> --seeds 0 --keep-checkpoints
```

which prints `[checkpoint] kept: <dir>` at the end. For a two-phase cell the
trained model is that directory's `pt/` subdirectory.

The evaluation window is stated rather than read off the checkpoint: a run's
config records the window that run trained under, which is not always the one
its published numbers were scored at. The defaults are the reported window.

## Relationship to nanochat

This repository is a fork of [nanochat](https://github.com/karpathy/nanochat)
at commit
[`3a5e0bc`](https://github.com/karpathy/nanochat/tree/3a5e0bc) (2025-10-13).
Much of the library is that code, moved into packages and extended; the parts
that are substantially upstream's are:

| here | upstream at `3a5e0bc` |
| --- | --- |
| `rustbpe/` | [`rustbpe/`](https://github.com/karpathy/nanochat/tree/3a5e0bc/rustbpe) — the BPE trainer, unchanged but for one line |
| `nanochat/tokenizer.py` | [`nanochat/tokenizer.py`](https://github.com/karpathy/nanochat/blob/3a5e0bc/nanochat/tokenizer.py) |
| `nanochat/common.py` | [`nanochat/common.py`](https://github.com/karpathy/nanochat/blob/3a5e0bc/nanochat/common.py) |
| `nanochat/optim/muon.py` | [`nanochat/muon.py`](https://github.com/karpathy/nanochat/blob/3a5e0bc/nanochat/muon.py) — code unchanged |
| `nanochat/optim/adamw.py` | [`nanochat/adamw.py`](https://github.com/karpathy/nanochat/blob/3a5e0bc/nanochat/adamw.py) |
| `nanochat/models/nanochat_gpt/vanilla_3a5e0bc.py` | [`nanochat/gpt.py`](https://github.com/karpathy/nanochat/blob/3a5e0bc/nanochat/gpt.py) — the model every reported experiment trains |
| `nanochat/eval/core/harness.py` | [`nanochat/core_eval.py`](https://github.com/karpathy/nanochat/blob/3a5e0bc/nanochat/core_eval.py) |
| `nanochat/data/sources/fineweb/source.py` | [`nanochat/dataset.py`](https://github.com/karpathy/nanochat/blob/3a5e0bc/nanochat/dataset.py) |

The model file is named for the commit its architecture matches, so a result
stays attached to the shape that produced it.

What is ours: the two-phase and interleaved training programs, the data
registry and its procedural corpora, the FLOP-budgeted schedule, the
bits-per-byte and BLiMP evaluations, the hyperparameter-search harness, and
the reproduction manifest.

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that $100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## License

MIT
