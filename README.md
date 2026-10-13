# Rethinking Pre-pretraining

Code for the paper *Rethinking Pre-pretraining*. It studies pre-pretraining
(PPT) a language model on synthetic data before pretraining (PT) it on natural
text, under compute-matched comparison. The PPT corpora are neural cellular
automata (NCA), k-Dyck, and DeclRef (short for declarative reference), a
language we created for this work, with code and C4 as controls. The code is a
fork of [nanochat](https://github.com/karpathy/nanochat).

## Layout

- `nanochat/` — models, data sources and loaders, optimizers and evaluation
- `setup/` — the scripts that generated and tokenized each corpus
- `experiments/` — the training programs and the hyperparameter-search harness
- `reproduce/` — `cells.json`, one record per experiment the paper reports
- `scripts/` — `train.sh`, which trains experiments from `cells.json`, and `eval.sh`, which scores a model

## Data

Every corpus is pre-tokenized on the Hugging Face Hub and downloads on first
use; `nanochat/data/registry.py` maps each name a run uses to its dataset.

| Corpus | Dataset |
|---|---|
| FineWeb-Edu, 20B tokens (data-unconstrained, additive and interleave PT) | [`alexkstern/fineweb-nanochatbpe-20B`](https://huggingface.co/datasets/alexkstern/fineweb-nanochatbpe-20B) |
| FineWeb-Edu, 100M tokens (data-constrained and additive PT; the first 100M tokens of the 20B) | [`alexkstern/fineweb-nanochatbpe-100M`](https://huggingface.co/datasets/alexkstern/fineweb-nanochatbpe-100M) |
| FineWeb-Edu, 55B tokens (over-Chinchilla PT) | [`alexkstern/fineweb-nanochatbpe-55B`](https://huggingface.co/datasets/alexkstern/fineweb-nanochatbpe-55B) |
| NCA (20 epochs over one 16k-rule pool) | [`alexkstern/nca-paper-share20-seq_len_2048-657M`](https://huggingface.co/datasets/alexkstern/nca-paper-share20-seq_len_2048-657M) |
| NCA (the same recipe at 200 epochs) | [`alexkstern/nca-paper-share200-seq_len_2048-6.5B`](https://huggingface.co/datasets/alexkstern/nca-paper-share200-seq_len_2048-6.5B) |
| k-Dyck, k = 128 | [`alexkstern/dyck-k128-seq_len_2048-1B`](https://huggingface.co/datasets/alexkstern/dyck-k128-seq_len_2048-1B) |
| DeclRef | [`alexkstern/declref-01-declref_01_symbols-2B`](https://huggingface.co/datasets/alexkstern/declref-01-declref_01_symbols-2B) |
| C4 | [`alexkstern/c4-nanochatbpe-10B`](https://huggingface.co/datasets/alexkstern/c4-nanochatbpe-10B) |
| GitHub code | [`alexkstern/github-code-nanochatbpe-1B`](https://huggingface.co/datasets/alexkstern/github-code-nanochatbpe-1B) |

## Training

`reproduce/cells.json` holds the full training configuration of every
experiment the paper reports, and `scripts/train.sh` trains them:

```bash
bash scripts/train.sh scaling__c4_1_1e18 --seeds 0   # one cell, one seed
bash scripts/train.sh scaling__c4_1_1e18             # one cell, three seeds
bash scripts/train.sh --setting interleave           # one setting (see below)
bash scripts/train.sh --all                          # every cell
```

Training runs on 4 GPUs by default; `--nproc N` sets 1 or 2 instead.
`--max-device-batch N` lowers the per-GPU micro-batch, keeping the effective
batch the same.
Runs log to W&B unless you pass `--no-wandb`. Each run keeps its final weights
and prints where they are, for `eval.sh`.
`uv run python -m reproduce.launch --help` lists every option.

### Which cells belong to which result

Each record's `setting` names the comparison it belongs to:

| `setting` | in the paper |
| --- | --- |
| `additive` | §4 *Additive Compute* — PPT tokens added on top of a fixed natural-language budget |
| `constrained` | §5 *Compute-Matched*, the **data-constrained** configuration — natural language fixed at 100M tokens, total budget 1e18 FLOPs |
| `scaling` | §5 *Compute-Matched*, the **data-unconstrained** configuration — 1e18, 3e18 and 9e18 FLOPs, with the model size searched at each budget |
| `over_chinchilla` | §5 *Compute-Matched*, the **over-Chinchilla** configuration — 881M parameters on 50B tokens |
| `interleave` | §6 *Retention Across the PPT-to-PT Transition* — 2.7e19 FLOPs with 5% of compute on NCA, k-Dyck or code, either as a PPT phase first (`*_phase`) or mixed into every batch (`*_interleaved`) |

A record's `selectors.init_variant` names one of the three weight
initializations defined in §3, implemented in
`experiments/training/two_phase.py`:

| `init_variant` | initialization |
| --- | --- |
| `spectral` | linear weights from `N(0, sigma)` with `sigma = min(1, sqrt(d_out/d_in))/sqrt(d_in)`, token embeddings from `N(0, 1)`, lm_head and residual projections zeroed (the nanochat default) |
| `n01_embed` | token embeddings from `N(0, 1)`, every other parameter from `N(0, 0.02)` |
| `n002_all` | every parameter from `N(0, 0.02)` |

The hyperparameter-search harness (`llmtuna`, used by `experiments/tuner/`
and `experiments/tasks/`) is a git dependency, installed with
`uv sync --extra optimize`.

## Scoring a model

Every model the paper reports is on the Hugging Face Hub as
`alexkstern/rppt-<cell>-seed<n>`, listed under `checkpoints` in `cells.json`.
`scripts/eval.sh` scores a published model or one you trained:

```bash
bash scripts/eval.sh --cell scaling__nca_1_1e18 --seed 0   # alexkstern/rppt-scaling-nca-1pct-1e18-seed0
bash scripts/eval.sh --checkpoint <dir>                    # a model you trained
```

It reports bits-per-byte on held-out FineWeb, C4, GitHub code and the model's
PPT corpus, BLiMP (67 minimal-pair grammar paradigms) and DCLM CORE (22 tasks,
as centered accuracy), and writes them as JSON beside a local checkpoint or
under `outputs/eval/`. `--suite bpb`, `blimp` or `core` runs one of them.

`--cell` scores with the settings the paper's FineWeb numbers were scored
with (`fineweb_bpb_scored_with` in `cells.json`), and on B200 GPUs reproduces
them exactly. To score every published model on FineWeb:

```bash
uv run python -m reproduce.rescore_fineweb
```

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
