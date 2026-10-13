"""Load a nanochat training checkpoint for eval, from disk or from the Hub.

A training phase writes a per-step pair into its checkpoint directory, and
`nanochat.hf_upload.push_checkpoint` uploads that same pair to an HF model repo:

    model_{step:06d}.pt     — model weights (state_dict)
    config_{step:06d}.json  — the resolved run-config snapshot

`load_from_dir` reads that pair from a local directory, `load_from_hf`
downloads it first. Both rebuild the architecture from the snapshot's
`model_pt` section, load the weights, and build the tokenizer the run trained
on (derived from its `data_pt` registry key), returning arch family, tokenizer
key and lineage alongside the model.

Architecture dispatch is on the `model_pt` field signature, each supported
family having a distinct field set; anything unrecognized raises.

Models are at most ~600M parameters, so loading is single-device.
"""

from __future__ import annotations

import functools
import os
import re
from dataclasses import dataclass

import torch
from huggingface_hub import HfApi, hf_hub_download

from nanochat.checkpoint import load_config_snapshot
from nanochat.eval.core.tasks import EvalModel
from nanochat.hf_upload import retry_on_timeout


def _latest(names, where: str) -> int:
    steps = [int(m.group(1)) for f in names if (m := re.fullmatch(r"model_(\d+)\.pt", f))]
    if not steps:
        raise FileNotFoundError(f"no model_*.pt in {where}")
    return max(steps)


def latest_step(repo: str) -> int:
    """Highest step number among the repo's `model_{step:06d}.pt` files."""
    files = retry_on_timeout(HfApi().list_repo_files, repo, repo_type="model")
    return _latest(files, repo)


def latest_step_in_dir(checkpoint_dir: str) -> int:
    """Highest step number among a local directory's `model_{step:06d}.pt` files."""
    return _latest(os.listdir(checkpoint_dir), checkpoint_dir)


@dataclass
class LoadedCheckpoint:
    """A checkpoint rebuilt for eval, plus the lineage read off its config
    snapshot."""

    model: EvalModel
    tokenizer: object                  # RustBPETokenizer (callable, get_bos_token_id)
    config: dict                       # full run-config snapshot (config_{step}.json)
    arch_family: str
    tokenizer_key: str
    step: int
    hf_repo: str


def _strip_compile_prefix(sd: dict) -> dict:
    # `checkpoint.save_checkpoint` unwraps compiled models before saving; this
    # covers externally-produced state dicts that kept the prefix.
    return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}


def resolve_arch(model_pt: dict):
    """`(arch_family, ModelSpec, config class, ignorable state-dict key
    substrings)` for a config snapshot's `model_pt` field signature.

    Each branch imports its architecture module only when selected. An empty
    ignorable set makes the weight load fully strict.
    """
    keys = set(model_pt)
    if "n_kv_head" in keys:
        from nanochat.models.nanochat_gpt import vanilla_3a5e0bc

        return ("nanochat_gpt", vanilla_3a5e0bc.MODEL_SPEC,
                vanilla_3a5e0bc.NanochatGPTConfig, frozenset())
    raise ValueError(f"unrecognized model_pt field signature: {sorted(keys)}")


def _check_state_dict_keys(label: str, missing, unexpected, ignorable: frozenset):
    """Raise on any missing/unexpected key not covered by an ignorable substring."""
    bad = [k for k in (*missing, *unexpected) if not any(ig in k for ig in ignorable)]
    if bad:
        raise ValueError(f"state-dict key mismatch loading {label}: {bad}")


def _build(spec, config_cls, ignorable: frozenset, model_pt: dict,
           device: torch.device, sd: dict) -> torch.nn.Module:
    m = spec.build_model(config_cls(**model_pt))
    m.to_empty(device=device)
    m.init_weights()
    missing, unexpected = m.load_state_dict(sd, strict=False)
    _check_state_dict_keys(config_cls.__name__, missing, unexpected, ignorable)
    return m


@functools.lru_cache(maxsize=None)
def build_tokenizer(data_pt_key: str):
    """(tokenizer, tokenizer_key) for the text corpus a run pretrained on,
    dispatched on the registry entry's `token_bytes` kind: "rustbpe" -> the
    `tokenizer/tokenizer.pkl` the corpus repo bundles, key "nanochatbpe".

    Synthetic corpora carry no text tokenizer, so their checkpoints cannot be
    CORE-evaluated and raise here. Cached per key.
    """
    from nanochat.data.registry import build_tokenizer as build_kind_tokenizer
    from nanochat.data.registry import get_source

    source = get_source(data_pt_key)
    if source.token_bytes == "rustbpe":
        return build_kind_tokenizer("rustbpe", repo_id=source.repo_id), "nanochatbpe"
    raise ValueError(
        f"data source {data_pt_key!r} has token_bytes={source.token_bytes!r}: no text "
        f"tokenizer exists for it, so its checkpoints cannot be CORE-evaluated"
    )


def load_from_dir(checkpoint_dir: str, device: torch.device, *,
                  step: int | None = None,
                  source: str | None = None) -> LoadedCheckpoint:
    """Rebuild a checkpoint from a local directory on `device`, in eval mode.

    `step` defaults to the highest present, the end of the phase that wrote the
    directory. `source` labels where it came from, for the record.
    """
    if step is None:
        step = latest_step_in_dir(checkpoint_dir)
    where = source or checkpoint_dir
    config = load_config_snapshot(checkpoint_dir, step)
    model_pt = config["model_pt"]

    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    sd = _strip_compile_prefix(
        torch.load(model_path, map_location="cpu", weights_only=True, mmap=True))

    arch_family, spec, config_cls, ignorable = resolve_arch(model_pt)
    trained_vocab = sd[spec.embed_key].shape[0]
    if trained_vocab != model_pt["vocab_size"]:
        raise ValueError(
            f"{where}: config vocab_size {model_pt['vocab_size']} != embedding rows "
            f"{trained_vocab} in the saved weights"
        )
    model = _build(spec, config_cls, ignorable, model_pt, device, sd).eval()

    tokenizer, tokenizer_key = build_tokenizer(config["data_pt"])
    return LoadedCheckpoint(
        model=EvalModel(model, max_seq_len=model_pt["sequence_len"]),
        tokenizer=tokenizer,
        config=config,
        arch_family=arch_family,
        tokenizer_key=tokenizer_key,
        step=step,
        hf_repo=where,
    )


def load_from_hf(hf_repo: str, device: torch.device,
                 step: int | None = None) -> LoadedCheckpoint:
    """Rebuild one checkpoint step of `hf_repo` on `device`, in eval mode.
    `step` defaults to the highest present."""
    step = latest_step(hf_repo) if step is None else step
    # Both files land in one cache directory, read as a local checkpoint dir.
    config_path = retry_on_timeout(
        hf_hub_download, hf_repo, f"config_{step:06d}.json", repo_type="model")
    retry_on_timeout(
        hf_hub_download, hf_repo, f"model_{step:06d}.pt", repo_type="model")
    return load_from_dir(os.path.dirname(config_path), device,
                         step=step, source=hf_repo)
