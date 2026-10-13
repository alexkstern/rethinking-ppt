"""BLiMP (Warstadt et al., 2020): 67 paradigms of 1,000 minimal pairs, each one
grammatical and one ungrammatical sentence, the model correct when it gives the
grammatical one the higher likelihood.

Scored as `eval_checkpoint.py:blimp_eval` of Hu et al. (2025) scores it:

  - a sentence's score is its TOTAL log-probability, not a per-token mean;
  - by default no BOS is prepended, so the first token is unscored;
  - good > bad STRICTLY, so exact ties count as wrong;
  - the headline number is the unweighted mean over all pairs.

`prepend_bos` is an addition: their models trained without a document-start
token and ours train with one. Report the `prepend_bos=False` number when
comparing against published BLiMP results.

The model contract is `nanochat.eval.core`'s: a callable
`model(input_ids) -> logits` of shape (B, T, vocab).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# The mirror Hu et al. load: all 67 paradigms flattened into one `train` split.
BLIMP_DATASET = "WillHeld/blimp"
# BLiMP sentences are short; this is an assertion boundary, not a truncation
# budget. A sentence past it means the tokenizer is not the expected one.
MAX_TOKENS = 128
# `UID` is the paradigm (67), `linguistics_term` the phenomenon (12), `field`
# the coarse area (4).
GROUP_FIELDS = ("UID", "linguistics_term", "field")


@dataclass(frozen=True)
class EncodedPair:
    """One minimal pair, tokenized, with the grouping labels carried along."""
    good: list[int]
    bad: list[int]
    uid: str
    linguistics_term: str
    field: str


def sequence_logprobs(logits: torch.Tensor, ids: torch.Tensor,
                      lengths: torch.Tensor) -> torch.Tensor:
    """Total log-probability of each row's first `length` tokens, in float32.

    Scores positions 1..length-1 — token 0 is input, never a prediction — the
    same span Hu et al.'s cross-entropy indexing covers. Padding past `length`
    is masked rather than trimmed, so a batch can mix lengths.
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must be (B, T, V), got {tuple(logits.shape)}")
    if ids.shape != logits.shape[:2]:
        raise ValueError(
            f"ids {tuple(ids.shape)} does not match logits {tuple(logits.shape[:2])}")
    if lengths.shape != ids.shape[:1]:
        raise ValueError(
            f"lengths {tuple(lengths.shape)} does not match batch {ids.shape[0]}")

    scores = logits[:, :-1].float()
    targets = ids[:, 1:]
    token_lp = (scores.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                - scores.logsumexp(dim=-1))
    positions = torch.arange(targets.shape[1], device=ids.device).unsqueeze(0)
    # position i of `targets` is token i+1, scored iff i + 1 < length
    scored = positions < (lengths.unsqueeze(1) - 1)
    return (token_lp * scored).sum(dim=1)


def encode_pairs(tokenizer, rows, *, prepend_bos: bool = False,
                 max_tokens: int = MAX_TOKENS) -> list[EncodedPair]:
    """Tokenize BLiMP rows — iterable of mappings with `sentence_good`,
    `sentence_bad` and the three grouping fields — failing on implausible
    lengths."""
    bos = tokenizer.get_bos_token_id() if prepend_bos else None
    out: list[EncodedPair] = []
    for row in rows:
        pair = []
        for key in ("sentence_good", "sentence_bad"):
            ids = tokenizer.encode(row[key])
            if bos is not None:
                ids = [bos] + list(ids)
            if len(ids) > max_tokens:
                raise ValueError(
                    f"{row.get('UID', '?')}: {key} is {len(ids)} tokens, over the "
                    f"{max_tokens}-token sanity bound — is this the right tokenizer?")
            if len(ids) < 2:
                raise ValueError(
                    f"{row.get('UID', '?')}: {key} has {len(ids)} tokens; nothing "
                    f"to score")
            pair.append(list(ids))
        out.append(EncodedPair(good=pair[0], bad=pair[1], uid=row["UID"],
                               linguistics_term=row["linguistics_term"],
                               field=row["field"]))
    return out


def _pad_batch(seqs: list[list[int]], device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(s) for s in seqs)
    ids = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    return ids.to(device), lengths.to(device)


@torch.no_grad()
def score_pairs(model, pairs: list[EncodedPair], device,
                batch_size: int = 64) -> list[bool]:
    """True where the model prefers the grammatical sentence, pair by pair.

    The comparison is strict: a tie is wrong.
    """
    correct: list[bool] = []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        ids, lengths = _pad_batch([p.good for p in chunk] + [p.bad for p in chunk],
                                  device)
        lp = sequence_logprobs(model(ids), ids, lengths)
        good_lp, bad_lp = lp[:len(chunk)], lp[len(chunk):]
        correct.extend((good_lp > bad_lp).tolist())
    return correct


def aggregate(pairs: list[EncodedPair], correct: list[bool],
              prefix: str = "blimp") -> dict[str, float]:
    """Overall, per-paradigm and per-phenomenon accuracy.

    The headline `{prefix}` is the mean over PAIRS, matching Hu et al. On the
    complete dataset that equals the mean over paradigms; on a subset it does
    not.
    """
    if len(pairs) != len(correct):
        raise ValueError(f"{len(pairs)} pairs but {len(correct)} verdicts")
    if not pairs:
        raise ValueError("no pairs to aggregate")

    metrics = {prefix: sum(correct) / len(correct), f"{prefix}_n": float(len(correct))}
    for field in GROUP_FIELDS:
        tally: dict[str, list[int]] = {}
        for pair, ok in zip(pairs, correct):
            tally.setdefault(getattr(pair, _attr(field)), []).append(bool(ok))
        for key, oks in sorted(tally.items()):
            metrics[f"{prefix}_{_slug(field)}/{key}"] = sum(oks) / len(oks)
    return metrics


def _attr(field: str) -> str:
    return {"UID": "uid", "linguistics_term": "linguistics_term",
            "field": "field"}[field]


def _slug(field: str) -> str:
    return {"UID": "uid", "linguistics_term": "term", "field": "area"}[field]


def load_blimp(dataset_name: str = BLIMP_DATASET, max_per_uid: int = -1):
    """The BLiMP rows, optionally capped per paradigm for a smoke pass."""
    import datasets

    rows = list(datasets.load_dataset(dataset_name)["train"])
    if max_per_uid and max_per_uid > 0:
        seen: dict[str, int] = {}
        kept = []
        for row in rows:
            n = seen.get(row["UID"], 0)
            if n < max_per_uid:
                kept.append(row)
                seen[row["UID"]] = n + 1
        rows = kept
    return rows


def run_blimp_eval(model, tokenizer, device, *, prepend_bos: bool = False,
                   batch_size: int = 64, max_per_uid: int = -1) -> dict[str, float]:
    """Public entrypoint. Returns a flat W&B-summary metric dict::

        {"blimp": <mean over pairs>,
         "blimp_n": <pairs scored>,
         "blimp_uid/<paradigm>": ...,      # 67 of them
         "blimp_term/<phenomenon>": ...,   # 12
         "blimp_area/<field>": ...}        # 4

    With `prepend_bos=True` the keys carry a `blimp_bos` prefix instead, so both
    variants can be logged to one run without colliding.
    """
    prefix = "blimp_bos" if prepend_bos else "blimp"
    pairs = encode_pairs(tokenizer, load_blimp(max_per_uid=max_per_uid),
                         prepend_bos=prepend_bos)
    correct = score_pairs(model, pairs, device, batch_size=batch_size)
    return aggregate(pairs, correct, prefix=prefix)
