"""nanochat-style GPT, architecturally as karpathy/nanochat at 3a5e0bc.

Notable features:
- rotary embeddings (no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA)
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.config import OptimConfig
from nanochat.models.base import BaseModel, ModelSpec
from nanochat.optim.base import ParamGroupSpec


ARCH_PROVENANCE = {
    "upstream_repo": "karpathy/nanochat",
    "upstream_commit": "3a5e0bc",
    "upstream_date": "2025-10-13",
}

@dataclass
class NanochatGPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last time into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3) # re-assemble
    out = out.to(x.dtype) # ensure input/output dtypes match
    return out

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Rotary embeddings on q and k give relative positional encoding.
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # QK rotary embedding
        q, k = norm(q), norm(k) # QK norm
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)

        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2) # queries in this forward pass
        Tk = k.size(2) # keys/values in total: cache + this forward pass

        enable_gqa = self.n_head != self.n_kv_head # duplicate kv heads to match query heads
        if kv_cache is None or Tq == Tk:
            # Training, or a cache whose contents are exactly this pass: plain causal.
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        elif Tq == 1:
            # Inference, single query: it attends to everything in the cache.
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            # Inference over a chunk of queries: each attends to the whole cached
            # prefix, then causally within the chunk.
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device) # True = keep
            prefix_len = Tk - Tq
            if prefix_len > 0:
                attn_mask[:, :prefix_len] = True
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

        # Re-assemble the heads side by side and project back to the residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(BaseModel):
    def __init__(self, config: NanochatGPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Initialized here only to support meta-device construction; the real
        # values are computed in init_weights. The cache is cheap, so it is
        # over-computed 10x and asserted against rather than grown dynamically.
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        self.apply(self._init_weights)
        # Zero the classifier and every block's output projection.
        torch.nn.init.zeros_(self.lm_head.weight)
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # bf16 embeddings: the optimizer tolerates it and it saves memory in both
        # the model and the activations.
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # Rotation frequency at each (time, channel) pair.
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # batch + head dims, for broadcasting
        return cos, sin

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # Rotary embeddings are (1, seq_len, 1, head_dim).
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # With a kv cache, the rotary embeddings offset to the cache position.
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        # Precision is owned here, inside the model: the train loop and FLOPs
        # measurement call forward with no autocast of their own. On CUDA the
        # token embedding is bfloat16 (see init_weights), so the compute runs
        # under bf16 autocast and the fp32 linear weights match the bf16
        # activations; on CPU/MPS everything stays fp32.
        compute_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if idx.device.type == "cuda"
            else contextlib.nullcontext()
        )
        with compute_ctx:
            x = self.transformer.wte(idx)
            x = norm(x)
            for block in self.transformer.h:
                x = block(x, cos_sin, kv_cache)
            x = norm(x)

            softcap = 15
            if targets is not None:
                logits = self.lm_head(x)
                logits = softcap * torch.tanh(logits / softcap)
                logits = logits.float() # tf32/fp32 for the logits
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
                return loss
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap)
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """Naive autoregressive streaming inference, assuming batch size 1 and
        plain Python lists/ints for the ids and the yielded tokens."""
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token


def build_model(cfg: NanochatGPTConfig) -> GPT:
    """Build a GPT on the meta device. The caller must `to_empty(device=...)` and
    then `init_weights()` before training."""
    with torch.device("meta"):
        return GPT(cfg)


# AdamW settings for the vocab-facing groups. Module-level so that variants
# which must stay identical to these groups import it rather than re-hardcode
# the literals.
ADAMW_KWARGS = {"betas": (0.8, 0.95), "eps": 1e-10}


def build_param_groups(model: GPT, cfg: OptimConfig) -> list[ParamGroupSpec]:
    """Group nanochat-GPT parameters into matrix (Muon) / embedding / unembedding
    (AdamW).

    The architecture-aware half of optimizer construction, so all arch-specific
    LR scaling lives here rather than in `build_optimizers`: the matrix LR passes
    through unscaled per the Muon convention, while the embedding and unembedding
    LRs are scaled by `(n_embd / 768) ** -0.5`, since they were tuned at
    n_embd=768.
    """
    embedding_lr_scale = (model.config.n_embd / 768) ** -0.5
    return [
        ParamGroupSpec(
            name="matrix",
            params=list(model.transformer.h.parameters()),
            optim_name="muon",
            lr=cfg.matrix_lr,
            weight_decay=cfg.weight_decay,
            optim_kwargs={"momentum": 0.95},
        ),
        ParamGroupSpec(
            name="embedding",
            params=list(model.transformer.wte.parameters()),
            optim_name="adamw",
            lr=cfg.embedding_lr * embedding_lr_scale,
            weight_decay=cfg.weight_decay,
            optim_kwargs=dict(ADAMW_KWARGS),
        ),
        ParamGroupSpec(
            name="unembedding",
            params=list(model.lm_head.parameters()),
            optim_name="adamw",
            lr=cfg.unembedding_lr * embedding_lr_scale,
            weight_decay=cfg.weight_decay,
            optim_kwargs=dict(ADAMW_KWARGS),
        ),
    ]


# wte + lm_head change shape under a vocab transition; the untied body does not.
MODEL_SPEC = ModelSpec(
    build_model=build_model,
    build_param_groups=build_param_groups,
    embed_key="transformer.wte.weight",
    surgery_keys=frozenset({"transformer.wte.weight", "lm_head.weight"}),
)
