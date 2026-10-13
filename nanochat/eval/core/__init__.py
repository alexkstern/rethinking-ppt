"""CORE benchmark evaluation (DCLM, https://arxiv.org/abs/2406.11794).

The harness consumes a built `(model, tokenizer, device)` triple and never
branches on model internals; which architecture to build and which tokenizer it
trained with live only in `checkpoint_loading`, imported lazily by its
consumers.
"""

from nanochat.eval.core.tasks import run_core_eval

__all__ = ["run_core_eval"]
