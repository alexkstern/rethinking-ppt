"""System prompts and context builders for the llmtuna-based optimizer.

``balanced_system()`` builds the proposer's system prompt for the knobs a study
tunes. Architecture context is added via ``tuner.context.add_summary()``.
"""

KEY_KNOBS = ("depth", "matrix_lr", "ppt_lr")
_COUNT_WORD = {1: "one", 2: "two", 3: "three"}


def _join(names: list[str]) -> str:
    marked = [f"`{n}`" for n in names]
    if not marked:
        return ""
    if len(marked) == 1:
        return marked[0]
    if len(marked) == 2:
        return f"{marked[0]} and {marked[1]}"
    return ", ".join(marked[:-1]) + f", and {marked[-1]}"


def balanced_system(*, hparams) -> str:
    """The default proposer prompt, naming only the ``KEY_KNOBS`` this study tunes."""
    hparams = list(hparams)
    named = [k for k in KEY_KNOBS if k in hparams]
    joined = _join(named)
    one = len(named) == 1
    has_rest = len(hparams) > len(named)

    key_para = ""
    if named:
        key_para = (
            f"Most of the result at a given scale comes from "
            f"{_COUNT_WORD[len(named)]} knob{'' if one else 's'}: {joined}. "
            + ("It has its own optimum, which shifts with scale. Getting it near "
               "its minimum for THIS scale"
               if one else
               "Each has its own optimum, and each optimum shifts with scale. "
               "Getting them near their minimum for THIS scale")
            + " matters more than refining the other knobs, so make sure "
            + ("it has" if one else "each has")
            + " been searched widely enough that you would notice if it were off."
        )
        if has_rest:
            key_para += (" The remaining knobs are worth tuning once "
                         + ("it looks" if one else "those look") + " settled.")
        key_para += "\n\n"

    steps = "Search locally around the current best with moderate steps: "
    if "depth" in hparams:
        steps += "move `depth` by 1-3 layers, and change learning rates / ratios "
    else:
        steps += "change learning rates / ratios "
    steps += ("by roughly a factor of 1.5-3. A larger jump is worth it when several "
              "trials show a consistent trend that has not yet reversed; otherwise "
              "it tends to spend the budget without resolving anything.")

    bracket = ""
    if named:
        bracket = (
            f"- {joined} "
            + ("has an interior optimum, so bracket it rather than revisiting it "
               "repeatedly."
               if one else
               "each have an interior optimum, so bracket each one rather than "
               "revisiting it repeatedly. Re-open one only if a change to another "
               "suggests its optimum has moved.")
            + "\n"
        )

    return (
        "You are a hyperparameter optimizer minimizing val_bpb (validation "
        "bits-per-byte; lower is better).\n\n"
        "You see all prior trials in the order they were run. Balance exploiting the "
        "current best config with informative exploration \u2014 the trial budget is "
        "small, so every proposal should either improve the best or teach you "
        "something specific.\n\n"
        f"{key_para}"
        "Guidelines:\n"
        "- Prefer changing one or two parameters per proposal so results stay "
        "attributable. Change more at once only when prior trials clearly support "
        "the combination.\n"
        f"- {steps}\n"
        f"{bracket}"
        "- Try not to set several parameters to the extremes of their ranges at "
        "once, and avoid values that would obviously break training (e.g. a "
        "learning rate at or near zero).\n"
        "- If an early conclusion about one knob may have been confounded by "
        "another knob's value at the time, re-test it at the current best.\n\n"
        "You MUST call propose_config with values for ALL parameters."
    )


# The system prompt used for architecture summarization.
SUMMARIZE_SYSTEM = """\
You are a hyper parameter optimizer. You will be given source code files from \
a training codebase, along with the specific hyperparameters being optimized \
and the training mode.

Produce a concise summary (300-500 words) covering ONLY what is relevant to \
the listed hyperparameters:

1. **Model architecture**: Key structural choices. How depth/width relate to \
parameter count and compute.
2. **Training setup**: Optimizer, schedules. Note that training runs for a fixed \
FLOP budget, so larger models get fewer steps.
3. **Tuned hyperparameters**: For EACH listed hparam, explain what it controls. \
Mention default values for any related params that are held fixed.
4. **Loss landscape intuition**: Which tuned hparams likely interact? \
Which have log-scale vs linear sensitivity?
5. **Scaling behavior**: The optimizer will run at multiple FLOP scales. \
For each tuned hparam, reason about how its optimal value should change \
as the FLOP budget increases. More FLOPs means you can train larger \
models on more data, so optimal hparams can shift substantially between \
scales — do not assume the best config at one scale transfers to the next.

Do NOT discuss hyperparameters, features, or training modes that are not being \
optimized. Stay focused on what the optimizer needs to know.\
"""

# What the proposer is shown about the code it is tuning: only what drives the
# tuned hparams and the FLOP-budgeted loop.
CONTEXT_SOURCE_FILES: list[str] = [
    "nanochat/phase.py",                         # FLOP split, build_phase, eval, per-phase LR
    "nanochat/loop.py",                                # training loop; objective metric
    "nanochat/models/nanochat_gpt/vanilla_3a5e0bc.py",  # depth -> width; Muon + AdamW param groups
    "nanochat/optim/factory.py",                        # matrix_lr / embedding_lr routing
    "nanochat/lr_schedules.py",                         # warmup_ratio, warmdown_ratio shapes
    "nanochat/flops.py",                                # target_flops -> iteration count
    "nanochat/data/registry.py",                        # PT/PPT corpus keys + eval metrics
    "experiments/tuner/space.py",                       # tuned hparam definitions + bounds
    "experiments/training/config.py",                   # shared study config
    "experiments/training/two_phase.py",                      # the training entry every study runs
]

