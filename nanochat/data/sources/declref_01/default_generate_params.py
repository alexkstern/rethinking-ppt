"""Default decl-ref-01 generation parameters.

A scoped declare/reference skeleton plus four knobs — noisy references, a
part-transition chain, a stickiness ramp, and topic shifts — that tune the
per-position entropy dynamics of a model trained on the data. The values below
are the natural-language-matched preset: they put the entropy-vs-position
profile inside the natural envelope (code / C4 / FineWeb), matched on positional
dynamics, fluctuation texture, and the entropy-quantile distribution, and hold
that match as training doubles.
"""

# --- skeleton ---
N_PARTS = 512          # distinct name-parts and value-parts (each its own id block)
MAX_DEPTH = 8          # maximum scope nesting depth
P_BRACKET = 0.04       # per-step probability of OPEN (and, symmetrically, CLOSE)
LEN_PROBS = (0.5, 0.3, 0.2)  # symbol length distribution over 1..3 parts
ALPHA_RANGE = (0.8, 1.4)     # per-document Zipf exponent range for symbol parts

# --- reference dynamics ---
# NOISE_P: probability a REF re-binds a *fresh* value instead of copying the
#   existing one. Raises the floor of "easy" tokens off zero — natural text has
#   almost no truly deterministic positions.
NOISE_P = 0.22
# REF_RECENCY_ALPHA: Zipf exponent over recency ranks for which visible name a
#   REF points at; larger = more strongly biased to the most recent declaration.
REF_RECENCY_ALPHA = 1.38
# REF_SHARE_LO/HI: the REF fraction of statements ramps linearly from LO at the
#   document start to HI at the end (early declaring, later referencing — a
#   downward entropy tail, like code reusing what it defined).
REF_SHARE_LO = 0.28
REF_SHARE_HI = 0.82

# --- keyword-run stickiness ---
# STAY_LO/HI: probability the next DECL/REF repeats the previous action, ramped
#   from LO over the document's first third to HI. Low early = interleaved
#   openings (no all-cheap opening reference burst).
STAY_LO = 0.55
STAY_HI = 0.94

# --- part-transition chain (makes content partially predictable, like words) ---
# CHAIN_MIX: probability a symbol part is drawn from its chain successors rather
#   than the per-document Zipf. Compresses the entropy dynamic range toward
#   text's (~4:1 vs an all-independent ~40:1).
CHAIN_MIX = 0.75
CHAIN_BRANCH = 8       # plausible chain successors per part
CHAIN_SUCC_POOL = 16   # successor pool per part the CHAIN_BRANCH are chosen from
CHAIN_SEED = 123       # fixes the (part -> successors) table; part of the corpus identity

# --- topic shifts (keep mid-document difficulty up; make the language inexhaustible) ---
# Every SHIFT_PERIOD tokens (first at FIRST_SHIFT), re-permute SHIFT_FRAC of the
# name distribution. A bigger model keeps facing fresh structure rather than
# mastering a finite bag of tricks, which is what keeps the profile match
# stationary across training budget.
SHIFT_PERIOD = 110
FIRST_SHIFT = 48
SHIFT_FRAC = 0.30

SEQ_LENGTH = 2048      # tokens per generated document
# Token ids: 0=OPEN 1=CLOSE 2=DECL 3=REF, name-parts [4, 4 + N_PARTS), value-parts
# [4 + N_PARTS, 4 + 2 * N_PARTS), so a pretokenized bin has
# vocab_size = 4 + 2 * N_PARTS = 1028.
