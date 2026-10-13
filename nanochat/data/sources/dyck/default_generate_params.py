"""Default k-shuffle Dyck generation parameters."""

K = 128              # bracket types; the vocabulary spans 2 * K ids
MAX_DEPTH = 16       # maximum bracket nesting depth
P_OPEN = 0.5         # probability of opening a new bracket (vs. closing one)
SEQ_LENGTH = 2048    # tokens per generated dyck sequence
# Token ids are 0-based: opening bracket type i is id i, closing is id i + K, so
# ids span [0, 2 * K) and a pretokenized dyck bin has vocab_size = 2 * K.
