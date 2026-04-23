"""
Additional language-emergence metrics beyond topographic similarity.
"""

import numpy as np
from collections import Counter
from scipy.stats import entropy as scipy_entropy


# ---------------------------------------------------------------------------
# Bosdiscriminability / entropy
# ---------------------------------------------------------------------------

def message_entropy(messages: np.ndarray) -> float:
    """Shannon entropy over the full message distribution (treated as strings)."""
    strings = [tuple(m) for m in messages]
    counts = Counter(strings)
    probs = np.array(list(counts.values()), dtype=float)
    probs /= probs.sum()
    return float(scipy_entropy(probs, base=2))


def positional_entropy(messages: np.ndarray, vocab_size: int) -> np.ndarray:
    """Per-position symbol entropy. Shape: (max_len,)."""
    max_len = messages.shape[1]
    entropies = np.zeros(max_len)
    for pos in range(max_len):
        counts = np.bincount(messages[:, pos], minlength=vocab_size).astype(float)
        counts /= counts.sum()
        entropies[pos] = float(scipy_entropy(counts, base=2))
    return entropies


# ---------------------------------------------------------------------------
# Language discreteness proxy
# ---------------------------------------------------------------------------

def symbol_unigram_freq(messages: np.ndarray, vocab_size: int) -> np.ndarray:
    flat = messages.flatten()
    counts = np.bincount(flat, minlength=vocab_size).astype(float)
    return counts / counts.sum()


def effective_vocab_size(messages: np.ndarray, vocab_size: int, threshold: float = 1e-3) -> int:
    freq = symbol_unigram_freq(messages, vocab_size)
    return int((freq > threshold).sum())


# ---------------------------------------------------------------------------
# Message length statistics
# ---------------------------------------------------------------------------

def message_length_stats(messages: np.ndarray, eos_token: int = 0) -> dict:
    lengths = []
    for msg in messages:
        eos_positions = np.where(msg == eos_token)[0]
        length = int(eos_positions[0]) if len(eos_positions) else len(msg)
        lengths.append(length)
    lengths = np.array(lengths)
    return {
        "mean_len": float(lengths.mean()),
        "std_len": float(lengths.std()),
        "min_len": int(lengths.min()),
        "max_len": int(lengths.max()),
    }


# ---------------------------------------------------------------------------
# Convenience: compute all metrics at once
# ---------------------------------------------------------------------------

def compute_all_metrics(messages: np.ndarray, vocab_size: int, eos_token: int = 0) -> dict:
    stats = message_length_stats(messages, eos_token)
    stats["message_entropy"] = message_entropy(messages)
    stats["effective_vocab_size"] = effective_vocab_size(messages, vocab_size)
    stats["positional_entropy"] = positional_entropy(messages, vocab_size).tolist()
    return stats
