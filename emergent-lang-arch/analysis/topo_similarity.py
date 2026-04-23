"""
Topographic Similarity (topo_sim / rho_topo).

Measures the correlation between pairwise distances in meaning space
and pairwise distances in message space — a proxy for compositionality.

Brighton & Kirby (2006); Lazaridou et al. (2018).
"""

import torch
import numpy as np
from itertools import combinations
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine, hamming
from Levenshtein import distance as levenshtein_distance


def meaning_distance(v1: np.ndarray, v2: np.ndarray, metric: str = "cosine") -> float:
    if metric == "cosine":
        return cosine(v1, v2)
    elif metric == "euclidean":
        return float(np.linalg.norm(v1 - v2))
    raise ValueError(f"Unknown metric: {metric}")


def message_distance(m1: np.ndarray, m2: np.ndarray, metric: str = "edit") -> float:
    if metric == "edit":
        return levenshtein_distance(tuple(m1.tolist()), tuple(m2.tolist()))
    elif metric == "hamming":
        min_len = min(len(m1), len(m2))
        return hamming(m1[:min_len], m2[:min_len])
    raise ValueError(f"Unknown metric: {metric}")


def compute_topo_similarity(
    meanings: np.ndarray,
    messages: np.ndarray,
    meaning_metric: str = "cosine",
    message_metric: str = "edit",
    max_pairs: int = 5000,
) -> dict:
    """
    Args:
        meanings: (N, feature_dim) float array
        messages: (N, max_len) int array of token ids (0 = pad/eos)
        max_pairs: subsample pairs for speed when N is large

    Returns:
        dict with 'rho' (Spearman r) and 'p' (p-value)
    """
    n = len(meanings)
    all_pairs = list(combinations(range(n), 2))

    rng = np.random.default_rng(0)
    if len(all_pairs) > max_pairs:
        idx = rng.choice(len(all_pairs), max_pairs, replace=False)
        pairs = [all_pairs[i] for i in idx]
    else:
        pairs = all_pairs

    meaning_dists, message_dists = [], []
    for i, j in pairs:
        meaning_dists.append(meaning_distance(meanings[i], meanings[j], meaning_metric))
        message_dists.append(message_distance(messages[i], messages[j], message_metric))

    rho, p = spearmanr(meaning_dists, message_dists)
    return {"rho": float(rho), "p": float(p), "n_pairs": len(pairs)}


@torch.no_grad()
def collect_messages(sender, dataloader, device):
    """Run sender over dataloader and return (meanings, messages) arrays."""
    meanings_list, messages_list = [], []
    sender.eval()
    for sender_input, _labels, _receiver_input in dataloader:
        sender_input = sender_input.to(device)
        message, _log_prob, _entropy = sender(sender_input)
        meanings_list.append(sender_input.cpu().numpy())
        messages_list.append(message.cpu().numpy())
    return np.concatenate(meanings_list), np.concatenate(messages_list)
