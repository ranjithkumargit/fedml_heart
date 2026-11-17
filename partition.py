"""Dataset Partitioning Module: simulate Hospital A / B / C with private local shards."""
import numpy as np


def partition_data(X, y, num_hospitals=3, iid=True, seed=42):
    """iid=True: random equal split. iid=False: label-skewed split, e.g. Hospital A mostly
    sees disease cases and Hospital C mostly healthy ones (every hospital still has both)."""
    rng = np.random.default_rng(seed)
    if iid:
        idx = rng.permutation(len(X))
        shards = np.array_split(idx, num_hospitals)
    else:
        p_pos = np.array([0.6, 0.3, 0.1][:num_hospitals]); p_pos = p_pos / p_pos.sum()
        p_neg = p_pos[::-1]
        owner = np.array([rng.choice(num_hospitals, p=p_pos if t == 1 else p_neg) for t in y])
        shards = [np.where(owner == h)[0] for h in range(num_hospitals)]
    return {f"Hospital {chr(65 + i)}": (X[s], y[s]) for i, s in enumerate(shards)}
