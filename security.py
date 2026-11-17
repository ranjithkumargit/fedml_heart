"""
Privacy and Security Module.
1. Differential Privacy: clip each hospital's update to a max L2 norm, then add
   Gaussian noise (sigma = clip_norm / epsilon). Smaller epsilon = more noise.
2. Secure Aggregation: pairwise additive masks. Hospital i adds the mask, hospital j
   subtracts it, so masks cancel exactly when the server SUMS all updates. The server
   never sees any single hospital's update in the clear.
"""
import numpy as np


def clip_update(delta, clip_norm):
    n = np.linalg.norm(delta)
    return delta * (clip_norm / n) if n > clip_norm else delta


def add_dp_noise(delta, epsilon, clip_norm, rng):
    return delta + rng.normal(0, clip_norm / max(epsilon, 1e-6), delta.shape)


def make_pairwise_masks(names, dim, rng):
    names = list(names)
    return {(a, b): rng.normal(0, 1.0, dim)
            for i, a in enumerate(names) for b in names[i + 1:]}


def apply_mask(name, masks, vec):
    out = vec.copy()
    for (a, b), m in masks.items():
        if name == a:
            out += m
        elif name == b:
            out -= m
    return out
