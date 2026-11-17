"""
Federated Aggregation Module (FedAvg), works for any model with get_flat/set_flat.

Each round:
 1. Server sends global model to every hospital          (Global Model Distribution)
 2. Hospital trains locally on its private data only      (Local Training)
 3. Hospital sends back only its weight UPDATE (delta)    (Model Update)
 4. Server sums the size-weighted updates -> new global   (Federated Aggregation)

secure=True adds Secure Aggregation masking; epsilon adds Differential Privacy noise.
Hospitals scale their update by n_i/N BEFORE masking, so the server just sums and
the masks cancel exactly (the weighted average is preserved).
"""
import numpy as np
from security import clip_update, add_dp_noise, make_pairwise_masks, apply_mask


def federated_train(hospitals, make_model, rounds=15, local_epochs=5, lr=0.1,
                    batch_size=16, secure=False, epsilon=None, clip_norm=1.0, seed=0):
    rng = np.random.default_rng(seed)
    global_model = make_model()
    g = global_model.get_flat()
    names = list(hospitals)
    sizes = {n: len(hospitals[n][1]) for n in names}
    total = sum(sizes.values())

    for r in range(rounds):
        masks = make_pairwise_masks(names, len(g), rng) if secure else None
        agg = np.zeros_like(g)
        for n in names:
            X, y = hospitals[n]
            local = make_model()
            local.set_flat(g)
            local.train_local(X, y, epochs=local_epochs, lr=lr,
                              batch_size=batch_size, seed=seed * 1000 + r)
            delta = local.get_flat() - g
            if epsilon is not None:
                delta = add_dp_noise(clip_update(delta, clip_norm), epsilon, clip_norm, rng)
            delta = delta * sizes[n] / total
            if secure:
                delta = apply_mask(n, masks, delta)
            agg += delta
        g = g + agg

    global_model.set_flat(g)
    return global_model
