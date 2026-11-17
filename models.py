"""
Local Training Module: models trained INSIDE each hospital.
Both models expose get_flat()/set_flat() so the Model Update / Aggregation
modules can treat their weights as one vector and average them (FedAvg).
"""
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class LogisticRegressionFL:
    def __init__(self, n_features, seed=0):
        self.w = np.random.default_rng(seed).normal(0, 0.01, n_features)
        self.b = 0.0

    def get_flat(self):
        return np.append(self.w, self.b)

    def set_flat(self, v):
        self.w, self.b = v[:-1].copy(), float(v[-1])

    def train_local(self, X, y, epochs=5, lr=0.1, batch_size=16, seed=0):
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            order = rng.permutation(len(X))
            for s in range(0, len(X), batch_size):
                i = order[s:s + batch_size]
                err = sigmoid(X[i] @ self.w + self.b) - y[i]
                self.w -= lr * X[i].T @ err / len(i)
                self.b -= lr * err.mean()

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


class NeuralNetFL:
    """Small neural network: n_features -> 16 -> 8 -> 1 (ReLU, ReLU, sigmoid)."""
    def __init__(self, n_features, hidden=(16, 8), seed=0):
        rng = np.random.default_rng(seed)
        sizes = [n_features, *hidden, 1]
        self.shapes = [(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.W = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in self.shapes]
        self.B = [np.zeros(b) for _, b in self.shapes]

    def get_flat(self):
        return np.concatenate([np.concatenate([w.ravel(), b]) for w, b in zip(self.W, self.B)])

    def set_flat(self, v):
        p = 0
        for k, (a, b) in enumerate(self.shapes):
            self.W[k] = v[p:p + a * b].reshape(a, b).copy(); p += a * b
            self.B[k] = v[p:p + b].copy(); p += b

    def _forward(self, X):
        acts = [X]
        for k in range(len(self.W) - 1):
            acts.append(np.maximum(0, acts[-1] @ self.W[k] + self.B[k]))
        return acts, sigmoid(acts[-1] @ self.W[-1] + self.B[-1]).ravel()

    def train_local(self, X, y, epochs=5, lr=0.05, batch_size=16, seed=0):
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            order = rng.permutation(len(X))
            for s in range(0, len(X), batch_size):
                i = order[s:s + batch_size]
                acts, p = self._forward(X[i])
                dz = ((p - y[i]) / len(i))[:, None]
                for k in range(len(self.W) - 1, -1, -1):
                    gW, gB = acts[k].T @ dz, dz.sum(axis=0)
                    if k > 0:
                        dz = (dz @ self.W[k].T) * (acts[k] > 0)
                    self.W[k] -= lr * gW
                    self.B[k] -= lr * gB

    def predict_proba(self, X):
        return self._forward(X)[1]

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
