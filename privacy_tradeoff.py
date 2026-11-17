"""Privacy-utility trade-off: how Federated LR accuracy changes with DP epsilon.
Usage: python privacy_tradeoff.py"""
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing import load_heart, StandardScalerNP
from partition import partition_data
from models import LogisticRegressionFL
from federated_server import federated_train
from evaluate import evaluate

EPSILONS = [0.1, 0.5, 1, 2, 5, 10, None]   # None = no differential privacy


def tradeoff(seeds=range(5)):
    X, y, _, _ = load_heart()
    out = {}
    for eps in EPSILONS:
        accs = []
        for s in seeds:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=s, stratify=y)
            sc = StandardScalerNP().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            hosp = partition_data(Xtr, ytr, 3, seed=s)
            m = federated_train(hosp, lambda: LogisticRegressionFL(Xtr.shape[1], seed=1),
                                lr=0.3, secure=True, epsilon=eps, seed=s)
            accs.append(evaluate(yte, m.predict(Xte), "x")["accuracy"])
        out[eps] = (float(np.mean(accs)), float(np.std(accs)))
    return out


if __name__ == "__main__":
    print(f"{'epsilon':<12}{'accuracy (mean +/- std over 5 splits)'}")
    for eps, (m, s) in tradeoff().items():
        print(f"{'none' if eps is None else eps:<12}{m:.3f} +/- {s:.3f}")
