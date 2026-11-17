"""One full experiment: preprocess -> partition -> train all models -> evaluate."""
from sklearn.model_selection import train_test_split
from preprocessing import load_heart, StandardScalerNP
from partition import partition_data
from models import LogisticRegressionFL, NeuralNetFL
from federated_server import federated_train
from federated_rf import federated_random_forest, centralized_models
from evaluate import evaluate


def run_experiment(seed=42, rounds=15, local_epochs=5, epsilon=5.0, iid=True):
    X, y, df, features = load_heart()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    scaler = StandardScalerNP().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
    hospitals = partition_data(Xtr, ytr, 3, iid=iid, seed=seed)
    d = Xtr.shape[1]
    mk_lr = lambda: LogisticRegressionFL(d, seed=1)
    mk_nn = lambda: NeuralNetFL(d, seed=1)
    kw = dict(rounds=rounds, local_epochs=local_epochs, seed=seed)

    models = {
        "Federated LR (FedAvg)": federated_train(hospitals, mk_lr, lr=0.3, **kw),
        "Federated LR + DP + Secure Agg": federated_train(hospitals, mk_lr, lr=0.3, secure=True,
                                                            epsilon=epsilon, **kw),
        "Federated Neural Network (FedAvg)": federated_train(hospitals, mk_nn, lr=0.05, **kw),
        "Federated NN + Secure Agg": federated_train(hospitals, mk_nn, lr=0.05, secure=True, **kw),
        "Federated Random Forest": federated_random_forest(hospitals, seed=seed),
    }
    c_lr, c_rf = centralized_models(Xtr, ytr, seed)
    c_nn = mk_nn(); c_nn.train_local(Xtr, ytr, epochs=rounds * local_epochs, lr=0.05, seed=seed)
    models.update({"Centralized LR (baseline)": c_lr, "Centralized NN (baseline)": c_nn,
                   "Centralized RF (baseline)": c_rf})

    results = [evaluate(yte, m.predict(Xte), name) for name, m in models.items()]
    return {"results": results, "models": models, "scaler": scaler,
            "features": features, "df": df, "hospitals": hospitals}
