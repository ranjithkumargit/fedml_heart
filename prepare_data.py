"""Creates the physically separate hospital data files used by the networked demo.
Each hospital file contains ONLY that hospital's patients; clients read only their own."""
import json, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import load_heart, StandardScalerNP
from partition import partition_data


def prepare(out="federated_run", seed=42, iid=True):
    X, y, df, features = load_heart()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    sc = StandardScalerNP().fit(Xtr)                    # shared scaling statistics
    hosp = partition_data(sc.transform(Xtr), ytr, 3, iid=iid, seed=seed)
    os.makedirs(f"{out}/hospitals", exist_ok=True)
    sizes = {}
    for name, (Xh, yh) in hosp.items():
        f = pd.DataFrame(Xh, columns=features); f["target"] = yh
        f.to_csv(f"{out}/hospitals/{name.replace(' ', '_').lower()}.csv", index=False)
        sizes[name] = int(len(yh))
    t = pd.DataFrame(sc.transform(Xte), columns=features); t["target"] = yte
    t.to_csv(f"{out}/test.csv", index=False)           # held out, used only by the server to score
    json.dump({"mean": sc.mean_.tolist(), "std": sc.std_.tolist(), "features": features},
              open(f"{out}/scaler.json", "w"))
    return sizes
