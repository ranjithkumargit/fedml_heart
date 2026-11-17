"""
Hospital Client (device layer). Runs as its own process and reads ONLY its own data file.
It downloads the global model, trains locally, optionally applies Differential Privacy
noise and Secure Aggregation masks, and uploads only the protected weight update.
Usage: python hospital_client.py <config.json> "<Hospital A>"
"""
import hashlib, json, sys, time, urllib.request
import numpy as np
import pandas as pd
from models import LogisticRegressionFL, NeuralNetFL
from security import clip_update, add_dp_noise

cfg = json.load(open(sys.argv[1]))
me = sys.argv[2]
NAMES, N = cfg["hospitals"], sum(cfg["sizes"].values())
data = pd.read_csv(f"{cfg['out']}/hospitals/{me.replace(' ', '_').lower()}.csv")
X, y = data.drop(columns="target").values, data["target"].values
model = (LogisticRegressionFL if cfg["model"] == "lr" else NeuralNetFL)(X.shape[1], seed=1)
rng = np.random.default_rng(abs(hash(me)) % (2 ** 32))
URL = f"http://127.0.0.1:{cfg['port']}"


def call(path, payload=None):
    req = urllib.request.Request(URL + path, data=None if payload is None else json.dumps(payload).encode())
    return json.loads(urllib.request.urlopen(req, timeout=400).read())


def pair_mask(a, b, r, dim):
    seed = int(hashlib.sha256(f"{cfg['secret']}|{a}|{b}|{r}".encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed).normal(0, 1.0, dim)


for attempt in range(50):                      # wait for the server to come up
    try:
        call("/model?round=0"); break
    except Exception:
        time.sleep(0.2)

print(f"[{me}] ready with {len(y)} private patients (they never leave this process)", flush=True)
for r in range(cfg["rounds"] + 1):
    resp = call(f"/model?round={r}")
    if resp["done"]:
        break
    g = np.array(resp["weights"])
    model.set_flat(g)
    model.train_local(X, y, epochs=cfg["local_epochs"], lr=cfg["lr"], batch_size=16, seed=r)
    delta = model.get_flat() - g
    plain = delta.copy()
    if cfg["epsilon"] is not None:
        delta = add_dp_noise(clip_update(delta, cfg["clip_norm"]), cfg["epsilon"], cfg["clip_norm"], rng)
    delta = delta * len(y) / N
    if cfg["secure"]:
        for other in NAMES:
            if other != me:
                a, b = sorted([me, other])
                m = pair_mask(a, b, r, len(delta))
                delta = delta + m if me == a else delta - m
    if r == 0:
        print(f"[{me}] round 0: true update starts {np.round(plain[:3], 3).tolist()}  ->  "
              f"sent to server (protected) {np.round(delta[:3], 3).tolist()}", flush=True)
    call("/update", {"hospital": me, "round": r, "update": delta.tolist()})
print(f"[{me}] finished", flush=True)
