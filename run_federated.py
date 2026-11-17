"""
One-command REAL federated run: starts the server and 3 hospital processes that talk over
HTTP. Usage: python run_federated.py [--model lr|nn] [--epsilon 5] [--no-secure] [--non-iid]
"""
import argparse, json, subprocess, sys
from prepare_data import prepare

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="lr", choices=["lr", "nn"])
ap.add_argument("--rounds", type=int, default=15)
ap.add_argument("--epsilon", type=float, default=None, help="enable Differential Privacy")
ap.add_argument("--no-secure", action="store_true", help="disable Secure Aggregation")
ap.add_argument("--non-iid", action="store_true")
ap.add_argument("--port", type=int, default=8765)
a = ap.parse_args()

out = "federated_run"
sizes = prepare(out, iid=not a.non_iid)
cfg = {"out": out, "model": a.model, "rounds": a.rounds, "local_epochs": 5,
       "lr": 0.3 if a.model == "lr" else 0.05, "epsilon": a.epsilon, "clip_norm": 1.0,
       "secure": not a.no_secure, "secret": "demo-shared-secret", "port": a.port,
       "hospitals": list(sizes), "sizes": sizes}
json.dump(cfg, open(f"{out}/config.json", "w"))
print(f"Model={a.model} | secure aggregation={cfg['secure']} | DP epsilon={a.epsilon} | hospitals={sizes}\n")

server = subprocess.Popen([sys.executable, "server.py", f"{out}/config.json"])
clients = [subprocess.Popen([sys.executable, "hospital_client.py", f"{out}/config.json", h]) for h in sizes]
for c in clients:
    c.wait()
server.wait()
log = json.load(open(f"{out}/training_log.json"))
print(f"\nFinal global model accuracy on held-out patients: {log[-1]['accuracy']:.3f} (F1 {log[-1]['f1']:.3f})")
print(f"Saved: {out}/global_model.json, {out}/training_log.json")
