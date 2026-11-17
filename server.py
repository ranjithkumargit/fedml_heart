"""
Federated Server (cloud layer). Runs as its own process.
Hospitals GET the global model and POST back only weight updates. When all hospitals
have reported for a round, the server sums the (already weighted, masked) updates = FedAvg.
Usage: python server.py <config.json>
"""
import json, sys, threading
import numpy as np
import pandas as pd
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from models import LogisticRegressionFL, NeuralNetFL
from evaluate import evaluate

cfg = json.load(open(sys.argv[1]))
OUT, NAMES = cfg["out"], cfg["hospitals"]
test = pd.read_csv(f"{OUT}/test.csv")
Xte, yte = test.drop(columns="target").values, test["target"].values
model = (LogisticRegressionFL if cfg["model"] == "lr" else NeuralNetFL)(Xte.shape[1], seed=1)
g = model.get_flat()
state = {"round": 0, "updates": {}, "log": []}
cond = threading.Condition()
httpd = None


def finish_round():
    global g
    g = g + sum(np.array(u) for u in state["updates"].values())
    model.set_flat(g)
    r = evaluate(yte, model.predict(Xte), "global")
    state["round"] += 1
    state["updates"] = {}
    state["log"].append({"round": state["round"], "accuracy": r["accuracy"], "f1": r["f1"]})
    print(f"[server] round {state['round']:2d}/{cfg['rounds']} aggregated from {len(NAMES)} hospitals"
          f" | global model test accuracy {r['accuracy']:.3f}", flush=True)
    if state["round"] >= cfg["rounds"]:
        json.dump(state["log"], open(f"{OUT}/training_log.json", "w"))
        json.dump({"model": cfg["model"], "weights": g.tolist()}, open(f"{OUT}/global_model.json", "w"))
        threading.Timer(1.5, lambda: httpd.shutdown()).start()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        r = int(parse_qs(urlparse(self.path).query)["round"][0])
        with cond:
            cond.wait_for(lambda: state["round"] >= r, timeout=300)
            self._send({"round": state["round"], "done": state["round"] >= cfg["rounds"],
                        "weights": g.tolist()})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with cond:
            if body["round"] == state["round"]:
                state["updates"][body["hospital"]] = body["update"]
                if len(state["updates"]) == len(NAMES):
                    finish_round()
                    cond.notify_all()
        self._send({"ok": True})


httpd = ThreadingHTTPServer(("127.0.0.1", cfg["port"]), Handler)
print(f"[server] listening on port {cfg['port']}, waiting for {len(NAMES)} hospitals", flush=True)
httpd.serve_forever()
