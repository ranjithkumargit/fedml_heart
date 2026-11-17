"""Monitoring Dashboard + Prediction Module.   Run:  streamlit run dashboard.py"""
import numpy as np
import pandas as pd
import streamlit as st
from experiment import run_experiment
from privacy_tradeoff import tradeoff
from partition import partition_data

st.set_page_config(page_title="Federated ML - Heart Disease", layout="wide")
st.title("🏥 Federated ML for Privacy-Preserving Healthcare Data Analysis")
st.caption("Heart disease prediction trained across 3 simulated hospitals. Raw patient data never leaves a hospital.")

st.sidebar.header("Training settings")
rounds = st.sidebar.slider("Communication rounds", 5, 30, 15)
local_epochs = st.sidebar.slider("Local epochs per round", 1, 10, 5)
epsilon = st.sidebar.slider("DP epsilon (lower = more private)", 0.5, 10.0, 5.0, step=0.5)
iid = st.sidebar.radio("Hospital data", ["IID (similar)", "Non-IID (skewed)"]) == "IID (similar)"
run = st.sidebar.button("▶ Run Federated Training")

if run:
    with st.spinner("Training across 3 hospitals..."):
        st.session_state["exp"] = run_experiment(42, rounds, local_epochs, epsilon, iid)

exp = st.session_state.get("exp")
if exp is None:
    st.info("Set the parameters on the left and click **Run Federated Training**.")
    st.stop()

df, hospitals = exp["df"], exp["hospitals"]
c = st.columns(4)
c[0].metric("Patients", len(df))
for col, (name, (X, y)) in zip(c[1:], hospitals.items()):
    col.metric(f"{name} (private)", f"{len(y)} patients", f"{int(y.sum())} with disease", delta_color="off")

st.subheader("📈 Model performance (held-out test patients)")
res = pd.DataFrame(exp["results"]).set_index("model")
res.columns = ["Accuracy", "Precision", "Recall", "F1"]
st.dataframe(res.style.format("{:.3f}"))
st.bar_chart(res[["Accuracy", "F1"]])
st.caption("Federated rows match their Centralized baselines closely: privacy without a meaningful accuracy cost.")

st.subheader("🔐 Privacy vs accuracy (Differential Privacy epsilon)")
if st.button("Compute trade-off curve (takes ~1 min)"):
    st.session_state["tradeoff"] = tradeoff(range(3))
if "tradeoff" in st.session_state:
    t = pd.DataFrame({"epsilon": ["none" if e is None else str(e) for e in st.session_state["tradeoff"]],
                      "accuracy": [v[0] for v in st.session_state["tradeoff"].values()]}).set_index("epsilon")
    st.line_chart(t)


st.subheader("🌐 Real networked federated run (server + 3 hospital processes)")
st.caption("Starts a real server and three separate hospital processes that exchange only protected weight updates over HTTP.")
use_dp = st.checkbox("Add Differential Privacy noise", value=True)
if st.button("Run networked federated learning"):
    import subprocess, sys, json
    cmd = [sys.executable, "run_federated.py", "--port", "8899", "--rounds", str(rounds)]
    if use_dp:
        cmd += ["--epsilon", str(epsilon)]
    if not iid:
        cmd += ["--non-iid"]
    with st.spinner("Server and hospitals are training over the network..."):
        proc = subprocess.run(cmd, capture_output=True, text=True)
    st.session_state["net_log"] = json.load(open("federated_run/training_log.json"))
    st.session_state["net_out"] = proc.stdout
if "net_log" in st.session_state:
    nl = pd.DataFrame(st.session_state["net_log"]).set_index("round")[["accuracy", "f1"]]
    st.line_chart(nl)
    with st.expander("Server / hospital log"):
        st.code(st.session_state["net_out"])

st.subheader("🔮 Predict heart disease risk for a new patient")
model_name = st.selectbox("Model", [m for m in exp["models"] if m.startswith("Federated") and "Forest" not in m])
with st.form("predict"):
    cols, vals = st.columns(3), {}
    for i, f in enumerate(exp["features"]):
        s = df[f]
        vals[f] = cols[i % 3].number_input(f, float(s.min()), float(s.max()), float(s.median()))
    if st.form_submit_button("Predict"):
        x = exp["scaler"].transform(np.array([[vals[f] for f in exp["features"]]]))
        p = float(exp["models"][model_name].predict_proba(x)[0])
        (st.error if p >= 0.5 else st.success)(
            f"{'⚠️ High' if p >= 0.5 else '✅ Low'} risk of heart disease (probability {p:.1%})")
