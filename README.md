# Federated ML for Privacy-Preserving Healthcare Data Analysis
Heart disease prediction trained across 3 simulated hospitals with Federated Averaging (FedAvg),
Differential Privacy and Secure Aggregation. Raw patient data never leaves a hospital.

## Run
    pip install -r requirements.txt
    python main.py               # full experiment: 8 models x 5 random splits, IID and non-IID
    python privacy_tradeoff.py   # accuracy vs DP epsilon
    python run_federated.py --model lr --epsilon 5   # REAL networked run: server + 3 hospital processes over HTTP
    streamlit run dashboard.py   # monitoring dashboard + live prediction + networked run button

## Modules (match the PPT)
| File | PPT module |
|---|---|
| data/heart.csv | Data Collection (UCI Heart Disease, 302 unique patients, 13 features) |
| preprocessing.py | Data Preprocessing (duplicates, missing values, scaling fit on train only) |
| partition.py | Dataset Partitioning (Hospital A/B/C, IID or skewed) |
| models.py | Local Training (Logistic Regression, small Neural Network 13-16-8-1) |
| federated_server.py | Model Update + Federated Aggregation + Global Model Distribution (FedAvg) |
| security.py | Privacy and Security (Differential Privacy, Secure Aggregation) |
| federated_rf.py | Federated Random Forest (tree aggregation) + centralized baselines |
| evaluate.py | Performance Evaluation |
| experiment.py / main.py | End-to-end pipeline |
| prepare_data.py, server.py, hospital_client.py, run_federated.py | Real networked federation: separate server and hospital processes, each hospital reads only its own data file |
| privacy_tradeoff.py | Accuracy vs DP epsilon study |
| dashboard.py | Monitoring Dashboard + Prediction Module |

## Results (mean over 5 random splits, 3 hospitals, 15 rounds, epsilon = 5)
| Model | Accuracy IID | Accuracy non-IID |
|---|---|---|
| Federated LR (FedAvg) | 0.826 | 0.846 |
| Federated LR + DP + Secure Agg | 0.833 | 0.846 |
| Federated Neural Network | 0.777 | 0.774 |
| Federated NN + Secure Agg | 0.777 | 0.774 |
| Federated Random Forest | 0.820 | 0.820 |
| Centralized LR (baseline) | 0.833 | 0.833 |
| Centralized NN (baseline) | 0.790 | 0.790 |
| Centralized RF (baseline) | 0.823 | 0.823 |

Privacy vs accuracy (Federated LR, DP epsilon): 0.1 -> 0.639, 0.5 -> 0.754, 1 -> 0.813, 2 -> 0.820,
5 -> 0.833, 10 -> 0.836, no DP -> 0.826.

## Key points
* Federated models match centralized baselines within noise (std about 0.03-0.06 on 61 test patients).
* Secure Aggregation adds no accuracy cost: masks cancel exactly (identical NN rows prove it).
* DP is a real trade-off: epsilon >= 1 is nearly free, epsilon <= 0.5 hurts clearly.
* The neural network is slightly worse than Logistic Regression: 302 patients is too little data for a
  network. Stated honestly as a limitation.

## Limitations (say these before your guide does)
* Simulation on one machine; there is no real network between hospitals.
* Small dataset; differences under about 3 points are within random-split noise.
* DP noise is calibrated simply (sigma = clip / epsilon), without a formal privacy accountant.
* The neural network is built in NumPy so FedAvg can access its weights directly (no TensorFlow install needed).

## Networked run options
    python run_federated.py --model nn            # neural network instead of Logistic Regression
    python run_federated.py --non-iid --epsilon 5 # skewed hospitals + Differential Privacy
    python run_federated.py --no-secure           # disable Secure Aggregation (for comparison)
Output: per-round global accuracy printed by the server, plus federated_run/training_log.json.
Verified: with and without Secure Aggregation the final accuracy is identical (masks cancel exactly).
Note: the hospitals run as separate processes on localhost; a real deployment would put them on separate machines.
