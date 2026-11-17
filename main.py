"""
Main pipeline: runs the whole federated system on the Heart Disease dataset over
5 random train/test splits and reports mean +/- std, which is far more trustworthy
than a single lucky split.   Usage:  python main.py
"""
import numpy as np
from experiment import run_experiment
from partition import partition_data

SEEDS = [0, 1, 2, 3, 4]


def summarize(all_runs):
    names = [r["model"] for r in all_runs[0]]
    print(f"{'Model':<36}{'Accuracy':<18}{'Precision':<12}{'Recall':<10}{'F1':<8}")
    for i, name in enumerate(names):
        col = lambda k: np.array([run[i][k] for run in all_runs])
        a = col("accuracy")
        print(f"{name:<36}{a.mean():.3f} +/- {a.std():.3f}   {col('precision').mean():<12.3f}"
              f"{col('recall').mean():<10.3f}{col('f1').mean():<8.3f}")


if __name__ == "__main__":
    print("=" * 78)
    print("FEDERATED ML FOR PRIVACY-PRESERVING HEALTHCARE DATA ANALYSIS (Heart Disease)")
    print("3 hospitals | 15 rounds | 5 local epochs | DP epsilon = 5.0 | 5 random splits")
    print("=" * 78)
    for iid in (True, False):
        print(f"\n--- Hospital data distribution: {'IID (similar patients)' if iid else 'NON-IID (skewed: hospitals see different mixes)'} ---")
        runs = [run_experiment(seed=s, iid=iid)["results"] for s in SEEDS]
        summarize(runs)
    print("\nFederated models never pooled raw patient data; only weight updates were shared.")
