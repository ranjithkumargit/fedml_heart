"""Random Forest: federated by tree-ensemble aggregation (trees can't be averaged like
weights), plus centralized baselines trained on pooled data for comparison only."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def federated_random_forest(hospitals, trees_per_hospital=30, seed=0):
    trees, last = [], None
    for i, (X, y) in enumerate(hospitals.values()):
        last = RandomForestClassifier(n_estimators=trees_per_hospital, max_depth=6,
                                      random_state=seed + i).fit(X, y)
        trees.extend(last.estimators_)
    rf = RandomForestClassifier(n_estimators=len(trees))
    rf.estimators_ = trees
    rf.classes_, rf.n_classes_ = last.classes_, last.n_classes_
    rf.n_outputs_, rf.n_features_in_ = last.n_outputs_, last.n_features_in_
    return rf


def centralized_models(X, y, seed=0):
    lr = LogisticRegression(max_iter=1000).fit(X, y)
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=seed).fit(X, y)
    return lr, rf
