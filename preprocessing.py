"""Data Preprocessing Module: load, clean (duplicates, missing values), scale."""
import numpy as np
import pandas as pd

TARGET = "target"


def load_heart(path="data/heart.csv"):
    df = pd.read_csv(path)
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]
    df = df.drop_duplicates().reset_index(drop=True)
    features = [c for c in df.columns if c != TARGET]
    for c in features:                       # median imputation for missing values
        df[c] = df[c].fillna(df[c].median())
    X = df[features].values.astype(float)
    y = (df[TARGET].values > 0).astype(int)  # 1 = disease present
    return X, y, df, features


class StandardScalerNP:
    """Zero-mean / unit-variance scaler. Fit on TRAINING data only (no leakage)."""
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_
