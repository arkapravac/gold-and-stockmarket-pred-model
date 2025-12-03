# src/bayesian_models.py
import numpy as np
import pandas as pd
import pymc as pm
import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from src.utils.data import load_data, load_feature_columns
from src.utils.metrics import calculate_bayesian_metrics
from src.utils.viz import plot_bayesian_forecast

class BayesianTimeSeriesModel:
    def __init__(self, asset: str, n_samples: int = 1000, n_chains: int = 4):
        self.asset = asset
        self.n_samples = n_samples
        self.n_chains = n_chains  # 4 chains per Vehtari et al. (2019)
        self.scaler = None
        self.feature_cols = None
        self.train_data = None
        self.test_data = None
        self.idata = None

    def prepare_data(self, test_size: float = 0.2):
        print(f"Loading {self.asset} data...")
        df = load_data(self.asset, "processed")
        all_features = load_feature_columns(self.asset)
        
        # Using only 5 robust features to ensure convergence
        CORE_FEATURES = ['RSI', 'Volatility_10', 'Return_lag_1', 'SMA_20', 'MACD']
        self.feature_cols = [f for f in CORE_FEATURES if f in all_features]
        if not self.feature_cols:
            self.feature_cols = all_features[:5]
        
        X = df[self.feature_cols].values
        y = df['Close_next'].values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]

        n_train = int(len(X) * (1 - test_size))
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        self.train_data = (X_train_scaled, y_train)
        self.test_data = (X_test_scaled, y_test)
        print(f"Training: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")

   
