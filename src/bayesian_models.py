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

    def fit(self):
        X_train, y_train = self.train_data
        print(f"Running MCMC ({self.n_samples} samples, {self.n_chains} chains)...")

        with pm.Model() as model:
            # Strong priors for stability
            beta = pm.Normal('beta', mu=0, sigma=0.1, shape=X_train.shape[1])
            alpha = pm.Normal('alpha', mu=np.mean(y_train), sigma=np.std(y_train))
            sigma = pm.HalfNormal('sigma', sigma=np.std(y_train))
            mu = alpha + pm.math.dot(X_train, beta)
            pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_train)

            self.idata = pm.sample(
                draws=self.n_samples,
                chains=self.n_chains,
                tune=1000,
                target_accept=0.90,
                random_seed=cfg.RANDOM_SEED,
                return_inferencedata=True,
                progressbar=True
            )

        # Diagnostics
        summary = pm.summary(self.idata)
        max_rhat = summary['r_hat'].max()
        min_ess = summary['ess_bulk'].min()
        print(f"\nConvergence: R-hat={max_rhat:.3f} {'✅' if max_rhat < 1.01 else '⚠️'}, ESS={min_ess:.0f} {'✅' if min_ess > 100 else '⚠️'}")

    def predict(self, X_test=None):
        if X_test is None:
            X_test, _ = self.test_data
        
        beta_samples = self.idata.posterior['beta'].values
        alpha_samples = self.idata.posterior['alpha'].values
        
        beta_flat = beta_samples.reshape(-1, beta_samples.shape[-1])
        alpha_flat = alpha_samples.flatten()
        
        y_pred = alpha_flat[:, None] + np.dot(beta_flat, X_test.T)
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
        historical_max = np.max(self.train_data[1])
        y_pred = np.clip(y_pred, 0, historical_max * 10)
        
        return y_pred

    
    def evaluate(self):
        _, y_test = self.test_data
        y_pred_samples = self.predict()
        metrics = calculate_bayesian_metrics(y_pred_samples, y_test)
        print(f"\nBayesian Evaluation for {self.asset}:")
        print(f"  Directional Accuracy: {metrics['directional_accuracy']:.4f}")
        print(f"  Credible Interval Coverage: {metrics['credible_interval_coverage']:.4f}")
        return metrics, {'samples': y_pred_samples}

    def plot_results(self, save_path=None):
        _, y_test = self.test_data
        y_pred_samples = self.predict()
        y_mean = np.mean(y_pred_samples, axis=0)
        y_lower = np.percentile(y_pred_samples, 5, axis=0)
        y_upper = np.percentile(y_pred_samples, 95, axis=0)
        plot_bayesian_forecast(y_test, y_mean, y_lower, y_upper,
                              title=f'Bayesian Forecast for {self.asset.upper()}',
                              save_path=save_path)

def run_bayesian_model(asset: str):

