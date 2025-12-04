# src/ml_models.py
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from src.utils.data import load_data, load_feature_columns

def train_ml_models(asset="gold"):
    # Load same 5 features as Bayesian model
    df = load_data(asset, "processed")
    feature_cols = ['RSI', 'Volatility_10', 'Return_lag_1', 'SMA_20', 'MACD']
    X = df[feature_cols].values
    y = df['Close_next'].values
    
    # Chronological split (same as Bayesian)
    n_train = int(len(X) * 0.8)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    # Train XGBoost
    xgb = XGBRegressor(n_estimators=100, max_depth=4)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    
    # Save
    pred_dir = cfg.PREDICTIONS_DIR / asset
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.save(pred_dir / "xgboost_pred.npy", xgb_pred)
    print("✅ XGBoost trained and saved")
