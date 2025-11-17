import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score, roc_auc_score
from typing import Dict, Union, Tuple
import config as cfg
def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate accuracy of direction prediction"""
    if len(y_true) <= 1 or len(y_pred) <= 1:
        return 0.0
    
    y_true_dir = (y_true[1:] - y_true[:-1]) > 0
    y_pred_dir = (y_pred[1:] - y_pred[:-1]) > 0
    
    if len(y_true_dir) != len(y_pred_dir):
        min_len = min(len(y_true_dir), len(y_pred_dir))
        y_true_dir = y_true_dir[:min_len]
        y_pred_dir = y_pred_dir[:min_len]
    
    return accuracy_score(y_true_dir, y_pred_dir)
def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio"""
    if len(returns) == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / 252  # Daily risk free rate
    return (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252) if np.std(excess_returns) != 0 else 0.0

def max_drawdown(returns: np.ndarray) -> float:
    """Calculate maximum drawdown"""
    if len(returns) == 0:
        return 0.0
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return np.min(drawdown)
