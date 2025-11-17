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
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_class: np.ndarray = None) -> Dict[str, float]:
    """Calculate comprehensive metrics"""
    metrics = {}
    
    # Ensure arrays are the same length
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    # Regression metrics
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else float('inf')
    
    # Directional metrics
    metrics['directional_accuracy'] = directional_accuracy(y_true, y_pred)
    
    # If classification predictions are provided
    if y_pred_class is not None and len(y_pred_class) > 0:
        # Create binary true direction
        y_true_binary = (y_true[1:] - y_true[:-1]) > 0
        y_pred_class_trimmed = y_pred_class[:len(y_true_binary)]
        
        if len(y_true_binary) == len(y_pred_class_trimmed):
            metrics['classification_accuracy'] = accuracy_score(y_true_binary, y_pred_class_trimmed)
            metrics['f1_score'] = f1_score(y_true_binary, y_pred_class_trimmed, zero_division=0)
            if len(np.unique(y_true_binary)) > 1:  # Only if we have both classes
                metrics['auc_roc'] = roc_auc_score(y_true_binary, y_pred_class_trimmed)
    
    return metrics
