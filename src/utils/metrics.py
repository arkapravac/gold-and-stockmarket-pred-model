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
