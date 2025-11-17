import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import config as cfg
import seaborn as sns
def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Predictions vs Actual", save_path: str = None):
    """Plot predictions vs actual values"""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual', alpha=0.7)
    plt.plot(y_pred, label='Predicted', alpha=0.7)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
def plot_feature_importance(feature_names: list, importances: np.ndarray, save_path: str = None):
    """Plot feature importance"""
    plt.figure(figsize=(10, 6))
    indices = np.argsort(importances)[::-1]
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.title('Feature Importance')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
def plot_time_series_comparison(df: pd.DataFrame, columns: list, title: str = "Time Series Comparison", save_path: str = None):
    """Plot multiple time series on the same chart"""
    plt.figure(figsize=(12, 6))
    for col in columns:
        if col in df.columns:
            plt.plot(df.index, df[col], label=col, alpha=0.7)
    
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
