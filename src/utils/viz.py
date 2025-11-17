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

def plot_bayesian_forecast(y_true: np.ndarray, y_pred_mean: np.ndarray, y_pred_lower: np.ndarray, y_pred_upper: np.ndarray, title: str = "Bayesian Forecast with Uncertainty", save_path: str = None):
    """Plot Bayesian forecast with credible intervals"""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual', color='black', linewidth=2)
    plt.plot(y_pred_mean, label='Predicted Mean', color='blue', linewidth=2)
    plt.fill_between(range(len(y_pred_mean)), y_pred_lower, y_pred_upper, color='blue', alpha=0.2, label='95% Credible Interval')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_correlation_matrix(df: pd.DataFrame, save_path: str = None):
    """Plot correlation matrix of features"""
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True)
    plt.title('Feature Correlation Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
def plot_model_comparison(metrics_df: pd.DataFrame, metric: str = 'mae', save_path: str = None):
    """Plot comparison of different models based on a metric"""
    plt.figure(figsize=(10, 6))
    models = metrics_df['model']
    values = metrics_df[metric]
    
    plt.bar(models, values)
    plt.title(f'Model Comparison - {metric.upper()}')
    plt.ylabel(metric.upper())
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def save_plot(plt_obj, filename: str, asset: str):
    """Save plot to results directory"""
    plot_dir = cfg.PLOTS_DIR / asset
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt_obj.savefig(plot_dir / filename, dpi=300, bbox_inches='tight')
    plt_obj.close()
