# src/utils/data.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import config as cfg

def load_data(asset: str, data_type: str = "processed") -> pd.DataFrame:
    """Load data for a specific asset"""
    if data_type == "raw":
        path = cfg.DATA_RAW / f"{asset}_raw.csv"
    else:
        path = cfg.DATA_PROCESSED / f"{asset}_features.csv"
    
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    return df

def split_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split: train, validation, test"""
    n = len(df)
    test_start = int(n * (1 - test_size))
    val_start = int(n * (1 - test_size - val_size))
    
    train = df.iloc[:val_start]
    val = df.iloc[val_start:test_start]
    test = df.iloc[test_start:]
    
    return train, val, test

def validate_features(df: pd.DataFrame, required_features: List[str]) -> bool:
    """Validate that all required features are present"""
    missing = [f for f in required_features if f not in df.columns]
    if missing:
        print(f"Missing features: {missing}")
        return False
    return True

def save_predictions(predictions: Dict, model_name: str, asset: str):
    """Save predictions to results directory"""
    import json
    pred_dir = cfg.PREDICTIONS_DIR / asset
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    with open(pred_dir / f"{model_name}_predictions.json", 'w') as f:
        json.dump(predictions, f)

def load_feature_columns(asset: str) -> List[str]:
    """Load the list of feature columns for an asset"""
    import json
    feature_file = cfg.DATA_PROCESSED / f"{asset}_feature_cols.json"
    with open(feature_file, 'r') as f:
        return json.load(f)