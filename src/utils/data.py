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
