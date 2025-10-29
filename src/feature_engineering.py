import os 
import pandas as pd 
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD

def add_features(name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    raw_path = os.path.join(project_root, "data", f"{name}_raw.csv")
    # Reading CSV with index_col=0 AND converting to numeric
    df = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    # Forcing OHLCV columns to numeric (in case of string errors)
    ohlcv = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[ohlcv] = df[ohlcv].apply(pd.to_numeric, errors='coerce')
    # Dropping any rows that comes with NaN values 
    df = df.dropna(subset=ohlcv)
