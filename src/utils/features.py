import pandas as pd
import numpy as np
from typing import List
import config as cfg
def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Volume Weighted Average Price"""
    return (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to dataframe"""
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Simple moving averages
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_30'] = df['Close'].rolling(30).mean()
    df['SMA_100'] = df['Close'].rolling(100).mean()
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    df['BB_width'] = df['BB_upper'] - df['BB_lower']
    
    # Momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    
    # Price ratios
    df['High_Low_ratio'] = df['High'] / df['Low']
    df['Close_Open_ratio'] = df['Close'] / df['Open']
    
    # ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = high_low.combine(high_close, max).combine(low_close, max)
    df['ATR'] = tr.rolling(14).mean()
    
    return df

def get_feature_columns(df: pd.DataFrame, exclude_cols: List[str] = None) -> List[str]:
    """Get list of feature columns (excluding targets and datetime features)"""
    if exclude_cols is None:
        exclude_cols = ['Close_next', 'Target_dir']
    
    return [col for col in df.columns if col not in exclude_cols]

def add_lagged_features(df: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
    """Add lagged versions of key features"""
    for i in range(1, lags + 1):
        df[f'Return_lag_{i}'] = df['Return'].shift(i)
        df[f'Volume_lag_{i}'] = df['Volume'].shift(i)
        df[f'Close_lag_{i}'] = df['Close'].shift(i)
        df[f'High_lag_{i}'] = df['High'].shift(i)
        df[f'Low_lag_{i}'] = df['Low'].shift(i)
    
    return df
 def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based features"""
    df['dow'] = df.index.dayofweek  # 0=Monday, 6=Sunday
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['day_of_year'] = df.index.dayofyear
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    
    # Business day features
    df['is_month_start'] = df.index.is_month_start.astype(int)
    df['is_month_end'] = df.index.is_month_end.astype(int)
    df['is_quarter_start'] = df.index.is_quarter_start.astype(int)
    df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
    
    return df

