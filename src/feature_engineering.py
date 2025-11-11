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
    # Computation features
    df['Return'] = df['Close'].pct_change()
    df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
    df['Price_Change'] = df['Close'] - df['Open']
    df['RSI'] = RSIIndicator(df['Close']).rsi()
    df['MACD'] = MACD(df['Close']).macd()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Volatility_10'] = df['Return'].rolling(window=10).std()
    for i in range(1, 6):
        df[f'Return_lag_{i}'] = df['Return'].shift(i)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()
    features_path = os.path.join(project_root, "data", f"{name}_features.csv")
    df.to_csv(features_path)
    print(f"✅ Features saved for {name}: {features_path} | Shape: {df.shape}")

if __name__ == "__main__":
    add_features("gold")
    add_features("stock")