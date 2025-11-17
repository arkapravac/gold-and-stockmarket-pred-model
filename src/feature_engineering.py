import os, json
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
import config as cfg

def add_features(name, save=True):
    raw_path = os.path.join(cfg.DATA_RAW, f"{name}_raw.csv")
    df = pd.read_csv(raw_path, parse_dates=True, index_col=0)
    
    
    df.index = pd.to_datetime(df.index)
    
    ohlcv = ['Open','High','Low','Close','Volume']
    df[ohlcv] = df[ohlcv].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=ohlcv)
    # basic features
    df['Return'] = df['Close'].pct_change()
    df['High_Low_Pct'] = (df['High'] - df['Low'])/df['Close']
    df['Price_Change'] = df['Close'] - df['Open']
    df['RSI'] = RSIIndicator(df['Close']).rsi()
    df['MACD'] = MACD(df['Close']).macd()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['Volatility_10'] = df['Return'].rolling(10).std()
     # calendar
    df['dow'] = df.index.dayofweek  
    df['month'] = df.index.month
# lags
    for i in range(1,6):
        df[f'Return_lag_{i}'] = df['Return'].shift(i)
        df[f'Vol_lag_{i}'] = df['Volume'].shift(i)
# targets: regression and classification
    df['Close_next'] = df['Close'].shift(-1)
    df['Target_dir'] = (df['Close_next'] > df['Close']).astype(int)

    df = df.dropna()
    features_path = os.path.join(cfg.DATA_PROCESSED, f"{name}_features.csv")
    os.makedirs(cfg.DATA_PROCESSED, exist_ok=True)
    if save:
        df.to_csv(features_path)
    feature_cols = [c for c in df.columns if c not in ['Close_next','Target_dir']]
# save feature list
    with open(os.path.join(cfg.DATA_PROCESSED, f"{name}_feature_cols.json"), "w") as f:
        json.dump(feature_cols, f)
    print("Saved", features_path, "shape", df.shape)
    return df

if __name__ == "__main__":
    add_features("gold")
    add_features("stock")
