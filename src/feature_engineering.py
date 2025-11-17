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
    df['dow'] = df.index.dayofweek  # ✅ Now this will work
    df['month'] = df.index.month
