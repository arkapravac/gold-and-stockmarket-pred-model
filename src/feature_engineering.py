import os 
import pandas as pd 
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD

def add_features(name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    raw_path = os.path.join(project_root, "data", f"{name}_raw.csv")
    df = pd.read_csv(raw_path, index_col=0, parse_dates=True)  # ✅ FIXED HERE

    # Price-based features
    df['Return'] = df['Close'].pct_change()
    df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
    df['Price_Change'] = df['Close'] - df['Open']
    
    # Technical indicators
    df['RSI'] = RSIIndicator(df['Close']).rsi()
    df['MACD'] = MACD(df['Close']).macd()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Volatility
    df['Volatility_10'] = df['Return'].rolling(window=10).std()  # ✅ ADDED BACK
    
    # Lagged returns (t-1 to t-5)
    for i in range(1, 6):
        df[f'Return_lag_{i}'] = df['Return'].shift(i)
    
    # Target: 1 if next day price goes up, else 0
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Remove NaN rows
    df = df.dropna()
    
    # Save
    features_path = os.path.join(project_root, "data", f"{name}_features.csv")
    df.to_csv(features_path)
    print(f"✅ Features saved for {name}: {features_path} | Shape: {df.shape}")

if __name__ == "__main__":
    add_features("gold")
    add_features("stock")
