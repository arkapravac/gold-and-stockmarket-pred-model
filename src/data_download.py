import yfinance as yf
import os
import config as cfg
import pandas as pd
import config as cfg
import pandas as pd

def download_asset(ticker, name, start="2010-01-01", end="2025-10-23"):
    print(f"Downloading {name} data ({ticker})...")
    data = yf.download(ticker, start=start, end=end)
    
    
    n_cols = len(data.columns)
    if n_cols == 5:
        
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    elif n_cols == 6:
        
        data.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    else:
        print(f"WARNING!! Unexpected column count: {n_cols}. Columns: {data.columns.tolist()}")
       
        pass
        
    csv_path = os.path.join(cfg.DATA_RAW, f"{name}_raw.csv")
    
    
    n_cols = len(data.columns)
    if n_cols == 5:
        
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    elif n_cols == 6:
        
        data.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    else:
        print(f"WARNING!! Unexpected column count: {n_cols}. Columns: {data.columns.tolist()}")
       
        pass
    
    csv_path = os.path.join(cfg.DATA_RAW, f"{name}_raw.csv")
    data.to_csv(csv_path)
    print(f"Saved to: {csv_path} | Shape: {data.shape}")
    
if __name__ == "__main__":
    # Clear existing raw files first
    import shutil
    if cfg.DATA_RAW.exists():
        shutil.rmtree(cfg.DATA_RAW)
    cfg.DATA_RAW.mkdir(parents=True, exist_ok=True)
    
    # Clear existing raw files first
    import shutil
    if cfg.DATA_RAW.exists():
        shutil.rmtree(cfg.DATA_RAW)
    cfg.DATA_RAW.mkdir(parents=True, exist_ok=True)
    
    download_asset("GC=F", "gold")    
    download_asset("SPY", "stock")    