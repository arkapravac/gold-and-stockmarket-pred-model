import yfinance as yf
import os
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
    
    
