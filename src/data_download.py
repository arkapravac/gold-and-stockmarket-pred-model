# src/data_download.py
import yfinance as yf
import os

def download_asset(ticker, name, start="2010-01-01", end="2025-10-23"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)

    print(f"Downloading {name} data ({ticker})...")
    data = yf.download(ticker, start=start, end=end)
    csv_path = os.path.join(data_dir, f"{name}_raw.csv")
    data.to_csv(csv_path)
    print(f"Saved to: {csv_path} | Shape: {data.shape}")

if __name__ == "__main__":
    download_asset("GC=F", "gold")    
    download_asset("SPY", "stock")    