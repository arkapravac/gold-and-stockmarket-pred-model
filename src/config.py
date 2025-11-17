import os
from pathlib import Path
# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"
# Data settings
TICKERS = {
    "gold": "GC=F",
    "stock": "SPY"
}
DATE_RANGE = ("2010-01-01", "2025-10-23")
# Model settings
RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
# Feature engineering
LOOKBACK_WINDOW = 30
FEATURE_LAGS = 5
# Bayesian model settings
MCMC_SAMPLES = 1000
MCMC_CHAINS = 2
BAYESIAN_TARGET = "Close_next"  

def create_directories():
    """Create all necessary directories"""
    dirs = [
        DATA_RAW, DATA_PROCESSED, MODELS_DIR, 
        RESULTS_DIR, PREDICTIONS_DIR, METRICS_DIR, PLOTS_DIR
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
if __name__ == "__main__":
    create_directories()
    print("Done! Config and directories created!")