#src/bayesian_models.py
import numpy as np
import pandas as pd
import pymc as pm
import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from src.utils.data import load_data, load_feature_columns
from src.utils.metrics import calculate_bayesian_metrics
from src.utils.viz import plot_bayesian_forecast

class BayesianTimeSeriesModel:
    def __init__(self, asset: str, n_samples: int = 1000, n_chains: int = 4):
        self.asset = asset
        self.n_samples = n_samples
        self.n_chains = n_chains  
        self.scaler = None
        self.feature_cols = None
        self.train_data = None
        self.test_data = None
        self.idata = None
