Predicts whether gold price will go up or down tomorrow using technical indicators and XGBoost.
This Model is trained on the data from yfinance. Agey dekhi ki kora jai tar por README.md update debo.

eta hoye jacche amar project structure


model-gold-and-stock-market-prediction/
│
├── main.py                       Tkinter GUI (user clicks → runs src/ scripts)
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/gold_raw.csv
│   └── processed/gold_features.csv
│
├── src/
│   ├── 01_data_download.py       # Standalone: downloads data
│   ├── 02_feature_engineering.py # Standalone: adds features
│   ├── 03_baseline_models.py     # Trains ARIMA + Prophet → saves predictions
│   ├── 04_ml_models.py           # Trains RF, XGBoost, SVM → saves predictions
│   └── 05_dl_models.py           # Trains RNN, CNN-LSTM → saves predictions
│
├── models/                       # Saves .pkl (ML) or .h5 (DL) models
│
├── core/                         # ← NEW: for shared logic (optional but clean)
│   ├── utils.py                  # e.g., common plotting, metrics
│   └── ensemble.py               # Computes mean + trains meta-learner
│
└── results/
    ├── predictions/
    │   ├── arima.csv
    │   ├── prophet.csv
    │   ├── xgboost.csv
    │   ├── lstm.csv
    │   └── ... 
    ├── ensemble_mean.csv         # Simple average of all
    ├── stacked_ensemble.csv      # Meta-learner output
    ├── metrics.csv               # All models + ensembles
    └── plots/
        └── forecast_comparison.png