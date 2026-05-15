
# 0.  IMPORTS

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")
tf.random.set_seed(42)
np.random.seed(42)



# 1.  CONFIGURATION

TICKER      = "AAPL"
START_DATE  = "2020-01-01"
END_DATE    = "2023-12-31"
SEQ_LEN     = 30
TEST_RATIO  = 0.20
EPOCHS      = 50
BATCH_SIZE  = 32
LSTM_UNITS  = 64
DROPOUT     = 0.2
LEARNING_RATE = 1e-3



# 2.  DATA COLLECTION

def download_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance."""
    print(f"[INFO] Downloading stock data for {ticker} …")
    df = yf.download(ticker, start=start, end=end, progress=False)
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    print(f"[INFO] Downloaded {len(df)} trading days.")
    return df


def load_or_generate_news(df: pd.DataFrame) -> pd.DataFrame:
    """
    In a real project, replace this with actual financial news headlines
    fetched from a news API.
    """
    print("[INFO] Generating synthetic news headlines (replace with real data).")
    headlines = []
    for date, row in df.iterrows():
        try:
            close_val = float(row["Close"])
            open_val  = float(row["Open"])
        except (KeyError, TypeError):
            close_val = float(row.iloc[3])
            open_val  = float(row.iloc[0])

        change = close_val - open_val
        if change > 1.0:
            text = f"{TICKER} surges on strong earnings and positive outlook."
        elif change > 0:
            text = f"{TICKER} edges higher amid cautious optimism in markets."
        elif change > -1.0:
            text = f"{TICKER} slips slightly as investors await guidance."
        else:
            text = f"{TICKER} drops sharply on disappointing revenue outlook."
        headlines.append({"Date": date, "Headline": text})

    return pd.DataFrame(headlines).set_index("Date")