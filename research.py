
#   IMPORTS

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



#  DATA COLLECTION

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



#  TECHNICAL INDICATORS

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute MA, RSI, and MACD from close prices."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"].astype(float)

    df["MA20"] = close.rolling(window=20).mean()

    delta   = close.diff()
    gain    = delta.clip(lower=0).rolling(window=14).mean()
    loss    = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs      = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12      = close.ewm(span=12, adjust=False).mean()
    ema26      = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26

    df.dropna(inplace=True)
    return df





# SENTIMENT ANALYSIS

def compute_daily_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score each headline with VADER and aggregate to a daily sentiment score.
    """
    analyzer = SentimentIntensityAnalyzer()

    def score_to_label(compound: float) -> int:
        if compound >= 0.05:
            return 1
        elif compound <= -0.05:
            return -1
        return 0

    scores = news_df["Headline"].apply(
        lambda h: score_to_label(analyzer.polarity_scores(str(h))["compound"])
    )
    daily = scores.groupby(scores.index).mean()
    return daily.rename("Sentiment").to_frame()