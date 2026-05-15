
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




#  FEATURE INTEGRATION

def build_feature_matrix(stock_df: pd.DataFrame,
                         sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """Merge OHLCV + indicators + sentiment into a unified DataFrame."""
    if isinstance(stock_df.columns, pd.MultiIndex):
        stock_df.columns = stock_df.columns.get_level_values(0)

    merged = stock_df.join(sentiment_df, how="left")
    merged["Sentiment"].fillna(0, inplace=True)

    feature_cols = ["Open", "High", "Low", "Close", "Volume",
                    "MA20", "RSI", "MACD", "Sentiment"]
    merged = merged[feature_cols].dropna()
    print(f"[INFO] Feature matrix shape: {merged.shape}")
    return merged



#  SEQUENCE PREPARATION

def create_sequences(data: np.ndarray,
                     seq_len: int,
                     target_col: int = 3):
    """Build (X, y) pairs."""
    X, y = [], []
    for i in range(seq_len, len(data) - 1):
        X.append(data[i - seq_len:i, :])
        movement = 1 if data[i + 1, target_col] > data[i, target_col] else 0
        y.append(movement)
    return np.array(X), np.array(y)


def prepare_data(feature_df: pd.DataFrame):
    """Scale features and split into train/test sequences."""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(feature_df.values)

    X, y = create_sequences(scaled, SEQ_LEN)

    split = int(len(X) * (1 - TEST_RATIO))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler, feature_df.index[SEQ_LEN + 1:]

# MODEL ARCHITECTURE

def build_lstm_model(input_shape: tuple,
                     lstm_units: int = LSTM_UNITS,
                     dropout: float = DROPOUT) -> Sequential:
    """
    LSTM-based binary classifier.
    """
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        LSTM(lstm_units // 2, return_sequences=False),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dense(1,  activation="sigmoid")
    ])
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()
    return model


#  TRAINING

def train_model(model: Sequential,
                X_train, y_train,
                X_test,  y_test):
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=0)
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    return history

    
    #  EVALUATION

def evaluate_model(model, X_test, y_test, label: str = "Model"):
    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred      = (y_pred_prob >= 0.5).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Precision : {prec:.4f}  ({prec*100:.1f}%)")
    print(f"  Recall    : {rec:.4f}  ({rec*100:.1f}%)")
    print(f"  F1-Score  : {f1:.4f}  ({f1*100:.1f}%)")
    print(classification_report(y_test, y_pred, target_names=["Down", "Up"]))

    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1}
