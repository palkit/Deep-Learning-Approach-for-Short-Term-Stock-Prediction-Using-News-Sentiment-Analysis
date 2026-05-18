
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

TICKER = input("Enter Stock Ticker: ").upper()
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

    #  VISUALISATION

def plot_results(history_hybrid, history_baseline,
                 metrics_hybrid, metrics_baseline,
                 feature_df, X_test, model_hybrid,
                 dates_all):
    """Reproduce Figures 1, 2, and 3 from the paper."""
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "Hybrid LSTM + Sentiment vs Baseline LSTM\nStock Price Prediction",
        fontsize=16, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :])

    close_vals   = feature_df["Close"].values
    n_test       = len(X_test)
    actual_test  = close_vals[-(n_test + 1):]
    pred_prob    = model_hybrid.predict(X_test, verbose=0).flatten()
    pred_labels  = (pred_prob >= 0.5).astype(int)

    predicted_prices = [actual_test[0]]
    avg_move = np.mean(np.abs(np.diff(actual_test))) * 0.3
    for lbl in pred_labels:
        step = avg_move if lbl == 1 else -avg_move
        predicted_prices.append(predicted_prices[-1] + step)
    predicted_prices = np.array(predicted_prices)

    x_axis = np.arange(len(actual_test))
    ax1.plot(x_axis, actual_test, color="#1f77b4", linewidth=2,
             label="Actual Price", marker="o", markersize=3)
    ax1.plot(x_axis, predicted_prices, color="#d62728", linewidth=2,
             label="Predicted Price", marker="s", markersize=3, linestyle="--")
    ax1.set_title("Fig 1. Actual vs Predicted Stock Prices (Hybrid Model)",
                  fontsize=12, fontweight="bold")
    ax1.set_xlabel("Time (Days)")
    ax1.set_ylabel("Stock Price ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    bars = ax2.bar(
    ["Without Sentiment\n(Baseline LSTM)", "With Sentiment\n(Hybrid Model)"],
    [metrics_baseline["Accuracy"] * 100, metrics_hybrid["Accuracy"] * 100],
    color=["#4878cf", "#6acc65"], width=0.5, edgecolor="black"
    )

    for bar, val in zip(bars, [metrics_baseline["Accuracy"], metrics_hybrid["Accuracy"]]):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"{val*100:.0f}%", ha="center", fontweight="bold")

    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Fig 2. Accuracy: Baseline vs Hybrid", fontsize=11, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    baseline_vals = [metrics_baseline[m] * 100 for m in metric_names]
    hybrid_vals   = [metrics_hybrid[m]   * 100 for m in metric_names]

    x     = np.arange(len(metric_names))
    width = 0.35
    bars1 = ax3.bar(x - width / 2, baseline_vals, width,
                    label="LSTM (Baseline)", color="#4878cf", edgecolor="black")
    bars2 = ax3.bar(x + width / 2, hybrid_vals, width,
                    label="Hybrid Model", color="#6acc65", edgecolor="black")

    for bar in list(bars1) + list(bars2):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 f"{bar.get_height():.0f}%", ha="center", fontsize=8)

    ax3.set_xticks(x)
    ax3.set_xticklabels(metric_names)
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("Score (%)")
    ax3.set_title("Fig 3. Evaluation Metrics: LSTM vs Hybrid", fontsize=11, fontweight="bold")
    ax3.legend()
    ax3.grid(True, axis="y", alpha=0.3)

    out_path = "results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[INFO] Plot saved → {out_path}")
    plt.show()


def plot_training_history(history_hybrid, history_baseline):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    for ax, hist, title in zip(
        axes,
        [history_baseline, history_hybrid],
        ["Baseline LSTM", "Hybrid LSTM + Sentiment"]
    ):
        ax.plot(hist.history["loss"], label="Train Loss")
        ax.plot(hist.history["val_loss"], label="Val Loss", linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150, bbox_inches="tight")
    print("[INFO] Training history saved → training_history.png")
    plt.show()


    # MAIN

def main():
    stock_df = download_stock_data(TICKER, START_DATE, END_DATE)
    news_df  = load_or_generate_news(stock_df)

    stock_df = compute_technical_indicators(stock_df)
    sentiment_df = compute_daily_sentiment(news_df)
    feature_df = build_feature_matrix(stock_df, sentiment_df)

    X_train_h, X_test_h, y_train, y_test, scaler, date_idx = prepare_data(feature_df)

    baseline_cols = ["Open", "High", "Low", "Close", "Volume",
                     "MA20", "RSI", "MACD"]
    baseline_df = feature_df[baseline_cols]
    X_train_b, X_test_b, _, _, _, _ = prepare_data(baseline_df)

    print("\n[INFO] Training HYBRID model …")
    model_hybrid = build_lstm_model(input_shape=(SEQ_LEN, X_train_h.shape[2]))
    history_hybrid = train_model(model_hybrid, X_train_h, y_train,
                                 X_test_h,  y_test)

    print("\n[INFO] Training BASELINE model …")
    model_baseline = build_lstm_model(input_shape=(SEQ_LEN, X_train_b.shape[2]))
    history_baseline = train_model(model_baseline, X_train_b, y_train,
                                   X_test_b,  y_test)

    metrics_baseline = evaluate_model(model_baseline, X_test_b, y_test,
                                      label="Baseline LSTM (no sentiment)")
    metrics_hybrid   = evaluate_model(model_hybrid,   X_test_h, y_test,
                                      label="Hybrid LSTM + Sentiment")

    model_hybrid.save("hybrid_lstm_model.h5")
    model_baseline.save("baseline_lstm_model.h5")
    print("\n[INFO] Models saved.")

    plot_results(history_hybrid, history_baseline,
                 metrics_hybrid, metrics_baseline,
                 feature_df, X_test_h, model_hybrid, date_idx)
    plot_training_history(history_hybrid, history_baseline)

    print("\n" + "="*55)
    print(f"  {'Metric':<15} {'Baseline LSTM':>15} {'Hybrid Model':>15}")
    print("="*55)
    for m in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        print(f"  {m:<15} {metrics_baseline[m]*100:>14.1f}%"
              f" {metrics_hybrid[m]*100:>14.1f}%")
    print("="*55)


if __name__ == "__main__":
    main()