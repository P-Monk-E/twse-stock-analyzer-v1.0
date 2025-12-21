# =========================================
# /mnt/data/chart_utils.py
# =========================================
from __future__ import annotations
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 互動設定（保留全螢幕、滾輪縮放、拿掉框選等）
PLOTLY_TV_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "lasso2d", "select2d", "autoScale2d", "toggleSpikelines",
        "zoom2d", "zoomIn2d", "zoomOut2d", "resetScale2d", "pan2d"
    ],
    "modeBarButtonsToAdd": ["toggleFullscreen"],
    "toImageButtonOptions": {"format": "png"},
}

# ---------- robust OHLC standardizer ----------
def _ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    x = df.copy()

    # 確保 DatetimeIndex
    if not isinstance(x.index, pd.DatetimeIndex):
        for c in ["Date", "date", "Datetime", "datetime", "Time", "time"]:
            if c in x.columns:
                x[c] = pd.to_datetime(x[c], errors="coerce")
                x = x.set_index(c)
                break
    if not isinstance(x.index, pd.DatetimeIndex):
        x.index = pd.to_datetime(x.index, errors="coerce")

    # 去 tz → tz-naive（避免 rangebreaks 問題）
    if isinstance(x.index, pd.DatetimeIndex) and x.index.tz is not None:
        x.index = x.index.tz_convert(None)

    x = x[~x.index.duplicated(keep="last")].sort_index()

    cols = {c.lower(): c for c in x.columns}
    def pick(k: str): return cols.get(k.lower())
    o, h, l, c = pick("open"), pick("high"), pick("low"), pick("close")
    if not all([o, h, l, c]):
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])

    out = x[[o, h, l, c]].rename(columns={o:"Open", h:"High", l:"Low", c:"Close"})
    out = out.apply(pd.to_numeric, errors="coerce")
    return out.dropna(how="any")

# ---------- 指標 ----------
def _ma(df: pd.DataFrame) -> pd.DataFrame:
    y = df.copy()
    y["MA5"]  = y["Close"].rolling(5).mean()
    y["MA10"] = y["Close"].rolling(10).mean()
    y["MA20"] = y["Close"].rolling(20).mean()
    return y

def _bb(df: pd.DataFrame, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    m = df["Close"].rolling(n).mean()
    s = df["Close"].rolling(n).std()
    return pd.DataFrame({"BB_MID": m, "BB_UPPER": m + k*s, "BB_LOWER": m - k*s}, index=df.index)

def _macd_hist(s: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    ema_f = s.ewm(span=fast, adjust=False).mean()
    ema_s = s.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    sig  = macd.ewm(span=signal, adjust=False).mean()
    return (macd - sig).rename("MACD_HIST")

def _kdj_j(df: pd.DataFrame, n=9, ks=3, ds=3) -> pd.Series:
    low_n = df["Low"].rolling(n).min()
    high_n = df["High"].rolling(n).max()
    rsv = (df["Close"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100.0
    k = rsv.rolling(ks).mean()
    d = k.rolling(ds).mean()
    return (3*k - 2*d).rename("KDJ_J")

def _rsi(close: pd.Series, n=14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return (100 - 100/(1+rs)).rename("RSI")

# ---------
