# =========================================
# /mnt/data/chart_utils.py
# =========================================
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PLOTLY_TV_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines"],
    "toImageButtonOptions": {"format": "png"},
}

# ---------- robust OHLC standardizer ----------
def _ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize arbitrary price DataFrame to columns: Open, High, Low, Close
    and a clean, ascending DatetimeIndex. Non-numeric values are coerced to NaN.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])

    x = df.copy()

    # Ensure DatetimeIndex
    if not isinstance(x.index, pd.DatetimeIndex):
        for c in ["Date", "date", "Datetime", "datetime", "Time", "time"]:
            if c in x.columns:
                x[c] = pd.to_datetime(x[c], errors="coerce")
                x = x.set_index(c)
                break
    if not isinstance(x.index, pd.DatetimeIndex):
        x.index = pd.to_datetime(x.index, errors="coerce")
    x = x[~x.index.duplicated(keep="last")].sort_index()
    x.index = pd.to_datetime(x.index, utc=True).tz_convert(None)

    cols = {c.lower(): c for c in x.columns}
    def pick(k: str) -> Optional[str]:
        return cols.get(k.lower()) or cols.get(k[:1].lower())  # tolerate O/H/L/C prefixes

    o, h, l, c = pick("open"), pick("high"), pick("low"), pick("close")
    if not all([o, h, l, c]):
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])

    out = x[[o, h, l, c]].rename(columns={o: "Open", h: "High", l: "Low", c: "Close"})
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.dropna(how="any")

# ---------- 指標 ----------
def _ma(df: pd.DataFrame) -> pd.DataFrame:
    y = df.copy()
    y["MA5"] = y["Close"].rolling(5).mean()
    y["MA10"] = y["Close"].rolling(10).mean()
    y["MA20"] = y["Close"].rolling(20).mean()
    return y

def _bb(df: pd.DataFrame, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = df["Close"].rolling(n).mean()
    sd = df["Close"].rolling(n).std()
    return pd.DataFrame(
        {"BB_MID": mid, "BB_UPPER": mid + k * sd, "BB_LOWER": mid - k * sd},
        index=df.index,
    )

def _macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - macd_signal
    return pd.DataFrame({"MACD": macd, "SIGNAL": macd_signal, "HIST": hist}, index=close.index)

def _kdj_j(df: pd.DataFrame, n: int = 9) -> pd.Series:
    low_min = df["Low"].rolling(n).min()
    high_max = df["High"].rolling(n).max()
    rsv = (df["Close"] - low_min) / (high_max - low_min) * 100.0
    k = rsv.ewm(alpha=1/3, adjust=False).mean()
    d = k.ewm(alpha=1/3, adjust=False).mean()
    j = 3 * k - 2 * d
    return j

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi

# ---------- 主圖 ----------
def plot_candlestick_with_indicators(
    df: pd.DataFrame,
    *,
    title: str = "",
    uirevision_key: Optional[str] = None,
) -> go.Figure:
    """
    Build a 3-row plotly chart:
      1) Candlestick + MA(5/10/20) + Bollinger Bands(20, 2σ)
      2) RSI(14) with 70/30 reference lines
      3) MACD(12,26,9) histogram + line + KDJ-J (secondary y)
    **包含 rangebreaks：隱藏週末與其他修市日，避免所有子圖出現縫隙**
    """
    data = _ensure_ohlc(df)
    if data.empty:
        raise ValueError("Empty or invalid OHLC dataframe")

    data = _ma(data)
    bb = _bb(data)
    macd_h = _macd_hist(data["Close"])
    kdj_j = _kdj_j(data)
    rsi = _rsi(data["Close"])

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.56, 0.18, 0.26], specs=[[{}], [{}], [{"secondary_y": True}]],
    )

    # Row 1: K + MA + BB（全部「連續實線」）
    fig.add_trace(
        go.Candlestick(
            x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"],
            name="Price", increasing_line_width=1, decreasing_line_width=1,
        ),
        row=1, col=1,
    )
    for name in ["MA5", "MA10", "MA20"]:
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data[name], name=name,
                mode="lines", connectgaps=True, line=dict(width=2, dash="solid"),
            ),
            row=1, col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=bb.index, y=bb["BB_MID"], name="BB20",
            mode="lines", connectgaps=True, line=dict(width=2, dash="solid"), opacity=0.55,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=bb.index, y=bb["BB_UPPER"], name="+2σ",
            mode="lines", connectgaps=True, line=dict(width=2, dash="solid"), opacity=0.35,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=bb.index, y=bb["BB_LOWER"], name="-2σ",
            mode="lines", connectgaps=True, line=dict(width=2, dash="solid"), opacity=0.35,
        ),
        row=1, col=1,
    )

    # Row 2: RSI（實線 + 70/30 參考線）
    fig.add_trace(
        go.Scatter(
            x=rsi.index, y=rsi, name="RSI(14)",
            mode="lines", connectgaps=True, line=dict(width=2, dash="solid"),
        ),
        row=2, col=1,
    )
    for yv, nm in [(80, "80"), (20, "20")]:
        fig.add_hline(y=yv, line_width=1, line_dash="solid", row=2, col=1)

    # Row 3: MACD histogram + line（實線） + KDJ-J（次 y 軸）
    fig.add_trace(
        go.Bar(x=macd_h.index, y=macd_h["HIST"], name="MACD-Hist", opacity=0.85),
        row=3, col=1, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=macd_h.index, y=macd_h["MACD"], name="MACD",
            mode="lines", connectgaps=True, line=dict(width=2, dash="solid"),
        ),
        row=3, col=1, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=macd_h.index, y=macd_h["SIGNAL"], name="Signal",
            mode="lines", connectgaps=True, line=dict(width=2, dash="solid"),
        ),
        row=3, col=1, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=kdj_j.index, y=kdj_j, name="KDJ-J",
            mode="lines", connectgaps=True, line=dict(width=2, dash="solid")),
        row=3, col=1, secondary_y=True,
    )

    # 版面
    fig.update_layout(
        title=title or "",
        xaxis_rangeslider_visible=False,
        uirevision=uirevision_key,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=8, r=8, t=36, b=8),
        hovermode="x unified",
    )

    # y 軸與網格
    fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor",
                     showline=True, ticks="outside", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="MACD", zeroline=True, row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="KDJ-J", range=[-20, 120], row=3, col=1, secondary_y=True)

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, row=1, col=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, row=2, col=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, row=3, col=1, secondary_y=False)
    fig.update_yaxes(showgrid=False, row=3, col=1, secondary_y=True)

    # ---------------------------
    # 重要：隱藏修市日期（無縫）
    # ---------------------------
    try:
        idx = pd.to_datetime(data.index).normalize()
        all_days = pd.date_range(idx.min(), idx.max(), freq="D")
        trading_days = pd.DatetimeIndex(idx.unique())
        non_trading = all_days.difference(trading_days)
        # 套用到所有 x 軸（含子圖）
        fig.update_xaxes(rangebreaks=[
            dict(bounds=["sat", "mon"]),   # 跳過週末
            dict(values=non_trading),      # 跳過其他非交易日（國定假日等）
        ])
    except Exception:
        # 為穩定性，任何異常都略過，不影響主圖
        pass

    return fig
