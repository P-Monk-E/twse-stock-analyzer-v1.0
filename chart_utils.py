# ================================
# /mnt/data/chart_utils.py
# K + MA5/10/20 + BB20±2；RSI(14)；MACD 柱體 + KDJ-J
# 統一日期線、移除休市日/假日、60m 移除夜間
# ================================
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

# --- 統一化 OHLC（關鍵修復：避免圖表空白） ---
def _ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """將輸入資料清洗成帶 DatetimeIndex 的 OHLC。
    為什麼：來源欄位大小寫/日期欄/時區常不一致，會讓 Plotly 的 x 軸失效。"""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])

    x = df.copy()

    # 1) 取得日期索引
    if not isinstance(x.index, pd.DatetimeIndex):
        for c in ["Date", "date", "Datetime", "datetime", "Time", "time"]:
            if c in x.columns:
                x[c] = pd.to_datetime(x[c], errors="coerce")
                x = x.set_index(c)
                break
    if not isinstance(x.index, pd.DatetimeIndex):
        x.index = pd.to_datetime(x.index, errors="coerce")

    # 2) 去除時區（Plotly 與 yfinance 比較穩）
    if isinstance(x.index, pd.DatetimeIndex) and x.index.tz is not None:
        x.index = x.index.tz_convert(None)

    # 3) 對齊欄名
    cols = {c.lower(): c for c in x.columns}
    def pick(name: str) -> str | None: return cols.get(name.lower())
    o, h, l, c = pick("open"), pick("high"), pick("low"), pick("close")
    if not all([o, h, l, c]):
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])

    x = x[[o, h, l, c]].rename(columns={o: "Open", h: "High", l: "Low", c: "Close"})
    x = x.apply(pd.to_numeric, errors="coerce")
    x = x[~x.index.duplicated(keep="last")].sort_index()
    x = x.dropna(how="any")
    return x

# --- 指標 ---
def _ma(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["MA5"] = out["Close"].rolling(5).mean()
    out["MA10"] = out["Close"].rolling(10).mean()
    out["MA20"] = out["Close"].rolling(20).mean()
    return out

def _bb(df: pd.DataFrame, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    m = df["Close"].rolling(n).mean()
    s = df["Close"].rolling(n).std()
    return pd.DataFrame({"BB_MID": m, "BB_UPPER": m + k * s, "BB_LOWER": m - k * s}, index=df.index)

def _macd_hist(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_f = s.ewm(span=fast, adjust=False).mean()
    ema_s = s.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    sig = macd.ewm(span=signal, adjust=False).mean()
    out = (macd - sig).rename("MACD_HIST")
    return out

def _kdj_j(df: pd.DataFrame, n: int = 9, ks: int = 3, ds: int = 3) -> pd.Series:
    low_n = df["Low"].rolling(n).min()
    high_n = df["High"].rolling(n).max()
    rsv = (df["Close"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100.0
    k = rsv.rolling(ks).mean()
    d = k.rolling(ds).mean()
    return (3 * k - 2 * d).rename("KDJ_J")

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1 / n, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename("RSI")

# --- 休市日/夜間隱藏 ---
def _compute_rangebreaks(idx: pd.DatetimeIndex, is_intraday: bool) -> list[dict]:
    if not isinstance(idx, pd.DatetimeIndex) or idx.empty:
        return []
    breaks = [dict(bounds=["sat", "mon"])]
    all_days = pd.date_range(idx.min().normalize(), idx.max().normalize(), freq="D")
    trade_days = pd.DatetimeIndex(pd.to_datetime(idx.date)).unique()
    weekday_non_trade = all_days.difference(trade_days)
    weekday_non_trade = weekday_non_trade[weekday_non_trade.weekday < 5]
    if len(weekday_non_trade) > 0:
        breaks.append(dict(values=weekday_non_trade))
    if is_intraday:
        breaks.append(dict(pattern="hour", bounds=[14, 9]))  # 台股非盤時間
    return breaks

# --- 主圖 ---
def plot_candlestick_with_indicators(
    df: pd.DataFrame,
    title: str = "",
    height: int = 800,
    uirevision_key: Optional[str] = "tv_like",
) -> go.Figure:
    data = _ensure_ohlc(df)
    if data.empty:
        raise ValueError("Empty or invalid OHLC dataframe")

    # 判斷是否為 60m 內頻
    is_intraday = False
    if len(data) > 2:
        step = data.index.to_series().diff().dt.total_seconds().median()
        is_intraday = pd.notna(step) and step <= 3660

    data = _ma(data)
    bb = _bb(data)
    macd_h = _macd_hist(data["Close"])
    kdj_j = _kdj_j(data)
    rsi = _rsi(data["Close"])
    breaks = _compute_rangebreaks(data.index, is_intraday=is_intraday)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.56, 0.18, 0.26], specs=[[{}], [{}], [{"secondary_y": True}]],
    )

    # Row1
    fig.add_trace(go.Candlestick(
        x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"],
        name="Price", increasing_line_width=1, decreasing_line_width=1
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["MA5"], name="MA5"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["MA10"], name="MA10"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["MA20"], name="MA20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=bb.index, y=bb["BB_MID"], name="BB20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=bb.index, y=bb["BB_UPPER"], name="+2σ", line=dict(dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=bb.index, y=bb["BB_LOWER"], name="-2σ", line=dict(dash="dot")), row=1, col=1)

    # Row2
    fig.add_trace(go.Scatter(x=rsi.index, y=rsi, name="RSI(14)"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", row=2, col=1)

    # Row3
    fig.add_trace(go.Bar(x=macd_h.index, y=macd_h, name="MACD Hist", opacity=0.75),
                  row=3, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=kdj_j.index, y=kdj_j, name="KDJ J", mode="lines"),
                  row=3, col=1, secondary_y=True)

    fig.update_layout(
        title=title, height=height, dragmode="pan",
        hovermode="x unified", uirevision=uirevision_key,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
        spikedistance=-1, hoverdistance=0,
        xaxis=dict(type="date", showspikes=True, spikemode="across", spikesnap="cursor",
                   rangebreaks=breaks, rangeslider=dict(visible=False), showline=True, ticks="outside"),
        xaxis2=dict(showspikes=True, spikemode="across", spikesnap="cursor",
                    rangebreaks=breaks, showline=True, ticks="outside"),
        xaxis3=dict(showspikes=True, spikemode="across", spikesnap="cursor",
                    rangebreaks=breaks, showline=True, ticks="outside"),
    )
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
    return fig
