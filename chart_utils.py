from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 給前端用的 Plotly 設定（保留放大、隱藏多餘按鈕）
PLOTLY_TV_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines"],
    "toImageButtonOptions": {"format": "png"},
}

# ---- utils ----
def _ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """確保存在 OHLC 欄位與 DatetimeIndex（去 tz），並依時間排序。"""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    x = df.copy()

    if not isinstance(x.index, pd.DatetimeIndex):
        # 嘗試用常見欄位成為 index
        for c in ["Date", "date", "Datetime", "datetime", "Time", "time"]:
            if c in x.columns:
                x[c] = pd.to_datetime(x[c], errors="coerce")
                x = x.set_index(c)
                break
    if not isinstance(x.index, pd.DatetimeIndex):
        x.index = pd.to_datetime(x.index, errors="coerce")

    if isinstance(x.index, pd.DatetimeIndex) and x.index.tz is not None:
        x.index = x.index.tz_convert(None)

    cols = {c.lower(): c for c in x.columns}
    def pick(k: str) -> Optional[str]: return cols.get(k.lower())
    o, h, l, c = pick("open"), pick("high"), pick("low"), pick("close")
    if not all([o, h, l, c]):
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])

    out = x[[o, h, l, c]].rename(columns={o: "Open", h: "High", l: "Low", c: "Close"})
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.dropna(how="any")

# ---- indicators ----
def _ma(df: pd.DataFrame) -> pd.DataFrame:
    y = df.copy()
    y["MA5"]  = y["Close"].rolling(5).mean()
    y["MA10"] = y["Close"].rolling(10).mean()
    y["MA20"] = y["Close"].rolling(20).mean()
    return y

def _bb(df: pd.DataFrame, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = df["Close"].rolling(n).mean()
    std = df["Close"].rolling(n).std()
    return pd.DataFrame(
        {"BB_MID": mid, "BB_UPPER": mid + k * std, "BB_LOWER": mid - k * std},
        index=df.index,
    )

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0)
    down = -d.clip(upper=0.0)
    ru = up.ewm(alpha=1/n, adjust=False).mean()
    rd = down.ewm(alpha=1/n, adjust=False).mean()
    rs = ru / rd.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename("RSI")

def _macd_hist(close: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd  = ema_f - ema_s
    sig   = macd.ewm(span=signal, adjust=False).mean()
    return (macd - sig).rename("MACD_HIST")

def _kdj_j(df: pd.DataFrame, n=9, ks=3, ds=3) -> pd.Series:
    low_n  = df["Low"].rolling(n).min()
    high_n = df["High"].rolling(n).max()
    rsv = (df["Close"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100.0
    k = rsv.rolling(ks).mean()
    d = k.rolling(ds).mean()
    return (3 * k - 2 * d).rename("KDJ_J")

# ---- plotting ----
def plot_candlestick_with_indicators(
    df: pd.DataFrame,
    title: str = "",
    height: int = 820,
    uirevision_key: Optional[str] = "tv_like",
) -> go.Figure:
    """日K：K線 + MA5/10/20 + BB(20,2σ) + RSI + MACD Hist + KDJ J"""
    data = _ensure_ohlc(df)
    if data.empty:
        raise ValueError("Empty OHLC dataframe")

    data = _ma(data)
    bb = _bb(data)
    rsi = _rsi(data["Close"])
    macd_h = _macd_hist(data["Close"])
    kdj_j = _kdj_j(data)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.56, 0.18, 0.26], specs=[[{}], [{}], [{"secondary_y": True}]],
    )

    # Row 1: Candle + MA + BB（全部連續實線；BB 半透明實線）
    fig.add_trace(go.Candlestick(
        x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"],
        name="Price", increasing_line_width=1, decreasing_line_width=1
    ), row=1, col=1)

    for name in ["MA5", "MA10", "MA20"]:
        fig.add_trace(go.Scatter(
            x=data.index, y=data[name], name=name,
            mode="lines", line=dict(width=2)  # 實線
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=bb.index, y=bb["BB_MID"], name="BB20",
        mode="lines", line=dict(width=2), opacity=0.55
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=bb.index, y=bb["BB_UPPER"], name="+2σ",
        mode="lines", line=dict(width=2), opacity=0.35
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=bb.index, y=bb["BB_LOWER"], name="-2σ",
        mode="lines", line=dict(width=2), opacity=0.35
    ), row=1, col=1)

    # Row 2: RSI（連續實線 + 70/30虛線）
    fig.add_trace(go.Scatter(
        x=rsi.index, y=rsi, name="RSI(14)",
        mode="lines", line=dict(width=2)
    ), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", row=2, col=1)

    # Row 3: MACD 柱體 + KDJ J（J 走副軸）
    fig.add_trace(go.Bar(x=macd_h.index, y=macd_h, name="MACD Hist", opacity=0.85),
                  row=3, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(
        x=kdj_j.index, y=kdj_j, name="KDJ J",
        mode="lines", line=dict(width=2.2)
    ), row=3, col=1, secondary_y=True)

    # 互動：三圖共用日期線 + 日期文字
    rangebreaks = [dict(bounds=["sat", "mon"])]  # 移除週末；日K通常就只有交易日
    fig.update_layout(
        title=title, height=height, dragmode="pan",
        hovermode="x unified", hoverlabel=dict(namelength=-1),
        uirevision=uirevision_key,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        spikedistance=-1, hoverdistance=100,
        xaxis =dict(type="date", rangebreaks=rangebreaks, rangeslider=dict(visible=False), showline=True, ticks="outside"),
        xaxis2=dict(type="date", rangebreaks=rangebreaks, showline=True, ticks="outside"),
        xaxis3=dict(type="date", rangebreaks=rangebreaks, showline=True, ticks="outside"),
    )
    for ax in ("xaxis", "xaxis2", "xaxis3"):
        fig.layout[ax].update(
            showspikes=True, spikemode="across+toaxis+marker",
            spikesnap="cursor", spikethickness=1.5, spikedash="dot",
            spikecolor="black", hoverformat="%Y/%m/%d",
        )

    fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor",
                     showline=True, ticks="outside", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="MACD", zeroline=True, row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="KDJ-J", range=[-20, 120], row=3, col=1, secondary_y=True)

    # 網格
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, row=1, col=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, row=2, col=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, row=3, col=1, secondary_y=False)
    fig.update_yaxes(showgrid=False, row=3, col=1, secondary_y=True)

    return fig
