# =========================================
# /mnt/data/chart_utils.py
# 主圖：K + MA5/10/20 + 布林(BB20,±2)
# 中圖：RSI(14)
# 下圖：MACD 柱體(12,26,9) + KDJ J(9,3,3)
# 互動：滾輪縮放、十字線/統一日期線、保留縮放、無 rangeslider
# =========================================
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Streamlit plotly_chart 統一設定
PLOTLY_TV_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines"],
    "toImageButtonOptions": {"format": "png"},
}

# -------------------- 指標計算 --------------------
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
    out = macd - sig
    out.name = "MACD_HIST"
    return out

def _kdj_j(df: pd.DataFrame, n: int = 9, ks: int = 3, ds: int = 3) -> pd.Series:
    low_n = df["Low"].rolling(n).min()
    high_n = df["High"].rolling(n).max()
    rsv = (df["Close"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100.0
    k = rsv.rolling(ks).mean()
    d = k.rolling(ds).mean()
    j = 3 * k - 2 * d
    j.name = "KDJ_J"
    return j

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    # Wilder's RSI
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1 / n, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi.name = "RSI"
    return rsi

# -------------------- 主函式 --------------------
def plot_candlestick_with_indicators(
    df: pd.DataFrame,
    title: str = "",
    height: int = 780,
    uirevision_key: Optional[str] = "tv_like",
) -> go.Figure:
    """
    Row1：K + MA5/10/20 + BB20±2
    Row2：RSI(14)
    Row3：MACD histogram(12,26,9) + KDJ(J, 9,3,3) 於 secondary_y
    - 三列共用 X 軸；hovermode='x unified' → 在任一子圖 hover，三圖共享日期線
    """
    if df is None or df.empty:
        raise ValueError("Empty dataframe")

    data = df.copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    data = _ma(data)
    bb = _bb(data)
    macd_h = _macd_hist(data["Close"])
    kdj_j = _kdj_j(data)
    rsi = _rsi(data["Close"])

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.56, 0.18, 0.26],
        specs=[[{}], [{}], [{"secondary_y": True}]],
    )

    # Row1: Price + MA + BB
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

    # Row2: RSI
    fig.add_trace(go.Scatter(x=rsi.index, y=rsi, name="RSI(14)"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", row=2, col=1)

    # Row3: MACD hist + KDJ J
    fig.add_trace(go.Bar(x=macd_h.index, y=macd_h, name="MACD Hist", opacity=0.75),
                  row=3, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=kdj_j.index, y=kdj_j, name="KDJ J", mode="lines"),
                  row=3, col=1, secondary_y=True)

    # 互動/外觀（三列都開啟 spikelines；用 x unified 共享日期線）
    fig.update_layout(
        title=title,
        height=height,
        dragmode="pan",
        hovermode="x unified",
        uirevision=uirevision_key,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        spikedistance=-1,  # 任何位置都畫出 spike
        hoverdistance=0,
        xaxis=dict(type="date", showspikes=True, spikemode="across", spikesnap="cursor",
                   rangeslider=dict(visible=False), showline=True, ticks="outside"),
        xaxis2=dict(showspikes=True, spikemode="across", spikesnap="cursor", showline=True, ticks="outside"),
        xaxis3=dict(showspikes=True, spikemode="across", spikesnap="cursor", showline=True, ticks="outside"),
    )
    fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor",
                     showline=True, ticks="outside", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], showline=True, ticks="outside",
                     row=2, col=1)
    fig.update_yaxes(title_text="MACD", zeroline=True, showline=True, ticks="outside",
                     row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="KDJ-J", range=[-20, 120], showline=True, ticks="outside",
                     row=3, col=1, secondary_y=True)

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, row=1, col=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, row=2, col=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, row=3, col=1, secondary_y=False)  # MACD 網格
    fig.update_yaxes(showgrid=False, row=3, col=1, secondary_y=True)               # KDJ 無網格
    return fig
