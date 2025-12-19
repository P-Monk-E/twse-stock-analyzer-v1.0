# =========================================
# /mnt/data/chart_utils.py
# 主圖：K 線 + MA5/10/20 + 布林通道(BB20,±2)
# 副圖：MACD 柱體(12,26,9) + KDJ(J 線, 9,3,3)，與主圖共用 X 軸
# 互動：滾輪縮放、十字線、保留縮放狀態、無 rangeslider
# =========================================
from __future__ import annotations

from typing import Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Streamlit plotly_chart 的統一設定
PLOTLY_TV_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines"],
    "toImageButtonOptions": {"format": "png"},
}


# -------------------- 指標計算 --------------------
def _add_ma(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["MA5"] = out["Close"].rolling(5).mean()
    out["MA10"] = out["Close"].rolling(10).mean()
    out["MA20"] = out["Close"].rolling(20).mean()
    return out


def _bollinger_bands(df: pd.DataFrame, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = df["Close"].rolling(n).mean()
    std = df["Close"].rolling(n).std()
    upper = mid + k * std
    lower = mid - k * std
    out = pd.DataFrame({"BB_MID": mid, "BB_UPPER": upper, "BB_LOWER": lower}, index=df.index)
    return out


def _macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    # 為流暢與一致性，用 EMA
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    hist.name = "MACD_HIST"
    return hist


def _kdj_j(df: pd.DataFrame, n: int = 9, k_smooth: int = 3, d_smooth: int = 3) -> pd.Series:
    # RSV = (C - L_n) / (H_n - L_n) * 100
    low_n = df["Low"].rolling(n).min()
    high_n = df["High"].rolling(n).max()
    rsv = (df["Close"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100.0
    # K, D 使用移動平均平滑（常見 3 期）
    k = rsv.rolling(k_smooth).mean()
    d = k.rolling(d_smooth).mean()
    j = 3 * k - 2 * d
    j.name = "KDJ_J"
    return j


# -------------------- 圖表繪製 --------------------
def plot_candlestick_with_indicators(
    df: pd.DataFrame,
    title: str = "",
    height: int = 640,
    uirevision_key: Optional[str] = "tv_like_candles",
) -> go.Figure:
    """
    兩行子圖：
      Row1 主圖：K 線 + MA5/10/20 + 布林通道 (BB20,±2)
      Row2 副圖：MACD Histogram（12,26,9）+ KDJ 的 J 線（9,3,3）
    - 與主圖共用 X 軸，時間點垂直對齊
    - 互動：滾輪縮放、十字線、保留縮放、無 rangeslider
    """
    if df is None or df.empty:
        raise ValueError("Empty dataframe for candlestick chart")

    base = df.copy()
    if not isinstance(base.index, pd.DatetimeIndex):
        base.index = pd.to_datetime(base.index)

    # 計算技術指標
    base = _add_ma(base)
    bb = _bollinger_bands(base, 20, 2.0)
    macd_hist = _macd_hist(base["Close"])
    kdj_j = _kdj_j(base)

    # 建立子圖（第二列啟用 secondary_y，以便 MACD 與 KDJ 尺度分離）
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.68, 0.32],
        specs=[[{}], [{"secondary_y": True}]],
    )

    # -------- Row 1：K 線 + MA + 布林通道 --------
    fig.add_trace(
        go.Candlestick(
            x=base.index,
            open=base["Open"],
            high=base["High"],
            low=base["Low"],
            close=base["Close"],
            name="Price",
            increasing_line_width=1,
            decreasing_line_width=1,
        ),
        row=1,
        col=1,
    )
    # MA
    fig.add_trace(go.Scatter(x=base.index, y=base["MA5"], name="MA5"), row=1, col=1)
    fig.add_trace(go.Scatter(x=base.index, y=base["MA10"], name="MA10"), row=1, col=1)
    fig.add_trace(go.Scatter(x=base.index, y=base["MA20"], name="MA20"), row=1, col=1)
    # 布林通道
    fig.add_trace(go.Scatter(x=bb.index, y=bb["BB_MID"], name="BB20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=bb.index, y=bb["BB_UPPER"], name="+2σ", line=dict(dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=bb.index, y=bb["BB_LOWER"], name="-2σ", line=dict(dash="dot")), row=1, col=1)

    # -------- Row 2：MACD 柱體 + KDJ(J) --------
    # MACD 柱體（主 y）
    fig.add_trace(
        go.Bar(
            x=macd_hist.index,
            y=macd_hist,
            name="MACD Hist",
            opacity=0.7,
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    # KDJ J 線（副 y，0~100 常見區間）
    fig.add_trace(
        go.Scatter(
            x=kdj_j.index,
            y=kdj_j,
            name="KDJ J",
            mode="lines",
        ),
        row=2,
        col=1,
        secondary_y=True,
    )

    # 互動與外觀
    fig.update_layout(
        title=title,
        height=height,
        dragmode="pan",
        hovermode="x",
        uirevision=uirevision_key,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            type="date",
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            rangeslider=dict(visible=False),
            showline=True,
            ticks="outside",
        ),
        xaxis2=dict(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            showline=True,
            ticks="outside",
        ),
        yaxis=dict(showspikes=True, spikemode="across", spikesnap="cursor", showline=True, ticks="outside"),
        yaxis2=dict(title="MACD", zeroline=True, showline=True, ticks="outside"),
        yaxis3=dict(title="KDJ-J", range=[-20, 120], showline=True, ticks="outside"),  # 寬鬆以容納 J 超界
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, row=1, col=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, row=2, col=1, secondary_y=False)   # MACD
    fig.update_yaxes(showgrid=False, row=2, col=1, secondary_y=True)                # KDJ-J

    return fig
