# =========================================
# /mnt/data/chart_utils.py
# TV-like 互動 K 線：滑輪縮放、十字線、保留縮放狀態、無 rangeslider
# 並提供 MA 與基礎重採樣工具。
# =========================================
from __future__ import annotations

from typing import Optional, Literal
import pandas as pd
import plotly.graph_objects as go

Timeframe = Literal["60m", "D", "W", "M"]

# 給 streamlit 的統一 config（滑輪縮放、精簡工具列）
PLOTLY_TV_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines"],
    "toImageButtonOptions": {"format": "png"},
}


def add_ma(df: pd.DataFrame) -> pd.DataFrame:
    """不改動原 df；新增 MA5/10/20。"""
    out = df.copy()
    out["MA5"] = out["Close"].rolling(5).mean()
    out["MA10"] = out["Close"].rolling(10).mean()
    out["MA20"] = out["Close"].rolling(20).mean()
    return out


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    將日線或更細資料重採樣為 W/M 等頻率。
    需要 DatetimeIndex，且包含 Open/High/Low/Close 欄。
    """
    ohlc = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
    }
    out = df.resample(rule).agg(ohlc).dropna(how="any")
    return out


def plot_candlestick_with_ma(
    df: pd.DataFrame,
    title: str = "",
    height: int = 520,
    uirevision_key: Optional[str] = "tv_like_candles",
) -> go.Figure:
    """
    互動 K 線：
    - 滑輪縮放（scrollZoom）
    - 十字準星（spikes）
    - dragmode=pan
    - 保留縮放狀態（uirevision）
    - 移除 rangeslider
    """
    if df is None or df.empty:
        raise ValueError("Empty dataframe for candlestick chart")

    df = add_ma(df)

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_width=1,
            decreasing_line_width=1,
        )
    )
    fig.add_trace(go.Scatter(x=df.index, y=df["MA5"], name="MA5"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA10"], name="MA10"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20"))

    fig.update_layout(
        title=title,
        height=height,
        dragmode="pan",
        hovermode="x",
        uirevision=uirevision_key,  # 保留縮放與視窗狀態
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(
            type="date",
            rangeslider=dict(visible=False),
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            showline=True,
            ticks="outside",
        ),
        yaxis=dict(
            fixedrange=False,
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            showline=True,
            ticks="outside",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1)
    return fig
