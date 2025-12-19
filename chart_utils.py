from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    確保含有 Open/High/Low/Close 欄位與 DatetimeIndex。
    為避免 tz 問題，統一轉成 tz-naive。
    """
    cols = {"Open", "High", "Low", "Close"}
    if not cols.issubset(set(df.columns)):
        raise ValueError("DataFrame 需包含 Open/High/Low/Close 欄位。")
    out = df.copy()
    # index to DatetimeIndex
    if not isinstance(out.index, pd.DatetimeIndex):
        if "Date" in out.columns:
            out = out.set_index(pd.to_datetime(out["Date"]))
        else:
            out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_convert("Asia/Taipei").tz_localize(None)
    out = out.sort_index()
    return out


def plot_candlestick_with_ma(df: pd.DataFrame, title: Optional[str] = None) -> go.Figure:
    """
    日 K 專用：K 棒 + MA5/10/20 + Bollinger(20, 2σ)
    - 線條全為連續實線；BB 使用半透明實線
    - 不顯示休市日期（rangebreaks）
    """
    df = _ensure_ohlc(df)

    # Technicals
    for n in (5, 10, 20):
        df[f"MA{n}"] = df["Close"].rolling(n, min_periods=1).mean()
    mid = df["Close"].rolling(20, min_periods=1).mean()
    std = df["Close"].rolling(20, min_periods=1).std(ddof=0)
    df["BB_MID"] = mid
    df["BB_UPPER"] = mid + 2 * std
    df["BB_LOWER"] = mid - 2 * std

    x = df.index

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=x,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color="#2ca02c",
            decreasing_line_color="#d62728",
            showlegend=True,
        )
    )

    # MA：連續實線
    fig.add_trace(go.Scatter(x=x, y=df["MA5"],  mode="lines", name="MA5",  line=dict(width=2)))
    fig.add_trace(go.Scatter(x=x, y=df["MA10"], mode="lines", name="MA10", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=x, y=df["MA20"], mode="lines", name="MA20", line=dict(width=2)))

    # BB：半透明實線
    fig.add_trace(go.Scatter(x=x, y=df["BB_MID"],   mode="lines", name="BB20", line=dict(width=2, color="rgba(100,100,100,0.7)")))
    fig.add_trace(go.Scatter(x=x, y=df["BB_UPPER"], mode="lines", name="+2σ",  line=dict(width=2, color="rgba(0,180,0,0.35)")))
    fig.add_trace(go.Scatter(x=x, y=df["BB_LOWER"], mode="lines", name="-2σ",  line=dict(width=2, color="rgba(255,140,0,0.35)")))

    fig.update_layout(
        title=title or "",
        xaxis=dict(
            rangeslider=dict(visible=False),
            # 不顯示週末/休市
            rangebreaks=[
                dict(bounds=["sat", "mon"]),
            ],
        ),
        yaxis=dict(side="right"),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    return fig
