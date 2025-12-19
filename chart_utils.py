# =========================================
# /mnt/data/chart_utils.py
# 強化成 TV-like 互動體驗：滑輪縮放、十字線、保留縮放狀態、無 rangeslider
# =========================================
from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go


# 為 Streamlit 的 plotly_chart 準備的統一設定（啟用滑輪縮放、隱藏 logo、精簡工具列）
PLOTLY_TV_CONFIG = {
    "scrollZoom": True,                # ← 滑鼠滾輪縮放
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "lasso2d", "select2d", "autoScale2d", "toggleSpikelines"
    ],
    "toImageButtonOptions": {"format": "png"},
}


def _add_mas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["MA5"] = out["Close"].rolling(5).mean()
    out["MA10"] = out["Close"].rolling(10).mean()
    out["MA20"] = out["Close"].rolling(20).mean()
    return out


def plot_candlestick_with_ma(
    df: pd.DataFrame,
    title: str = "",
    height: int = 520,
    uirevision_key: Optional[str] = "tv_like_candles",
) -> go.Figure:
    """
    TV-like 互動 K 線：可滑輪縮放、十字線、保留縮放狀態（避免 rerun 後重置）。
    - 關閉 rangeslider（放大後更清楚，不會被壓縮）。
    - dragmode 設 pan（右鍵或工具列仍可框選縮放）。
    - hovermode 'x' + spikes 提供類似 TV 的十字線。
    """
    if df is None or df.empty:
        raise ValueError("Empty dataframe for candlestick chart")

    df = _add_mas(df)

    fig = go.Figure()

    # K 線（線條細一點，縮放後更清楚）
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
    # MA 線
    fig.add_trace(go.Scatter(x=df.index, y=df["MA5"], name="MA5"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA10"], name="MA10"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20"))

    # 版面與互動
    fig.update_layout(
        title=title,
        height=height,
        dragmode="pan",              # ← 以拖曳移動圖表
        hovermode="x",               # ← 單一垂直游標
        uirevision=uirevision_key,   # ← Streamlit rerun 後保留縮放/視窗狀態
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(
            type="date",
            rangeslider=dict(visible=False),  # ← 拿掉下方灰色縮圖
            showspikes=True,
            spikemode="across",               # ← 橫跨整張圖的十字線
            spikesnap="cursor",
            showline=True,
            ticks="outside",
            tickmode="auto",
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

    # 互動手感微調（縮放後 candle 間距不要太擠）
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1)

    return fig
