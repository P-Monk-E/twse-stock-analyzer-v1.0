# =========================================
# /mnt/data/chart_utils.py
# =========================================
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 只保留「全螢幕」；允許滾輪縮放
PLOTLY_TV_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "modeBarButtonsToAdd": ["toggleFullscreen"],
    "modeBarButtonsToRemove": [
        "zoom2d", "pan2d", "select2d", "lasso2d",
        "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d",
        "hoverClosestCartesian", "hoverCompareCartesian", "toggleSpikelines",
        "toImage",
    ],
    "toImageButtonOptions": {"format": "png"},
}

# ---------- robust OHLC standardizer ----------
def _ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize to ['Open','High','Low','Close'] with ascending DatetimeIndex."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])

    x = df.copy()

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
        return cols.get(k.lower()) or cols.get(k[:1].lower())

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
    3-row plot:
      1) Candlestick + MA(5/10/20) + Bollinger(20,2σ)
      2) RSI(14) + 70/30 lines
      3) MACD histogram（正負四色） + KDJ-J（右軸，0–100；背景區塊）
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
        row_heights=[0.52, 0.14, 0.34],
        specs=[[{}], [{}], [{"secondary_y": True}]],
    )

    # Row 1: K 線 + MA + BB
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
    fig.add_trace(go.Scatter(x=bb.index, y=bb["BB_MID"],   name="BB20", mode="lines", connectgaps=True, line=dict(width=2, dash="solid"), opacity=0.55), row=1, col=1)
    fig.add_trace(go.Scatter(x=bb.index, y=bb["BB_UPPER"], name="+2σ",  mode="lines", connectgaps=True, line=dict(width=2, dash="solid"), opacity=0.35), row=1, col=1)
    fig.add_trace(go.Scatter(x=bb.index, y=bb["BB_LOWER"], name="-2σ",  mode="lines", connectgaps=True, line=dict(width=2, dash="solid"), opacity=0.35), row=1, col=1)

    # Row 2: RSI
    fig.add_trace(
        go.Scatter(
            x=rsi.index, y=rsi, name="RSI(14)",
            mode="lines", connectgaps=True, line=dict(width=2, dash="solid"),
        ),
        row=2, col=1,
    )
    for yv in (70, 30):
        fig.add_hline(y=yv, line_width=1, line_dash="solid", row=2, col=1)

    # Row 3: MACD + KDJ（同圖）
    # 背景區塊（KDJ 的右軸，0–1座標，但我們顯示 0–100）
    fig.add_hrect(y0=0.0, y1=0.2, fillcolor="green", opacity=0.08, line_width=0,
                  row=3, col=1, secondary_y=True)
    fig.add_hrect(y0=0.2, y1=0.8, fillcolor="gray", opacity=0.06, line_width=0,
                  row=3, col=1, secondary_y=True)
    fig.add_hrect(y0=0.8, y1=1.0, fillcolor="red", opacity=0.08, line_width=0,
                  row=3, col=1, secondary_y=True)
    # 20 / 80 虛線
    fig.add_hline(y=0.2, line_width=1, line_dash="dash", row=3, col=1, secondary_y=True)
    fig.add_hline(y=0.8, line_width=1, line_dash="dash", row=3, col=1, secondary_y=True)

    # MACD 柱狀：正/負 + 上升/下降 四色
    hist = macd_h["HIST"].astype(float)
    dh = hist.diff().fillna(0.0)
    colors = []
    for h, d in zip(hist.values, dh.values):
        if h >= 0 and d >= 0:
            colors.append("#169873")   # 正、上升（深綠）
        elif h >= 0 and d < 0:
            colors.append("#7bd4ba")   # 正、下降（淺綠）
        elif h < 0 and d <= 0:
            colors.append("#c24545")   # 負、下降（深紅）
        else:
            colors.append("#e8a3a3")   # 負、上升（淺紅）

    fig.add_trace(
        go.Bar(
            x=macd_h.index,
            y=hist,
            name="MACD-Hist",
            marker_color=colors,
            opacity=1.0,
            marker_line_width=0,
        ),
        row=3, col=1, secondary_y=False,
    )

    # KDJ-J（右軸，以 0–1 繪製，標籤顯示 0–100）
    fig.add_trace(
        go.Scatter(
            x=kdj_j.index, y=kdj_j / 100.0, name="KDJ-J",
            mode="lines", connectgaps=True,
            line=dict(width=2, color="#b23a48"),
        ),
        row=3, col=1, secondary_y=True,
    )

    # 版面與互動
    fig.update_layout(
        title=title or "",
        xaxis_rangeslider_visible=False,
        uirevision=uirevision_key,
        dragmode="pan",
        hovermode="x unified",
        legend=dict(orientation="h", x=0, xanchor="left", y=-0.15, yanchor="top"),
        margin=dict(l=8, r=8, t=48, b=72),
        bargap=0.05,
    )

    # 主圖可縮放 Y
    fig.update_yaxes(row=1, col=1, fixedrange=False, showspikes=True, spikemode="across",
                     spikesnap="cursor", showline=True, ticks="outside")

    # RSI 固定 Y
    fig.update_yaxes(title_text="RSI", range=[0, 100], fixedrange=True, row=2, col=1)

    # MACD 左軸：初始對稱範圍，允許縮放
    if hist.notna().any():
        hmax = float(np.nanmax(np.abs(hist.values)))
        pad = 0.4 * hmax
        fig.update_yaxes(title_text="MACD", range=[-(hmax + pad), (hmax + pad)],
                         fixedrange=False, row=3, col=1, secondary_y=False)
    else:
        fig.update_yaxes(title_text="MACD", fixedrange=False, row=3, col=1, secondary_y=False)

    # KDJ 右軸：0–1，tick 顯示 0–100；允許縮放
    tickvals = np.linspace(0, 1, 6).tolist()
    ticktext = [str(int(v * 100)) for v in tickvals]
    fig.update_yaxes(title_text="KDJ-J", range=[-0.05, 1.05], fixedrange=False,
                     tickvals=tickvals, ticktext=ticktext,
                     row=3, col=1, secondary_y=True)

    # 網格
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, row=1, col=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, row=2, col=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, row=3, col=1, secondary_y=False)
    fig.update_yaxes(showgrid=False, row=3, col=1, secondary_y=True)

    # 隱藏休市日（週末 + 其他非交易日）
    idx = pd.DatetimeIndex(data.index).normalize()
    all_days = pd.date_range(idx.min(), idx.max(), freq="D")
    non_trading = all_days.difference(idx.unique())
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(values=non_trading)])

    return fig
