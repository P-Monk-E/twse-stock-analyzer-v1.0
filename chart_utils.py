# =========================================
# /mnt/data/chart_utils.py
# K + MA5/10/20 + BB20±2 ；RSI(14)；MACD Hist + KDJ J
# 統一日期線；移除休市日/假日；60m 移除夜間
# =========================================
from __future__ import annotations
from typing import Optional, Iterable
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
    """標準化成帶 DatetimeIndex 的 OHLC；解決欄名大小寫/Date 欄/時區/排序等問題。"""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])

    tmp = df.copy()

    # 1) 取日期索引
    if not isinstance(tmp.index, pd.DatetimeIndex):
        # 常見情況：有 Date/Datetime 欄
        for c in ["Date", "date", "Datetime", "datetime", "Time", "time"]:
            if c in tmp.columns:
                tmp[c] = pd.to_datetime(tmp[c], errors="coerce")
                tmp = tmp.set_index(c)
                break
    if not isinstance(tmp.index, pd.DatetimeIndex):
        try:
            tmp.index = pd.to_datetime(tmp.index, errors="coerce")
        except Exception:
            pass

    # 2) 去時區（為了 yfinance 與 Plotly）
    if isinstance(tmp.index, pd.DatetimeIndex):
        if tmp.index.tz is not None:
            tmp.index = tmp.index.tz_convert(None)
        else:
            # 有些來源混入 tz-aware 值，保守處理
            tmp.index = pd.to_datetime(tmp.index, utc=False)

    # 3) 欄名標準化
    cols = {c.lower(): c for c in tmp.columns}
    def pick(name: str) -> Optional[str]:
        return cols.get(name.lower())

    o = pick("Open") or pick("open")
    h = pick("High") or pick("high")
    l = pick("Low") or pick("low")
    c = pick("Close") or pick("close")
    if not all([o, h, l, c]):
        # 無法對應，回傳空
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])

    out = tmp[[o, h, l, c]].rename(columns={o: "Open", h: "High", l: "Low", c: "Close"})

    # 4) 數值轉型、排序、去重、去 NA
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(how="any")
    return out

# ---------- 指標 ----------
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
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1 / n, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi.name = "RSI"
    return rsi

# ---------- 休市日/夜間移除 ----------
def _compute_rangebreaks(idx: pd.DatetimeIndex, is_intraday: bool) -> list[dict]:
    if not isinstance(idx, pd.DatetimeIndex) or idx.empty:
        return []
    breaks = [dict(bounds=["sat", "mon"])]
    # 假日：工作日但沒有資料
    all_days = pd.date_range(idx.min().normalize(), idx.max().normalize(), freq="D")
    trade_days = pd.DatetimeIndex(pd.to_datetime(idx.date)).unique()
    non_trade = all_days.difference(trade_days)
    weekday_non_trade = non_trade[non_trade.weekday < 5]
    if len(weekday_non_trade) > 0:
        breaks.append(dict(values=weekday_non_trade))
    if is_intraday:
        breaks.append(dict(pattern="hour", bounds=[14, 9]))  # 台股夜間
    return breaks

# ---------- 主圖 ----------
def plot_candlestick_with_indicators(
    df: pd.DataFrame,
    title: str = "",
    height: int = 800,
    uirevision_key: Optional[str] = "tv_like",
) -> go.Figure:
    data = _ensure_ohlc(df)
    if data.empty:
        raise ValueError("Empty or invalid OHLC dataframe")

    # 60m 粗判斷
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

    # Row2：RSI
    fig.add_trace(go.Scatter(x=rsi.index, y=rsi, name="RSI(14)"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", row=2, col=1)

    # Row3：MACD Hist + KDJ J
    fig.add_trace(go.Bar(x=macd_h.index, y=macd_h, name="MACD Hist", opacity=0.75),
                  row=3, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=kdj_j.index, y=kdj_j, name="KDJ J", mode="lines"),
                  row=3, col=1, secondary_y=True)

    # 互動/外觀
    fig.update_layout(
        title=title, height=height, dragmode="pan",
        hovermode="x unified", uirevision=uirevision_key,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
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
