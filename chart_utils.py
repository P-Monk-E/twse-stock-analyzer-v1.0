# =========================================
# /mnt/data/chart_utils.py
# =========================================
from __future__ import annotations
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 互動設定（保留全螢幕、滾輪縮放、拿掉框選等）
PLOTLY_TV_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "lasso2d", "select2d", "autoScale2d", "toggleSpikelines",
        "zoom2d", "zoomIn2d", "zoomOut2d", "resetScale2d", "pan2d"
    ],
    "modeBarButtonsToAdd": ["toggleFullscreen"],
    "toImageButtonOptions": {"format": "png"},
}

# ---------- robust OHLC standardizer ----------
def _ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    x = df.copy()

    # 確保 DatetimeIndex
    if not isinstance(x.index, pd.DatetimeIndex):
        for c in ["Date", "date", "Datetime", "datetime", "Time", "time"]:
            if c in x.columns:
                x[c] = pd.to_datetime(x[c], errors="coerce")
                x = x.set_index(c)
                break
    if not isinstance(x.index, pd.DatetimeIndex):
        x.index = pd.to_datetime(x.index, errors="coerce")

    # 去 tz → tz-naive（避免 rangebreaks 問題）
    if isinstance(x.index, pd.DatetimeIndex) and x.index.tz is not None:
        x.index = x.index.tz_convert(None)

    x = x[~x.index.duplicated(keep="last")].sort_index()

    cols = {c.lower(): c for c in x.columns}
    def pick(k: str): return cols.get(k.lower())
    o, h, l, c = pick("open"), pick("high"), pick("low"), pick("close")
    if not all([o, h, l, c]):
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])

    out = x[[o, h, l, c]].rename(columns={o:"Open", h:"High", l:"Low", c:"Close"})
    out = out.apply(pd.to_numeric, errors="coerce")
    return out.dropna(how="any")

# ---------- 指標 ----------
def _ma(df: pd.DataFrame) -> pd.DataFrame:
    y = df.copy()
    y["MA5"]  = y["Close"].rolling(5).mean()
    y["MA10"] = y["Close"].rolling(10).mean()
    y["MA20"] = y["Close"].rolling(20).mean()
    return y

def _bb(df: pd.DataFrame, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    m = df["Close"].rolling(n).mean()
    s = df["Close"].rolling(n).std()
    return pd.DataFrame({"BB_MID": m, "BB_UPPER": m + k*s, "BB_LOWER": m - k*s}, index=df.index)

def _macd_hist(s: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    ema_f = s.ewm(span=fast, adjust=False).mean()
    ema_s = s.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    sig  = macd.ewm(span=signal, adjust=False).mean()
    return (macd - sig).rename("MACD_HIST")

def _kdj_j(df: pd.DataFrame, n=9, ks=3, ds=3) -> pd.Series:
    low_n = df["Low"].rolling(n).min()
    high_n = df["High"].rolling(n).max()
    rsv = (df["Close"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100.0
    k = rsv.rolling(ks).mean()
    d = k.rolling(ds).mean()
    return (3*k - 2*d).rename("KDJ_J")

def _rsi(close: pd.Series, n=14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return (100 - 100/(1+rs)).rename("RSI")

# ---------- RSI 背離偵測（多/空/二度） ----------
def detect_rsi_divergence(
    df: pd.DataFrame,
    rsi_period: int = 14,
    window: int = 2,           # 標記局部高低點用（左右各 window 根）
    min_separation: int = 5,   # 兩個枢紐點至少間隔 K 根
    lookback: int = 250,       # 回看長度
) -> Dict[str, object]:
    """
    回傳：
      {
        "bullish": True/False,          # 多頭背離（低點）
        "bearish": True/False,          # 空頭背離（高點）
        "bullish_double": True/False,   # 二度多頭背離
        "bearish_double": True/False,   # 二度空頭背離
        "points": {
            "bullish": [(x0,y0),(x1,y1)],  # 供畫線/箭頭用（RSI 子圖）
            "bearish": [(x0,y0),(x1,y1)]
        }
      }
    """
    data = _ensure_ohlc(df)
    if data.empty:
        return {"bullish": False, "bearish": False, "bullish_double": False, "bearish_double": False, "points": {}}

    data = data.tail(lookback).copy()
    rsi = _rsi(data["Close"], n=rsi_period)
    c = data["Close"]

    def pivots(series: pd.Series, is_low: bool) -> List[int]:
        s = series.values
        idxs: List[int] = []
        for i in range(window, len(s)-window):
            seg = s[i-window:i+window+1]
            if is_low and s[i] == float(np.nanmin(seg)):
                idxs.append(i)
            if (not is_low) and s[i] == float(np.nanmax(seg)):
                idxs.append(i)
        # 合併太近的點（僅保留靠後者）
        filtered: List[int] = []
        for i in idxs:
            if not filtered or (i - filtered[-1]) >= min_separation:
                filtered.append(i)
            else:
                filtered[-1] = i
        return filtered

    lows  = pivots(c, True)
    highs = pivots(c, False)

    def check(piv_idx: List[int], kind: str) -> Tuple[bool, bool, List[Tuple[pd.Timestamp,float]] | None]:
        single = False; double = False; pts = None
        if len(piv_idx) >= 2:
            p2, p1 = piv_idx[-2], piv_idx[-1]
            if kind == "bull":
                single = (c.iloc[p1] < c.iloc[p2]) and (rsi.iloc[p1] > rsi.iloc[p2])
            else:
                single = (c.iloc[p1] > c.iloc[p2]) and (rsi.iloc[p1] < rsi.iloc[p2])
            if single:
                pts = [(data.index[p2], float(rsi.iloc[p2])), (data.index[p1], float(rsi.iloc[p1]))]
        if len(piv_idx) >= 3:
            p3, p2, p1 = piv_idx[-3], piv_idx[-2], piv_idx[-1]
            if kind == "bull":
                double = ((c.iloc[p2] < c.iloc[p3]) and (rsi.iloc[p2] > rsi.iloc[p3]) and
                          (c.iloc[p1] < c.iloc[p2]) and (rsi.iloc[p1] > rsi.iloc[p2]))
            else:
                double = ((c.iloc[p2] > c.iloc[p3]) and (rsi.iloc[p2] < rsi.iloc[p3]) and
                          (c.iloc[p1] > c.iloc[p2]) and (rsi.iloc[p1] < rsi.iloc[p2]))
        return single, double, pts

    bul_s, bul_d, bul_pts = check(lows, "bull")
    ber_s, ber_d, ber_pts = check(highs, "bear")

    points = {}
    if bul_s and bul_pts: points["bullish"] = bul_pts
    if ber_s and ber_pts: points["bearish"] = ber_pts

    return {
        "bullish": bul_s, "bearish": ber_s,
        "bullish_double": bul_d, "bearish_double": ber_d,
        "points": points,
    }

# ---------- 主圖 ----------
def plot_candlestick_with_indicators(
    df: pd.DataFrame,
    *,
    title: str = "",
    height: int = 820,
    uirevision_key: Optional[str] = "tv_like",
) -> go.Figure:
    """日K：K棒 + MA5/10/20 + BB(20,2σ) + RSI + MACD Hist + KDJ J。"""
    data = _ensure_ohlc(df)
    if data.empty:
        raise ValueError("Empty or invalid OHLC dataframe")

    data = _ma(data)
    bb = _bb(data)
    macd_h = _macd_hist(data["Close"])
    kdj_j  = _kdj_j(data)
    rsi    = _rsi(data["Close"])

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.56, 0.18, 0.26], specs=[[{}], [{}], [{"secondary_y": True}]],
    )

    # Row 1: K + MA + BB
    fig.add_trace(go.Candlestick(
        x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"],
        name="Price", increasing_line_width=1, decreasing_line_width=1
    ), row=1, col=1)

    for name in ["MA5", "MA10", "MA20"]:
        fig.add_trace(go.Scatter(
            x=data.index, y=data[name], name=name,
            mode="lines", connectgaps=True, line=dict(width=2, dash="solid")
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=bb.index, y=bb["BB_MID"], name="BB20",
        mode="lines", connectgaps=True, line=dict(width=2, dash="solid"), opacity=0.55
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=bb.index, y=bb["BB_UPPER"], name="+2σ",
        mode="lines", connectgaps=True, line=dict(width=2, dash="solid"), opacity=0.35
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=bb.index, y=bb["BB_LOWER"], name="-2σ",
        mode="lines", connectgaps=True, line=dict(width=2, dash="solid"), opacity=0.35
    ), row=1, col=1)

    # Row 2: RSI（+ 80/20 線）
    fig.add_trace(go.Scatter(
        x=rsi.index, y=rsi, name="RSI(14)",
        mode="lines", connectgaps=True, line=dict(width=2, dash="solid")
    ), row=2, col=1)
    fig.add_hline(y=80, line_dash="solid", row=2, col=1)
    fig.add_hline(y=20, line_dash="solid", row=2, col=1)

    # === a) RSI 背離可視化：連線＋箭頭標註（多頭▲、空頭▼） ===
    try:
        _div = detect_rsi_divergence(data)
        pts = _div.get("points", {})
        # 多頭：連線 + ▲ + label
        if "bullish" in pts:
            xs = [pts["bullish"][0][0], pts["bullish"][1][0]]
            ys = [pts["bullish"][0][1], pts["bullish"][1][1]]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                     line=dict(width=1), name="BullDiv", showlegend=False),
                          row=2, col=1)
            fig.add_trace(go.Scatter(x=[xs[-1]], y=[ys[-1]], mode="markers+text",
                                     marker_symbol="triangle-up", marker_size=10,
                                     text=["多頭背離"], textposition="bottom center",
                                     name="BullDiv", showlegend=False),
                          row=2, col=1)
        # 空頭：連線 + ▼ + label
        if "bearish" in pts:
            xs = [pts["bearish"][0][0], pts["bearish"][1][0]]
            ys = [pts["bearish"][0][1], pts["bearish"][1][1]]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                     line=dict(width=1), name="BearDiv", showlegend=False),
                          row=2, col=1)
            fig.add_trace(go.Scatter(x=[xs[-1]], y=[ys[-1]], mode="markers+text",
                                     marker_symbol="triangle-down", marker_size=10,
                                     text=["空頭背離"], textposition="top center",
                                     name="BearDiv", showlegend=False),
                          row=2, col=1)
    except Exception:
        pass

    # Row 3: MACD 柱體 + KDJ J
    fig.add_trace(go.Bar(x=macd_h.index, y=macd_h, name="MACD Hist", opacity=0.85),
                  row=3, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(
        x=kdj_j.index, y=kdj_j, name="KDJ J",
        mode="lines", connectgaps=True, line=dict(width=2.2, dash="solid")
    ), row=3, col=1, secondary_y=True)

    # 互動 / 座標軸
    rangebreaks = [dict(bounds=["sat", "mon"])]
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
            spikesnap="cursor", spikethickness=1.5, spikedash="solid",
            spikecolor="black", hoverformat="%Y/%m/%d",
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
    return fig
