# =========================================
# /mnt/data/chart_utils.py
# =========================================
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

def _ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
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
    if isinstance(x.index, pd.DatetimeIndex) and x.index.tz is not None:
        x.index = x.index.tz_convert(None)
    cols = {c.lower(): c for c in x.columns}
    o, h, l, c = cols.get("open"), cols.get("high"), cols.get("low"), cols.get("close")
    if not all([o, h, l, c]):
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    out = x[[o, h, l, c]].rename(columns={o:"Open", h:"High", l:"Low", c:"Close"})
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.dropna(how="any")

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
    d = close.diff()
    up = d.clip(lower=0.0)
    down = -d.clip(upper=0.0)
    ru = up.ewm(alpha=1/n, adjust=False).mean()
    rd = down.ewm(alpha=1/n, adjust=False).mean()
    rs = ru / rd.replace(0, np.nan)
    return (100 - 100/(1+rs)).rename("RSI")

def _compute_rangebreaks(idx: pd.DatetimeIndex, is_intraday: bool) -> list[dict]:
    if not isinstance(idx, pd.DatetimeIndex) or idx.empty:
        return []
    brks = [dict(bounds=["sat", "mon"])]
    all_days = pd.date_range(idx.min().normalize(), idx.max().normalize(), freq="D")
    trade_days = pd.DatetimeIndex(pd.to_datetime(idx.date)).unique()
    weekday_non_trade = all_days.difference(trade_days)
    weekday_non_trade = weekday_non_trade[weekday_non_trade.weekday < 5]
    if len(weekday_non_trade) > 0:
        brks.append(dict(values=weekday_non_trade))
    if is_intraday:
        brks.append(dict(pattern="hour", bounds=[13.5, 9]))  # 13:30~09:00 不顯示
    return brks

def _break_at_session_open(index: pd.DatetimeIndex, s: pd.Series) -> pd.Series:
    if s is None or s.empty:
        return s
    day = pd.Series(index.normalize(), index=index)
    is_open = day.ne(day.shift(1))
    is_open.iloc[0] = False
    out = s.copy()
    out[is_open.to_numpy()] = np.nan  # why: 切斷跨日連線
    return out

def plot_candlestick_with_indicators(
    df: pd.DataFrame,
    title: str = "",
    height: int = 820,
    uirevision_key: Optional[str] = "tv_like",
) -> go.Figure:
    data = _ensure_ohlc(df)
    if data.empty:
        raise ValueError("Empty or invalid OHLC dataframe")

    is_intraday = False
    if len(data) > 2:
        step = data.index.to_series().diff().dt.total_seconds().median()
        is_intraday = pd.notna(step) and step <= 3660

    data = _ma(data)
    bb = _bb(data)
    macd_h = _macd_hist(data["Close"])
    kdj_j  = _kdj_j(data)
    rsi    = _rsi(data["Close"])
    breaks = _compute_rangebreaks(data.index, is_intraday)

    if is_intraday:
        for col in ["MA5", "MA10", "MA20"]:
            data[col] = _break_at_session_open(data.index, data[col])
        for col in ["BB_MID", "BB_UPPER", "BB_LOWER"]:
            bb[col] = _break_at_session_open(bb.index, bb[col])
        rsi   = _break_at_session_open(rsi.index, rsi)
        kdj_j = _break_at_session_open(kdj_j.index, kdj_j)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.56, 0.18, 0.26], specs=[[{}], [{}], [{"secondary_y": True}]],
    )

    fig.add_trace(go.Candlestick(
        x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"],
        name="Price", increasing_line_width=1, decreasing_line_width=1
    ), row=1, col=1)

    for name in ["MA5", "MA10", "MA20"]:
        fig.add_trace(go.Scatter(
            x=data.index, y=data[name], name=name,
            mode="lines", connectgaps=False,
            line=dict(width=2, dash="solid")
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=bb.index, y=bb["BB_MID"], name="BB20",
        mode="lines", connectgaps=False,
        line=dict(width=1.6, dash="solid"), opacity=0.55
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=bb.index, y=bb["BB_UPPER"], name="+2σ",
        mode="lines", connectgaps=False,
        line=dict(width=1.6, dash="solid"), opacity=0.35
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=bb.index, y=bb["BB_LOWER"], name="-2σ",
        mode="lines", connectgaps=False,
        line=dict(width=1.6, dash="solid"), opacity=0.35
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=rsi.index, y=rsi, name="RSI(14)",
        mode="lines", connectgaps=False,
        line=dict(width=2, dash="solid")
    ), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", row=2, col=1)

    fig.add_trace(go.Bar(x=macd_h.index, y=macd_h, name="MACD Hist", opacity=0.85),
                  row=3, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(
        x=kdj_j.index, y=kdj_j, name="KDJ J",
        mode="lines", connectgaps=False,
        line=dict(width=2.2, dash="solid")
    ), row=3, col=1, secondary_y=True)

    fig.update_layout(
        title=title, height=height, dragmode="pan",
        hovermode="x unified",
        hoverlabel=dict(namelength=-1),
        uirevision=uirevision_key,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        spikedistance=-1, hoverdistance=100,
        xaxis =dict(type="date", rangebreaks=breaks, rangeslider=dict(visible=False), showline=True, ticks="outside"),
        xaxis2=dict(type="date", rangebreaks=breaks, showline=True, ticks="outside"),
        xaxis3=dict(type="date", rangebreaks=breaks, showline=True, ticks="outside"),
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

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, row=1, col=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, row=2, col=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, row=3, col=1, secondary_y=False)
    fig.update_yaxes(showgrid=False, row=3, col=1, secondary_y=True)
    return fig
