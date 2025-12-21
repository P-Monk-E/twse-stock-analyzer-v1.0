
from __future__ import annotations
from typing import Optional, Iterable, Sequence
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

    cols = {c.lower(): c for c in x.columns}
    def pick(k: str) -> Optional[str]: return cols.get(k.lower())
    o, h, l, c = pick("open"), pick("high"), pick("low"), pick("close")
    if not all([o, h, l, c]):
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])

    out = x[[o, h, l, c]].rename(columns={o:"Open", h:"High", l:"Low", c:"Close"})
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.dropna(how="any")

# ---------- 指標 ----------
def _ma(df: pd.DataFrame, periods: Sequence[int] = (5,10,20,60,200)) -> pd.DataFrame:
    y = df.copy()
    for n in periods:
        y[f"MA{n}"] = y["Close"].rolling(int(n)).mean()
    return y

def _ema(df: pd.DataFrame, periods: Sequence[int] = (5,10,30)) -> pd.DataFrame:
    y = pd.DataFrame(index=df.index)
    for n in periods:
        y[f"EMA{n}"] = df["Close"].ewm(span=int(n), adjust=False).mean()
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

def _infer_holiday_gaps(index: pd.DatetimeIndex) -> list:
    """找出資料區間內「缺少的平日」（當作休市日）以供 Plotly rangebreaks 隱藏。"""
    if len(index) == 0:
        return []
    # 只處理「日線」：normalize 成日期，避免跨時區與時分秒
    d0 = index.min().normalize()
    d1 = index.max().normalize()
    # 只保留週一~週五
    all_weekdays = pd.date_range(d0, d1, freq="B")  # business days (Mon-Fri)
    have_days = pd.to_datetime(pd.Index(index.normalize().unique())).sort_values()
    # 缺少的平日 = 休市日（例如國定假日）
    missing = all_weekdays.difference(have_days)
    # 回傳字串清單，Plotly rangebreaks.values 支援字串或 datetime
    return [pd.Timestamp(d).strftime("%Y-%m-%d") for d in missing]

# ---------- 主圖 ----------
def plot_candlestick_with_indicators(
    df: pd.DataFrame,
    title: str = "",
    height: int = 820,
    uirevision_key: Optional[str] = "tv_like",
    # 新增：各指標顯示開關（可傳入 tuple/list 選擇要畫哪些週期）
    show_ma: Sequence[int] | bool = (5,10,20,60,200),
    show_ema: Sequence[int] | bool = (5,10,30),
    show_bb: bool = True,
    show_rsi: bool = True,
    show_macd: bool = True,
    show_kdj: bool = True,
    # 新增：是否跳過（隱藏）休市日（平日但無資料）
    skip_holidays: bool = True,
) -> go.Figure:
    """
    日K：K棒 + MA/EMA(可選) + BB(20,2σ,可選) + RSI(可選) + MACD Hist(可選) + KDJ J(可選)
    - 休市日（平日但沒交易）會被自動偵測並隱藏，週末亦隱藏。
    - 所有線條皆連續實線；BB 上下軌半透明。
    """
    data = _ensure_ohlc(df)
    if data.empty:
        raise ValueError("Empty or invalid OHLC dataframe")

    # 計算所有可能用到的指標，後面依旗標決定是否畫出
    data_ma = _ma(data)  # 內含 MA5/10/20/60/200
    data_ema = _ema(data)  # 內含 EMA5/10/30
    bb = _bb(data)
    macd_h = _macd_hist(data["Close"])
    kdj_j  = _kdj_j(data)
    rsi    = _rsi(data["Close"])

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.56, 0.18, 0.26], specs=[[{}], [{}], [{"secondary_y": True}]],
    )

    # Row 1: K + MA/EMA + BB（全部「連續實線」）
    fig.add_trace(go.Candlestick(
        x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"],
        name="Price", increasing_line_width=1, decreasing_line_width=1
    ), row=1, col=1)

    # --- MA 選擇性顯示 ---
    if show_ma:
        ma_periods: Iterable[int] = show_ma if not isinstance(show_ma, bool) else (5,10,20,60,200)
        for n in ma_periods:
            col = f"MA{int(n)}"
            if col in data_ma.columns:
                fig.add_trace(go.Scatter(
                    x=data_ma.index, y=data_ma[col], name=col,
                    mode="lines", connectgaps=True, line=dict(width=2, dash="solid")
                ), row=1, col=1)

    # --- EMA 選擇性顯示 ---
    if show_ema:
        ema_periods: Iterable[int] = show_ema if not isinstance(show_ema, bool) else (5,10,30)
        for n in ema_periods:
            col = f"EMA{int(n)}"
            if col in data_ema.columns:
                fig.add_trace(go.Scatter(
                    x=data_ema.index, y=data_ema[col], name=col,
                    mode="lines", connectgaps=True, line=dict(width=1.8, dash="solid")
                ), row=1, col=1)

    # --- BB（選擇性顯示） ---
    if show_bb:
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

    # Row 2: RSI（選擇性顯示；實線 + 80/20）
    if show_rsi:
        fig.add_trace(go.Scatter(
            x=rsi.index, y=rsi, name="RSI(14)",
            mode="lines", connectgaps=True, line=dict(width=2, dash="solid")
        ), row=2, col=1)
        fig.add_hline(y=80, line_dash="solid", row=2, col=1)
        fig.add_hline(y=20, line_dash="solid", row=2, col=1)

    # Row 3: MACD 柱體 + KDJ J（J 為連續實線；皆可選）
    if show_macd:
        fig.add_trace(go.Bar(x=macd_h.index, y=macd_h, name="MACD Hist", opacity=0.85),
                      row=3, col=1, secondary_y=False)
    if show_kdj:
        fig.add_trace(go.Scatter(
            x=kdj_j.index, y=kdj_j, name="KDJ J",
            mode="lines", connectgaps=True, line=dict(width=2.2, dash="solid")
        ), row=3, col=1, secondary_y=True)

    # 互動：三圖共用日期線；日期線也為「實線」
    rangebreaks = [dict(bounds=["sat", "mon"])]
    if skip_holidays:
        holiday_values = _infer_holiday_gaps(data.index)
        if holiday_values:
            rangebreaks.append(dict(values=holiday_values))

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
            spikesnap="cursor", spikethickness=1.5, spikedash="solid",  # 日期線 → 實線
            spikecolor="black", hoverformat="%Y/%m/%d",
        )

    # y 軸與網格
    fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor",
                     showline=True, ticks="outside", row=1, col=1)
    if show_rsi:
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    else:
        fig.update_yaxes(visible=False, row=2, col=1)

    if show_macd:
        fig.update_yaxes(title_text="MACD", zeroline=True, row=3, col=1, secondary_y=False)
    else:
        fig.update_yaxes(visible=False, row=3, col=1, secondary_y=False)

    if show_kdj:
        fig.update_yaxes(title_text="KDJ-J", range=[-20, 120], row=3, col=1, secondary_y=True)
    else:
        fig.update_yaxes(visible=False, row=3, col=1, secondary_y=True)

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, row=1, col=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, row=2, col=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, row=3, col=1, secondary_y=False)
    fig.update_yaxes(showgrid=False, row=3, col=1, secondary_y=True)
    return fig
