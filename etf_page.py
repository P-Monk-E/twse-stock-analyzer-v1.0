# =========================================
# /mnt/data/etf_page.py
# 60m/日 K；補到最新交易日（台北時區 +1 天）；不顯示休市日；三圖共用日期線
# =========================================
from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd
import streamlit as st
import yfinance as yf
import pytz

from risk_grading import grade_alpha, grade_sharpe, grade_treynor, summarize
from portfolio_risk_utils import diversification_warning
from stock_utils import find_ticker_by_name, get_metrics, is_etf, TICKER_NAME_MAP
from chart_utils import plot_candlestick_with_indicators, PLOTLY_TV_CONFIG

def _normalize_tw_ticker_once(sym: str) -> str:
    s = str(sym).upper().strip()
    return s if s.endswith((".TW", ".TWO")) or s.startswith("^") else f"{s}.TW"

@st.cache_data(ttl=1800, show_spinner=False)
def _download_ohlc_intraday(ticker: str, interval: str = "60m", period: str = "90d") -> pd.DataFrame:
    try:
        df = yf.Ticker(_normalize_tw_ticker_once(ticker)).history(period=period, interval=interval, auto_adjust=False)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df[["Open", "High", "Low", "Close"]].dropna(how="any")
    except Exception:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])

@st.cache_data(ttl=1800, show_spinner=False)
def _get_market_close_series(start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.Series]:
    for idx in ["^TWII", "^TAIEX", "^GSPC"]:
        try:
            h = yf.Ticker(idx).history(start=start, end=end, auto_adjust=False)
            if h is not None and not h.empty and "Close" in h:
                s = h["Close"].copy(); s.name = idx; return s
        except Exception:
            continue
    return None

def _prepare_tf_df(ticker: str, base_daily_df: pd.DataFrame, tf_label: str) -> Tuple[pd.DataFrame, str]:
    if tf_label == "60m":
        return _download_ohlc_intraday(ticker, "60m", "90d"), "（60 分鐘）"
    else:
        return base_daily_df.copy(), "（日 K）"

def _kpi_grid(metrics: list[tuple[str, str, str]], cols: int = 4) -> None:
    if not metrics: return
    rows = (len(metrics) + cols - 1) // cols
    it = iter(metrics)
    for _ in range(rows):
        cs = st.columns(cols)
        for c in cs:
            try: name, val, hp = next(it)
            except StopIteration: break
            with c: st.metric(label=name, value=val, help=hp or None)

def render(prefill_symbol: Optional[str] = None) -> None:
    st.header("ETF")
    c1, c2 = st.columns([3, 2])
    with c1:
        default_kw = prefill_symbol or st.session_state.get("last_etf_kw", "0050")
        keyword = st.text_input("輸入 ETF 代碼或名稱", value=default_kw)
    with c2:
        tf_label = st.radio("K 線週期", options=["60m", "日"], index=1, horizontal=True)

    if not keyword:
        st.info("請輸入關鍵字（例：0050 或 台灣50）"); return
    try:
        ticker = find_ticker_by_name(keyword)
        name = TICKER_NAME_MAP.get(ticker, "")
        st.session_state["last_etf_kw"] = keyword

        if not is_etf(ticker):
            st.warning("這不是 ETF，請改至「股票」分頁查詢。"); return

        tz = pytz.timezone("Asia/Taipei")
        now_tpe = pd.Timestamp.now(tz=tz)
        end = now_tpe.normalize() + pd.Timedelta(days=1)  # 包含今日
        start = end - pd.Timedelta(days=366)

        market_close = _get_market_close_series(start, end)
        if market_close is None:
            st.error("抓不到市場指數收盤價（^TWII/^TAIEX/^GSPC）"); return
        rf = 0.01

        stats = get_metrics(ticker, market_close, rf, start, end, is_etf=True)
        if not stats: st.error("查無此 ETF 的歷史資料。"); return

        st.subheader(f"{name or ticker}（{ticker}）")

        grades = {
            "Sharpe": grade_sharpe(stats.get("Sharpe Ratio")),
            "Treynor": grade_treynor(stats.get("Treynor")),
            "Alpha": grade_alpha(stats.get("Alpha")),
        }
        crit, warn, good = summarize(grades)
        if crit: st.error("關鍵風險：" + "、".join(crit))
        if warn: st.warning("注意項：" + "、".join(warn))
        if good: st.success("達標：" + "、".join(good))

        msg = diversification_warning(
            stats.get("Sharpe Ratio"),
            stats.get("Treynor"),
            non_sys_thr=float(st.session_state.get("non_sys_thr", 0.5)),
            sys_thr=float(st.session_state.get("sys_thr", 0.5)),
        )
        if msg: st.warning(msg)

        kpis = [
            ("Alpha", f'{stats.get("Alpha"):.2f}' if stats.get("Alpha") is not None else "—", ""),
            ("Beta", f'{stats.get("Beta"):.2f}' if stats.get("Beta") is not None else "—", ""),
            ("Sharpe", f'{stats.get("Sharpe Ratio"):.2f}' if stats.get("Sharpe Ratio") is not None else "—", ""),
            ("Treynor", f'{stats.get("Treynor"):.2f}' if stats.get("Treynor") is not None else "—", ""),
            ("MADR", f'{stats.get("MADR"):.2f}' if stats.get("MADR") is not None else "—", ""),
            ("配息TTM", f'{stats.get("EPS_TTM"):.2f}' if stats.get("EPS_TTM") is not None else "—", "近四次配息合計"),
        ]
        _kpi_grid(kpis, cols=4)

        base_df: pd.DataFrame = stats["df"].copy()
        if not isinstance(base_df.index, pd.DatetimeIndex): base_df.index = pd.to_datetime(base_df.index)
        tf_df, tf_note = _prepare_tf_df(ticker, base_df, tf_label)
        if tf_df.empty: st.error("查無對應週期的價格資料。"); return

        title = f"{name or ticker}（{ticker}）技術圖 {tf_note}"
        fig = plot_candlestick_with_indicators(tf_df, title=title, uirevision_key=f"{ticker}_{tf_label}")
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_TV_CONFIG)
    except Exception as e:
        st.error(f"❌ 查詢 ETF 失敗：{e}")

def show(prefill_symbol: Optional[str] = None) -> None:
    render(prefill_symbol=prefill_symbol)
