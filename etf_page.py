# =========================================
# /mnt/data/etf_page.py
# 只保留日K；主圖加布林；副圖 MACD 柱體 + KDJ(J)
# 嚴格分流（只允許 ETF）；KPI 區顯示；相容 app.py 的 show()
# =========================================
from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st
import yfinance as yf

from risk_grading import grade_alpha, grade_sharpe, grade_treynor, summarize
from portfolio_risk_utils import diversification_warning
from stock_utils import find_ticker_by_name, get_metrics, is_etf, TICKER_NAME_MAP
from chart_utils import plot_candlestick_with_indicators, PLOTLY_TV_CONFIG

@st.cache_data(ttl=1800, show_spinner=False)
def _get_market_close_series(start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.Series]:
    for idx in ["^TWII", "^TAIEX", "^GSPC"]:
        try:
            h = yf.Ticker(idx).history(start=start, end=end)
            if h is not None and not h.empty and "Close" in h:
                s = h["Close"].copy(); s.name = idx; return s
        except Exception:
            continue
    return None

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
    default_kw = prefill_symbol or st.session_state.get("last_etf_kw", "0050")
    keyword = st.text_input("輸入 ETF 代碼或名稱", value=default_kw)
    if not keyword:
        st.info("請輸入關鍵字（例：0050 或 台灣50）"); return
    try:
        ticker = find_ticker_by_name(keyword)
        name = TICKER_NAME_MAP.get(ticker, "")
        st.session_state["last_etf_kw"] = keyword

        if not is_etf(ticker):
            st.warning("這不是 ETF，請改至「股票」分頁查詢。"); return

        end = pd.Timestamp.today().normalize(); start = end - pd.Timedelta(days=365)
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
        title = f"{name or ticker}（{ticker}）技術圖（日 K）"
        fig = plot_candlestick_with_indicators(base_df, title=title, uirevision_key=f"{ticker}_D")
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_TV_CONFIG)
    except Exception as e:
        st.error(f"❌ 查詢 ETF 失敗：{e}")

def show(prefill_symbol: Optional[str] = None) -> None:
    render(prefill_symbol=prefill_symbol)
