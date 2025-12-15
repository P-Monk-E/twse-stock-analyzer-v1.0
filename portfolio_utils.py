from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

from portfolio_risk_utils import diversification_warning

# ---------- 內部工具 ----------
def _normalize_tw_ticker(ticker: str) -> list[str]:
    t = str(ticker).upper().strip()
    if t.endswith((".TW", ".TWO")):
        return [t]
    return [f"{t}.TW", f"{t}.TWO"]

def _latest_close(symbol: str) -> Optional[float]:
    """抓最新價作為市值權重；抓不到回 None。"""
    for cand in _normalize_tw_ticker(symbol):
        try:
            p = yf.Ticker(cand).fast_info.get("lastPrice")
            if p:
                return float(p)
        except Exception:
            pass
        try:
            hist = yf.Ticker(cand).history(period="1d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception:
            continue
    return None

def _fetch_close_series(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.Series]:
    """回傳 Close 價格序列（name=原輸入符號），抓不到回 None。"""
    for cand in _normalize_tw_ticker(symbol):
        try:
            h = yf.Ticker(cand).history(start=start, end=end)
            if h is not None and not h.empty and "Close" in h:
                s = h["Close"].copy()
                s.name = symbol
                return s
        except Exception:
            continue
    return None

def _get_market_returns(start: pd.Timestamp, end: pd.Timestamp) -> Tuple[Optional[pd.Series], Optional[str]]:
    """
    取得市場日報酬，依序嘗試多個指數；回 (returns, 使用的代號)。
    """
    for idx in ["^TWII", "^TAIEX", "^GSPC"]:
        try:
            h = yf.Ticker(idx).history(start=start, end=end)
            if h is not None and not h.empty and "Close" in h:
                r = h["Close"].pct_change().dropna()
                if not r.empty:
                    r.name = idx
                    return r, idx
        except Exception:
            continue
    return None, None

def set_portfolio_risk_warning(
    sharpe: Optional[float],
    treynor: Optional[float],
    *,
    non_sys_thr: float = 0.5,
    sys_thr: float = 0.5,
) -> None:
    """將訊息寫入 session_state，供頁首顯示。"""
    st.session_state["portfolio_risk_warning"] = diversification_warning(
        sharpe if sharpe is not None else None,
        treynor if treynor is not None else None,
        non_sys_thr=non_sys_thr,
        sys_thr=sys_thr,
    )

# ---------- 核心：估算組合風險 ----------
def estimate_portfolio_risk(positions: List[Dict]) -> Tuple[Optional[float], Optional[float], str]:
    """
    以近一年資料估算組合 Sharpe/Treynor。
    回傳: (sharpe, treynor, debug_message)
    - 權重 = 最新市值權重（抓不到價則均權）
    - 市場：^TWII → ^TAIEX → ^GSPC（備援）
    - rf：固定 1%
    - 若資料不足，sharpe/treynor 為 None，debug_message 說明原因
    """
    try:
        if not positions:
            return None, None, "沒有持股資料。"

        # ---- 準備權重（去重複符號）----
        weights = []
        symbols = []
        for p in positions:
            sym = str(p.get("symbol") or p.get("ticker") or "").strip()
            qty = float(p.get("qty") or p.get("shares") or 0)
            if not sym or qty <= 0:
                continue
            price = _latest_close(sym)
            mv = (price or 0.0) * qty
            symbols.append(sym)
            weights.append(mv)

        if not symbols:
            return None, None, "無有效持股（代碼或股數為 0）。"

        # 去重複：同一代碼只保留一次（權重相加）
        dedup: Dict[str, float] = {}
        for sym, w in zip(symbols, weights):
            dedup[sym] = dedup.get(sym, 0.0) + (w or 0.0)
        symbols = list(dedup.keys())
        w = np.array([dedup[s] for s in symbols], dtype=float)
        if not np.isfinite(w).any() or w.sum() <= 0:
            w = np.ones_like(w)
        w = w / w.sum()

        # ---- 下載近一年價格 ----
        end = pd.Timestamp.today(tz="UTC").normalize()
        start = end - pd.Timedelta(days=365)

        px_map = {}
        for sym in symbols:
            s = _fetch_close_series(sym, start, end)
            if s is not None:
                px_map[sym] = s

        if not px_map:
            return None, None, "下載個股/ETF 價格失敗。"

        prices = pd.concat(px_map.values(), axis=1, join="inner").dropna(how="any")
        if prices.empty:
            return None, None, "沒有共同交易日可用。"

        # 樣本數保護（避免過少造成不穩定）
        if len(prices) < 60:
            return None, None, f"樣本不足（僅 {len(prices)} 筆日資料）。"

        # ---- 組合報酬 ----
        rets = prices.pct_change().dropna()
        if rets.empty:
            return None, None, "無法計算日報酬。"
        port_ret = (rets * w).sum(axis=1)

        # ---- 市場報酬（多重備援）----
        mkt_ret, mkt_used = _get_market_returns(start, end)
        if mkt_ret is None:
            return None, None, "取不到市場指數 (^TWII/^TAIEX/^GSPC)。"

        df = pd.concat([port_ret, mkt_ret], axis=1, join="inner").dropna()
        df.columns = ["p", "m"]
        if df.empty or df["p"].std() == 0:
            return None, None, "組合日報酬樣本不足或為常數。"

        # ---- 指標 ----
        rf = 0.01
        sharpe = float(((df["p"] - rf / 252).mean() / df["p"].std()) * math.sqrt(252))

        # Treynor：用月資料估 beta，比日資料穩定
        mon = df.resample("M").last().dropna()
        if len(mon) < 6 or mon["m"].std() == 0:
            return sharpe, None, f"月資料不足以估 beta（使用指數：{mkt_used}）。"
        cov = float(np.cov(mon["p"], mon["m"])[0, 1])
        var_m = float(np.var(mon["m"]))
        beta = cov / var_m if var_m != 0 else np.nan
        if not np.isfinite(beta) or abs(beta) < 1e-6:
            return sharpe, None, f"估不出有效 beta（使用指數：{mkt_used}）。"

        ann_ret = float(df["p"].mean()) * 252.0
        treynor = (ann_ret - rf) / beta

        return sharpe, treynor, f"使用市場指數：{mkt_used}。"
    except Exception as e:
        return None, None, f"內部錯誤：{e!s}"
