import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from portfolio_risk_utils import diversification_warning

def _normalize_tw_ticker(ticker: str) -> list[str]:
    t = ticker.upper().strip()
    if t.endswith((".TW", ".TWO")):
        return [t]
    return [f"{t}.TW", f"{t}.TWO"]

def _latest_close(symbol: str) -> Optional[float]:
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

def set_portfolio_risk_warning(
    sharpe: Optional[float],
    treynor: Optional[float],
    *,
    non_sys_thr: float = 0.5,
    sys_thr: float = 0.5,
) -> None:
    """
    為與頁面滑桿一致，此函式接受門檻並將訊息寫入 session_state。
    """
    st.session_state["portfolio_risk_warning"] = diversification_warning(
        sharpe if sharpe is not None else None,
        treynor if treynor is not None else None,
        non_sys_thr=non_sys_thr,
        sys_thr=sys_thr,
    )

def estimate_portfolio_risk(positions: List[Dict]) -> Tuple[Optional[float], Optional[float]]:
    """
    以近一年資料估算組合 Sharpe/Treynor：
    - 權重 = 最新市值權重（不可取得則均權）
    - 市場：^TWII
    - rf：固定 1%
    失敗回傳 (None, None)
    """
    try:
        if not positions:
            return None, None

        weights = []
        symbols = []
        for p in positions:
            sym = str(p.get("symbol") or p.get("ticker"))
            qty = float(p.get("qty") or p.get("shares") or 0)
            if not sym or qty <= 0:
                continue
            px = _latest_close(sym)
            mv = (px or 0.0) * qty
            symbols.append(sym)
            weights.append(mv)
        if not symbols:
            return None, None
        w = np.array(weights, dtype=float)
        if not np.isfinite(w).any() or w.sum() <= 0:
            w = np.ones_like(w)
        w = w / w.sum()

        end = pd.Timestamp.today(tz="UTC").normalize()
        start = end - pd.Timedelta(days=365)
        px_map = {}
        for sym in symbols:
            hist = None
            for cand in _normalize_tw_ticker(sym):
                try:
                    h = yf.Ticker(cand).history(start=start, end=end)
                    if not h.empty:
                        hist = h
                        break
                except Exception:
                    continue
            if hist is None or hist.empty:
                continue
            px_map[sym] = hist["Close"].rename(sym)

        if not px_map:
            return None, None

        prices = pd.concat(px_map.values(), axis=1).dropna(how="any")
        if prices.empty:
            return None, None

        rets = prices.pct_change().dropna()
        port_ret = (rets * w).sum(axis=1)

        mkt = yf.Ticker("^TWII").history(start=start, end=end)["Close"].pct_change().dropna()
        df = pd.concat([port_ret, mkt], axis=1).dropna()
        df.columns = ["p", "m"]

        if df.empty or df["p"].std() == 0:
            return None, None

        rf = 0.01
        sharpe = float(((df["p"] - rf / 252).mean() / df["p"].std()) * math.sqrt(252))

        mon = df.resample("M").last()
        if len(mon) < 6 or mon["m"].std() == 0:
            treynor = None
        else:
            cov = float(np.cov(mon["p"], mon["m"])[0, 1])
            var_m = float(np.var(mon["m"]))
            beta = cov / var_m if var_m != 0 else np.nan
            if not np.isfinite(beta) or abs(beta) < 1e-6:
                treynor = None
            else:
                ann_ret = float(df["p"].mean()) * 252.0
                treynor = (ann_ret - rf) / beta

        return sharpe, (treynor if treynor is not None else None)
    except Exception:
        return None, None
