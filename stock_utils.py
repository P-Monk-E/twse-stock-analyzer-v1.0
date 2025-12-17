# /mnt/data/stock_utils.py
from __future__ import annotations

import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm

# ---------- 基本設定 ----------
TICKER_NAME_MAP = {
    "2330": "台積電",
    "2454": "聯發科",
    "2303": "聯電",
    "2317": "鴻海",
    "2881": "富邦金",
    "2882": "國泰金",
    "2603": "長榮",
    "2618": "長榮航",
    "1216": "統一",
    "2383": "台光電",
    "2313": "華通",
    "2211": "長榮鋼",
    "2451": "創見",
    "2377": "微星",
    "2379": "瑞昱",
    "1303": "南亞",
    "2615": "萬海",
    "0050": "元大台灣50",
    "0056": "元大高股息",
    "006208": "富邦台50",
    "00980A": "台新永續高息",
}

# ---------- 代碼工具 ----------
_ETF_RX = re.compile(r"^(00\d{2,}|009\d{2}[A-Z]?)$")

def is_etf(code: str) -> bool:
    return bool(_ETF_RX.match(str(code).upper()))

def find_ticker_by_name(q: str) -> Optional[str]:
    s = str(q).strip()
    if not s:
        return None
    if s.isdigit() or s.upper().endswith((".TW", ".TWO")):
        return s.upper().removesuffix(".TW").removesuffix(".TWO")
    # 中文名稱對照
    for k, v in TICKER_NAME_MAP.items():
        if v == s:
            return k
    return None

# ---------- 價價序列 ----------
def fetch_price_data(code: str, start, end) -> pd.DataFrame | None:
    """
    強化下載穩定性：
    1) 嘗試 {code}.TW 與 {code}.TWO
    2) 先用 start/end，失敗再退回 period="3y"
    3) 回傳第一個非空 DataFrame，否則 None
    """
    cands = [code] if code.endswith(('.TW', '.TWO')) else [f"{code}.TW", f"{code}.TWO"]
    for c in cands:
        try:
            t = yf.Ticker(c)
            df = t.history(start=start, end=end)
            if df is not None and not df.empty:
                return df
            # fallback: Yahoo 有時對 start/end 失敗；改用 period
            df = t.history(period="3y")
            if df is not None and not df.empty:
                return df
        except Exception:
            continue
    return None

# ---------- 財報工具 ----------
def _first_non_nan(values) -> Optional[float]:
    try:
        s = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
        return float(s.iloc[0]) if not s.empty else None
    except Exception:
        return None

def _find_row(df: pd.DataFrame, patterns: list[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    for p in patterns:
        hits = [c for c in df.index if re.search(p, str(c), re.I)]
        if hits:
            return hits[0]
    return None

def _annualize_return(close: pd.Series) -> float:
    r = close.pct_change().dropna()
    if r.empty:
        return np.nan
    return float((1 + r.mean()) ** 252 - 1)

def _calc_alpha(close: pd.Series, market_close: pd.Series, rf: float) -> float:
    try:
        market_r = market_close.pct_change().dropna()
        stock_r = close.pct_change().dropna()
        df = pd.DataFrame({"y": stock_r, "x": market_r}).dropna()
        if df.empty:
            return np.nan
        X = sm.add_constant(df["x"])
        model = sm.OLS(df["y"] - rf / 252, X).fit()
        alpha_daily = model.params["const"]
        return float((1 + alpha_daily) ** 252 - 1)
    except Exception:
        return np.nan

def _calc_beta(close: pd.Series, market_close: pd.Series) -> float:
    try:
        market_r = market_close.pct_change().dropna()
        stock_r = close.pct_change().dropna()
        cov = np.cov(stock_r.align(market_r, join="inner"))
        if cov.shape != (2, 2) or np.isnan(cov).any():
            return np.nan
        return float(cov[0, 1] / cov[1, 1])
    except Exception:
        return np.nan

def _calc_sharpe(close: pd.Series, rf: float) -> float:
    try:
        r = close.pct_change().dropna()
        if r.empty:
            return np.nan
        excess = r - rf / 252
        sr = excess.mean() / excess.std()
        return float(sr * np.sqrt(252))
    except Exception:
        return np.nan

def _calc_treynor(close: pd.Series, rf: float, beta: float) -> float:
    try:
        if beta is None or np.isnan(beta) or beta == 0:
            return np.nan
        r = close.pct_change().dropna()
        if r.empty:
            return np.nan
        excess = r - rf / 252
        er = float((1 + excess.mean()) ** 252 - 1)
        return float(er / beta)
    except Exception:
        return np.nan

def _calc_madr(close: pd.Series) -> float:
    try:
        r = close.pct_change().dropna()
        if r.empty:
            return np.nan
        return float(np.abs(r).mean())
    except Exception:
        return np.nan

# ---------- 主函數 ----------
def get_metrics(code: str, market_close: pd.Series, rf: float, start, end, is_etf: bool = False):
    df = fetch_price_data(code, start, end)
    if df is None or df.empty:
        return None

    close = df["Close"]
    alpha = _calc_alpha(close, market_close, rf)
    beta = _calc_beta(close, market_close)
    sharpe = _calc_sharpe(close, rf)
    treynor = _calc_treynor(close, rf, beta)
    madr = _calc_madr(close)

    # ---- 財報/配息：盡量取到資料（.TW/.TWO 任一能用即可）----
    debt_equity = np.nan
    current_ratio = np.nan
    roe = np.nan
    equity_val = np.nan
    eps_ttm = np.nan

    try:
        # 取得可用的 Ticker 物件（.TW / .TWO 皆嘗試）
        t = None
        for _cand in ([code] if code.endswith(('.TW', '.TWO')) else [f"{code}.TW", f"{code}.TWO"]):
            try:
                tmp = yf.Ticker(_cand)
                # 驗證：有基本欄位/或歷史資料不空即可
                _ = tmp.fast_info
                t = tmp
                break
            except Exception:
                try:
                    if not yf.Ticker(_cand).history(period="1mo").empty:
                        t = yf.Ticker(_cand)
                        break
                except Exception:
                    pass
                continue
        if t is None:
            # 無法取得財報資料就保留 NaN，後續用價量KPI
            t = yf.Ticker(f"{code}.TW")  # 最後一次嘗試（避免下方引用錯誤）

        # 年/季報
        bs_year = t.balance_sheet
        bs_quarter = t.quarterly_balance_sheet
        fs_year = t.financials
        fs_quarter = t.quarterly_financials

        # 常用指標
        # 負債權益比 = TotalLiab / TotalStockholderEquity
        tr = _find_row(bs_year, [r"(?i)total liab", r"負債"])
        er = _find_row(bs_year, [r"(?i)total stockholder", r"股東權益"])
        if tr and er:
            try:
                debt_equity = float(bs_year.loc[tr].iloc[0] / bs_year.loc[er].iloc[0])
            except Exception:
                pass

        # 流動比率 = CurrentAssets / CurrentLiabilities（改用季報較即時）
        ca = _find_row(bs_quarter, [r"(?i)total current assets", r"流動資產"])
        cl = _find_row(bs_quarter, [r"(?i)total current liab", r"流動負債"])
        if ca and cl:
            try:
                current_ratio = float(bs_quarter.loc[ca].iloc[0] / bs_quarter.loc[cl].iloc[0])
            except Exception:
                pass

        # ROE（用季報推 TTM）
        ni = _find_row(fs_quarter, [r"(?i)net income", r"淨利"])
        eq = _find_row(bs_quarter, [r"(?i)total stockholder", r"股東權益"])
        if ni and eq:
            try:
                roe = float((fs_quarter.loc[ni].iloc[:4].sum()) / bs_quarter.loc[eq].iloc[0])
            except Exception:
                pass

        # 股東權益（年報）
        if er:
            try:
                equity_val = float(bs_year.loc[er].iloc[0])
            except Exception:
                pass

        # EPS_TTM：股票=淨利TTM / 股數；ETF=近四次現金股利合計
        try:
            if is_etf:
                cash = t.dividends
                eps_ttm = float(cash.iloc[-4:].sum()) if cash is not None and not cash.empty else np.nan
            else:
                shares = t.get_shares_full(start=start, end=end)
                shares = float(shares.iloc[-1]) if shares is not None and not shares.empty else np.nan
                if ni:
                    net_ttm = float(fs_quarter.loc[ni].iloc[:4].sum())
                    eps_ttm = float(net_ttm / shares) if shares and not np.isnan(shares) else np.nan
        except Exception:
            pass

    except Exception:
        # 財報失敗不影響技術指標
        pass

    # 警告摘要（可由外部頁面決定如何展示）
    warnings = []
    if np.isnan(alpha) or np.isnan(sharpe):
        warnings.append("統計樣本不足")
    if not is_etf and (not np.isnan(roe) and roe < 0.08):
        warnings.append("ROE 偏低")
    if not np.isnan(madr) and madr > 0.03:
        warnings.append("波動偏高")

    return {
        "name": TICKER_NAME_MAP.get(code, ""),
        "負債權益比": debt_equity,
        "流動比率": current_ratio,
        "ROE": roe,
        "Equity": equity_val,
        "EPS_TTM": eps_ttm,   # 股票=淨利TTM/股；ETF=配息TTM
        "Alpha": alpha,
        "Beta": beta,
        "Sharpe Ratio": sharpe,
        "Treynor": treynor,
        "MADR": madr,
        "警告": warnings,
        "df": df,
    }
