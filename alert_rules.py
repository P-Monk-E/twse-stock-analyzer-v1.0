import random

def stock_pass(stock):
    return (stock["流動比率"] >= 1.25 and stock["ROE"] >= 0.08 and 
            stock["Alpha"] > 0 and stock["Sharpe Ratio"] >= 1)

def etf_pass(stock):
    return stock["Alpha"] > 0 and stock["Sharpe Ratio"] >= 1

def get_recommendations(tickers, market_close, rf, start, end, is_etf=False):
    from stock_utils import get_metrics
    candidates = []
    for t in tickers:
        m = get_metrics(t, market_close, rf, start, end)
        if m:
            if is_etf and etf_pass(m):
                candidates.append((t,m))
            elif not is_etf and stock_pass(m):
                candidates.append((t,m))
    return random.sample(candidates, 10) if len(candidates) >= 10 else candidates
