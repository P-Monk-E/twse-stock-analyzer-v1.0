from typing import Tuple

def grade_sharpe(x: float) -> Tuple[str, str]:
    if x is None:
        return "❓", "資料不足"
    if x < 1:
        return "❗", "風險過高"
    if x <= 2:
        return "⚠", "普通"
    return "✅", "優秀"


def grade_treynor(x: float) -> Tuple[str, str]:
    if x is None:
        return "❓", "資料不足"
    if x < 1:
        return "❗", "風險過高"
    if x <= 2:
        return "⚠", "普通"
    return "✅", "優秀"


def grade_debt_equity(x: float) -> Tuple[str, str]:
    if x is None:
        return "❓", "資料不足"
    if x < 0.5:
        return "✅", "健康"
    if x < 1.0:
        return "⚠", "普通"
    return "❗", "債務過高風險"


def grade_current_ratio(x: float) -> Tuple[str, str]:
    if x is None:
        return "❓", "資料不足"
    if 1.5 <= x <= 2.5:
        return "✅", "健康"
    if 1.25 <= x < 1.5 or 2.5 < x <= 3.0:
        return "⚠", "普通"
    return "❗", "短期償債風險"


def grade_roe(x: float) -> Tuple[str, str]:
    if x is None:
        return "❓", "資料不足"
    if x > 0.08:
        return "✅", "優秀"
    if x >= 0.05:
        return "⚠", "普通"
    return "❗", "表現不佳"
