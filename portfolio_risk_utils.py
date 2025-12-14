def diversification_warning(sharpe: float, treynor: float) -> str | None:
    if sharpe is None or treynor is None:
        return None

    diff = treynor - sharpe

    if diff > 0.5:
        return "⚠ 高非系統性風險！Treynor 明顯高於 Sharpe，代表投資組合分散不足，存在可避免風險。"

    if diff < -0.5:
        return "⚠ 高系統性風險！Sharpe 高於 Treynor，代表市場曝險過高。"

    return None
