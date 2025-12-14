from typing import Optional

def diversification_warning(
    sharpe: Optional[float],
    treynor: Optional[float],
    *,
    non_sys_thr: float = 0.5,  # Treynor - Sharpe > non_sys_thr => 非系統性風險
    sys_thr: float = 0.5,      # Treynor - Sharpe < -sys_thr   => 系統性風險
) -> Optional[str]:
    """
    可調門檻版。
    為避免誤報，保留預設 0.5；頁面可覆寫。
    """
    if sharpe is None or treynor is None:
        return None
    diff = treynor - sharpe
    # 大幅正差：分散不足（非系統性風險）
    if diff > float(non_sys_thr):
        return "⚠ 高非系統性風險！Treynor 明顯高於 Sharpe，代表投資組合分散不足，存在可避免風險。"
    # 大幅負差：市場曝險過高（系統性風險）
    if diff < -float(sys_thr):
        return "⚠ 高系統性風險！Sharpe 明顯高於 Treynor，代表市場曝險過高。"
    return None
