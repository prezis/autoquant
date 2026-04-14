# strategy.py — Fibonacci zone entry strategy
# ═══════════════════════════════════════════
# The autonomous agent modifies ONLY this file.
# Run: python strategy.py
# Metric: score (higher = better). Read results from stdout.
# ═══════════════════════════════════════════

import pandas as pd
import numpy as np

DESCRIPTION = "Baseline: RSI + golden zone (0.5-0.618) entry, premium zone (0.786+) short"


def strategy(df: pd.DataFrame, fib_levels: dict) -> pd.Series:
    """Generate trading signals based on Fibonacci zones + RSI.

    Args:
        df: OHLCV DataFrame with indicators (rsi, stoch_k, mfi, ema_*, atr, adx, supertrend, vol_ratio)
        fib_levels: dict mapping level names to prices ('0.618' -> 44.63)

    Returns:
        pd.Series of signals: +1 (long), -1 (short), 0 (flat)
    """
    signals = pd.Series(0, index=df.index, dtype=int)
    price = df['close']
    rsi = df['rsi']

    zone_low = fib_levels['0.5']
    zone_high = fib_levels['0.618']
    premium = fib_levels['0.786']

    for i in range(1, len(df)):
        p = price.iloc[i]
        r = rsi.iloc[i]

        if pd.isna(r):
            continue

        # Long: price in golden zone (0.5-0.618) + RSI oversold
        if zone_low <= p <= zone_high and r < 35:
            signals.iloc[i] = 1

        # Short: price in premium zone (above 0.786) + RSI overbought
        elif p >= premium and r > 70:
            signals.iloc[i] = -1

    return signals


# ── Runner (executed by agent) ───────────────────────────────────────
if __name__ == "__main__":
    from prepare import evaluate_strategy

    result = evaluate_strategy(strategy)

    # Output format that orchestrator.py parses
    print(f"score:        {result['score']:.4f}")
    print(f"sortino:      {result['sortino']:.4f}")
    print(f"max_drawdown: {result['max_drawdown']:.4f}")
    print(f"val_return:   {result['val_return']:.4f}")
    print(f"trades:       {result['trades']}")
    print(f"consistency:  {result['cross_asset_consistency']:.4f}")
    print(f"holdout:      {result['holdout_sharpe']:.4f}")
    print(f"assets:       {result['assets_evaluated']}")

    # Per-asset breakdown
    if 'asset_details' in result:
        print("\n--- Per-asset breakdown ---")
        for a in result['asset_details']:
            print(f"  {a['symbol']:12s}  ret={a['return']:+.4f}  sharpe={a['sharpe']:.2f}  dd={a['max_dd']:.4f}  trades={a['trades']}")
