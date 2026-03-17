"""
Trading strategy — agent modifies this file.
Exports: strategy(df) -> pd.Series of signals (+1 long, -1 short, 0 flat)
"""

import pandas as pd
import numpy as np


def strategy(df: pd.DataFrame) -> pd.Series:
    """Momentum + BB + ADX + Williams %R entry timing.

    Core: ROC(20) momentum with SMA50 trend + ADX strength filter.
    Enhancement: Williams %R for better entry timing.
    BB mean reversion for range-bound markets.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Trend filter
    sma50 = close.rolling(50).mean()
    trend_up = close > sma50
    trend_down = close < sma50

    # Momentum
    roc = close.pct_change(20)

    # Williams %R(14) - overbought/oversold oscillator
    hh = high.rolling(14).max()
    ll = low.rolling(14).min()
    willr = -100 * (hh - close) / (hh - ll).replace(0, np.nan)

    # Bollinger Bands (20, 2)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    # ADX(14)
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.rolling(14).mean()

    strong_trend = adx > 20

    signals = pd.Series(0, index=df.index)

    # Momentum + ADX signals, enhanced with Williams %R timing
    # Long: uptrend + positive momentum + strong trend + not overbought
    signals[trend_up & (roc > 0) & strong_trend & (willr < -20)] = 1
    # Short: downtrend + negative momentum + strong trend + not oversold
    signals[trend_down & (roc < 0) & strong_trend & (willr > -80)] = -1

    # BB mean reversion
    signals[trend_up & (close < bb_lower)] = 1
    signals[trend_down & (close > bb_upper)] = -1

    return signals


# ---------------------------------------------------------------------------
# Runner — imports prepare.py, backtests all assets, prints metrics
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from prepare import load_all_assets, run_backtest, print_metrics

    assets = load_all_assets()
    metrics = run_backtest(strategy, assets)
    print_metrics(metrics)
