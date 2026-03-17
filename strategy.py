"""
Trading strategy — agent modifies this file.
Exports: strategy(df) -> pd.Series of signals (+1 long, -1 short, 0 flat)
"""

import pandas as pd
import numpy as np


def strategy(df: pd.DataFrame) -> pd.Series:
    """Long-only: SMA50 + ADX/DI(12) + BB + volatility regime filter.

    Core: SMA50 trend + ADX>20 + DI spread>12.
    BB for dip-buying in uptrend.
    Regime: Go flat when realized vol is extreme (> 2x median).
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Trend filter
    sma50 = close.rolling(50).mean()
    trend_up = close > sma50

    # Bollinger Bands (20, 2)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_lower = bb_mid - 2 * bb_std

    # Volatility regime: 20-day realized vol
    daily_ret = close.pct_change()
    vol20 = daily_ret.rolling(20).std()
    vol_median = vol20.rolling(252).median()  # 1-year median vol
    extreme_vol = vol20 > (vol_median * 2.0)

    # ADX(20) with DI - smoother period
    adx_period = 20
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(adx_period).mean()
    plus_di = 100 * (plus_dm.rolling(adx_period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(adx_period).mean() / atr)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.rolling(adx_period).mean()

    di_spread = plus_di - minus_di
    di_strong_bullish = di_spread > 12
    strong_trend = adx > 20

    signals = pd.Series(0, index=df.index)

    # Primary: DI spread + uptrend + ADX confirmation
    signals[trend_up & strong_trend & di_strong_bullish] = 1

    # BB oversold bounce in uptrend
    signals[trend_up & (close < bb_lower)] = 1

    # Go flat during extreme volatility
    signals[extreme_vol] = 0

    return signals


# ---------------------------------------------------------------------------
# Runner — imports prepare.py, backtests all assets, prints metrics
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from prepare import load_all_assets, run_backtest, print_metrics

    assets = load_all_assets()
    metrics = run_backtest(strategy, assets)
    print_metrics(metrics)
