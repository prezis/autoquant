"""
Trading strategy — agent modifies this file.
Exports: strategy(df) -> pd.Series of signals (+1 long, -1 short, 0 flat)
"""

import pandas as pd
import numpy as np


def strategy(df: pd.DataFrame) -> pd.Series:
    """Momentum + BB + ADX/DI with EMA trend filter.

    Change: EMA50 instead of SMA50 for faster trend response.
    Core: ROC(20) momentum + ADX/DI directional filter.
    BB mean reversion for range-bound markets.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Trend filter: EMA50 (faster than SMA)
    ema50 = close.ewm(span=50, adjust=False).mean()
    trend_up = close > ema50
    trend_down = close < ema50

    # Momentum
    roc = close.pct_change(20)

    # Bollinger Bands (20, 2)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    # ADX(14) with DI
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
    di_bullish = plus_di > minus_di
    di_bearish = minus_di > plus_di

    signals = pd.Series(0, index=df.index)

    # Momentum signals with ADX + DI confirmation
    signals[trend_up & (roc > 0) & strong_trend & di_bullish] = 1
    signals[trend_down & (roc < 0) & strong_trend & di_bearish] = -1

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
