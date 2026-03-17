"""
Trading strategy — agent modifies this file.
Exports: strategy(df) -> pd.Series of signals (+1 long, -1 short, 0 flat)
"""

import pandas as pd
import numpy as np


def strategy(df: pd.DataFrame) -> pd.Series:
    """Long-only momentum + ADX/DI + BB + RSI dip-buying.

    Core: ROC(20) momentum with SMA50 trend + ADX/DI filter.
    BB: Oversold bounce entries in uptrend.
    RSI: Additional dip-buying when RSI < 40 in uptrend (not deeply oversold).
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Trend filter
    sma50 = close.rolling(50).mean()
    trend_up = close > sma50

    # Momentum
    roc = close.pct_change(20)

    # RSI(14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # Bollinger Bands (20, 2)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
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

    signals = pd.Series(0, index=df.index)

    # Primary: Long momentum with ADX/DI confirmation
    signals[trend_up & (roc > 0) & strong_trend & di_bullish] = 1

    # BB oversold bounce
    signals[trend_up & (close < bb_lower)] = 1

    # RSI dip-buy: in uptrend, RSI pulls back to 30-40 zone
    signals[trend_up & (rsi < 40) & (rsi > 25) & di_bullish] = 1

    return signals


# ---------------------------------------------------------------------------
# Runner — imports prepare.py, backtests all assets, prints metrics
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from prepare import load_all_assets, run_backtest, print_metrics

    assets = load_all_assets()
    metrics = run_backtest(strategy, assets)
    print_metrics(metrics)
