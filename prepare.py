# prepare.py — Data loading, indicators, Fib levels, backtest engine, scoring
# ═══════════════════════════════════════════════════════════════════════════
# READ-ONLY: The autonomous agent must NOT modify this file.
# ═══════════════════════════════════════════════════════════════════════════

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GROUND_TRUTH_DIR = Path("data/ground_truth")
CACHE_DIR = Path("data/cache")

# ── Symbol mapping for yfinance ──────────────────────────────────────
YF_MAP = {
    "BTCUSDT": "BTC-USD", "BTCUSD": "BTC-USD", "ZECUSDT": "ZEC-USD",
    "MONUSD": "XMR-USD", "RENDERUSDT": "RNDR-USD", "MARA": "MARA",
    "UKOIL": "BZ=F", "HYPEUSDT": "HYPE-USD", "TAOUSDT": "TAO-USD",
    "ENJUSDT": "ENJ-USD", "TRIAUSDT": "TRIA-USD", "SILVER": "SI=F",
}

TF_MAP = {
    "1D": "1d", "12h": "1d", "1W": "1wk", "4h": "1h",
    "3D": "1d", "1h": "1h", "1m": "1m",
}

# Train/holdout split
TRAIN_ASSETS = ["BTCUSDT", "BTCUSD", "ZECUSDT", "MONUSD", "RENDERUSDT", "MARA", "UKOIL", "TAOUSDT"]
HOLDOUT_ASSETS = ["HYPEUSDT", "ENJUSDT"]


# ── Data Loading ─────────────────────────────────────────────────────

def load_asset_ohlcv(symbol: str, timeframe: str, cache_dir: str = "data/cache") -> pd.DataFrame:
    """Load OHLCV data for an asset. Tries cache first, then yfinance."""
    cache_path = Path(cache_dir) / f"{symbol}_{timeframe}.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        if len(df) > 50:
            return df

    import yfinance as yf
    ticker = YF_MAP.get(symbol, symbol)
    interval = TF_MAP.get(timeframe, "1d")
    period = "5y" if interval in ("1d", "1wk") else "2y"
    raw = yf.download(ticker, period=period, interval=interval, progress=False)
    if raw.empty:
        raise ValueError(f"No data for {symbol} ({ticker})")

    df = raw.rename(columns=str.lower)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['open', 'high', 'low', 'close', 'volume']].copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.to_parquet(cache_path)
    return df


# ── Indicator Computation ────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to OHLCV DataFrame."""
    c = df['close']
    h, l, v = df['high'], df['low'], df['volume']

    # RSI(14)
    delta = c.diff()
    gain = delta.where(delta > 0, 0.0).ewm(span=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(span=14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))

    # Stochastic %K/%D(14,3)
    low_14 = l.rolling(14).min()
    high_14 = h.rolling(14).max()
    df['stoch_k'] = 100 * (c - low_14) / (high_14 - low_14).replace(0, np.nan)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # MFI(14)
    tp = (h + l + c) / 3
    mf = tp * v
    pos_mf = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
    neg_mf = mf.where(tp <= tp.shift(1), 0).rolling(14).sum()
    mr = pos_mf / neg_mf.replace(0, np.nan)
    df['mfi'] = 100 - (100 / (1 + mr))

    # EMAs
    for span in [33, 66, 144, 288]:
        df[f'ema_{span}'] = c.ewm(span=span, adjust=False).mean()

    # ATR(14)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    # Volume SMA ratio
    df['vol_ratio'] = v / v.rolling(20).mean().replace(0, np.nan)

    # ADX(14)
    plus_dm = h.diff().where(lambda x: (x > 0) & (x > -l.diff()), 0.0)
    minus_dm = (-l.diff()).where(lambda x: (x > 0) & (x > h.diff()), 0.0)
    atr_smooth = tr.ewm(span=14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr_smooth.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr_smooth.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df['adx'] = dx.ewm(span=14, adjust=False).mean()

    # SuperTrend(10, 3)
    hl2 = (h + l) / 2
    atr_st = tr.rolling(10).mean()
    upper = hl2 + 3 * atr_st
    lower = hl2 - 3 * atr_st
    st = pd.Series(0.0, index=df.index)
    direction = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        if c.iloc[i] > upper.iloc[i - 1]:
            direction.iloc[i] = 1
        elif c.iloc[i] < lower.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]
        st.iloc[i] = lower.iloc[i] if direction.iloc[i] == 1 else upper.iloc[i]
    df['supertrend'] = direction
    df['supertrend_level'] = st

    return df


# ── Fibonacci Levels ─────────────────────────────────────────────────

def compute_fib_levels(swing_low: float, swing_high: float) -> dict:
    """Compute standard + Popek Fib levels from swing points."""
    rng = swing_high - swing_low
    levels = {}
    for ratio in [0.0, 0.236, 0.382, 0.45, 0.5, 0.577, 0.618, 0.667, 0.705, 0.786, 0.875, 1.0]:
        levels[str(ratio)] = round(swing_low + ratio * rng, 6)
    return levels


# ── Backtest Engine ──────────────────────────────────────────────────

def backtest(df: pd.DataFrame, signals: pd.Series,
             commission: float = 0.001, slippage: float = 0.0003) -> dict:
    """Vectorized backtest. signals: +1 long, -1 short, 0 flat."""
    returns = df['close'].pct_change().fillna(0)
    # Shift signals by 1 to avoid lookahead (signal at bar N -> position at bar N+1)
    position = signals.shift(1).fillna(0)
    # Transaction costs on position changes
    trades = position.diff().fillna(0).abs()
    costs = trades * (commission + slippage)
    # Strategy returns
    strat_returns = position * returns - costs
    equity = (1 + strat_returns).cumprod()

    # Metrics
    sharpe = _sharpe(strat_returns)
    sortino = _sortino(strat_returns)
    dd = float((equity / equity.cummax() - 1).min())
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    num_trades = int((trades > 0).sum())
    win_rate = _win_rate(strat_returns, position)

    return {
        'equity': equity,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': dd,
        'total_return': total_return,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'val_return': total_return,
    }


def _sharpe(returns: pd.Series, periods: int = 252) -> float:
    std = returns.std()
    if std == 0 or pd.isna(std):
        return 0.0
    return float(returns.mean() / std * np.sqrt(periods))


def _sortino(returns: pd.Series, periods: int = 252) -> float:
    downside = returns[returns < 0]
    if len(downside) == 0:
        return 0.0
    std = downside.std()
    if std == 0 or pd.isna(std):
        return 0.0
    return float(returns.mean() / std * np.sqrt(periods))


def _win_rate(returns: pd.Series, position: pd.Series) -> float:
    active = returns[position != 0]
    if len(active) == 0:
        return 0.0
    return float((active > 0).sum() / len(active))


# ── Scoring ──────────────────────────────────────────────────────────

def compute_score(result: dict, cross_asset_consistency: float,
                  train_val_ratio: float) -> float:
    """Composite score with hard reject constraints.

    score = (0.40 * sortino + 0.25 * DD_control + 0.20 * consistency + 0.15 * return)
            x trade_penalty x overfit_penalty
    """
    # Hard rejects
    if result['max_drawdown'] < -0.20:
        return 0.0
    if result['num_trades'] < 30:
        return 0.0
    if train_val_ratio > 2.0:
        return 0.0

    sortino = max(result['sortino'], 0)
    dd_score = 1 - abs(result['max_drawdown'])
    consistency = max(cross_asset_consistency, 0)
    val_ret = max(result.get('val_return', 0), 0)

    raw = (0.40 * min(sortino / 3.0, 1.0)
         + 0.25 * dd_score
         + 0.20 * consistency
         + 0.15 * min(val_ret, 1.0))

    # Trade penalty: ramp from 0 at 0 trades to 1 at 30+ trades
    trade_penalty = min(result['num_trades'] / 30.0, 1.0)

    # Overfitting penalty
    overfit_penalty = 1.0 if train_val_ratio <= 1.5 else max(0, 1 - (train_val_ratio - 1.5))

    return round(raw * trade_penalty * overfit_penalty, 6)


# ── Ground Truth Loading ─────────────────────────────────────────────

def load_ground_truth(symbol: str) -> dict:
    """Load ground truth JSON for an asset. Tries exact match, then prefix match."""
    for suffix in ['_1D', '_12h', '_1W', '_3D', '_4h', '_1D_1D']:
        p = GROUND_TRUTH_DIR / f"{symbol}{suffix}.json"
        if p.exists():
            return json.loads(p.read_text())
    # Prefix match
    for p in sorted(GROUND_TRUTH_DIR.glob(f"{symbol}_*.json")):
        return json.loads(p.read_text())
    raise FileNotFoundError(f"No ground truth for {symbol}")


# ── Full Evaluation Pipeline ─────────────────────────────────────────

def evaluate_strategy(strategy_fn) -> dict:
    """Run strategy across all train assets, compute aggregate score."""
    train_results = []
    asset_details = []

    for symbol in TRAIN_ASSETS:
        try:
            gt = load_ground_truth(symbol)
            tf = gt.get('timeframe', '1D')
            df = load_asset_ohlcv(symbol, tf)
            df = compute_indicators(df)
            fib = compute_fib_levels(gt['swing_low'], gt['swing_high'])
            signals = strategy_fn(df, fib)
            result = backtest(df, signals)
            train_results.append(result)
            asset_details.append({
                'symbol': symbol, 'return': result['val_return'],
                'sharpe': result['sharpe'], 'max_dd': result['max_drawdown'],
                'trades': result['num_trades'],
            })
        except Exception as e:
            print(f"  SKIP {symbol}: {e}", file=sys.stderr)

    if not train_results:
        return {'score': 0.0, 'detail': 'no assets evaluated'}

    # Aggregate
    avg_sortino = float(np.mean([r['sortino'] for r in train_results]))
    worst_dd = float(min(r['max_drawdown'] for r in train_results))
    avg_return = float(np.mean([r['val_return'] for r in train_results]))
    total_trades = sum(r['num_trades'] for r in train_results)
    profitable = sum(1 for r in train_results if r['val_return'] > 0)
    consistency = profitable / len(train_results)

    agg = {
        'sortino': avg_sortino, 'max_drawdown': worst_dd,
        'val_return': avg_return, 'num_trades': total_trades,
        'sharpe': float(np.mean([r['sharpe'] for r in train_results])),
    }
    score = compute_score(agg, consistency, train_val_ratio=1.0)

    # Holdout check (informational, not used in optimization)
    holdout_results = []
    for symbol in HOLDOUT_ASSETS:
        try:
            gt = load_ground_truth(symbol)
            tf = gt.get('timeframe', '1D')
            df = load_asset_ohlcv(symbol, tf)
            df = compute_indicators(df)
            fib = compute_fib_levels(gt['swing_low'], gt['swing_high'])
            signals = strategy_fn(df, fib)
            result = backtest(df, signals)
            holdout_results.append(result)
        except Exception:
            pass

    holdout_sharpe = float(np.mean([r['sharpe'] for r in holdout_results])) if holdout_results else 0.0

    return {
        'score': score,
        'sortino': avg_sortino,
        'max_drawdown': worst_dd,
        'val_return': avg_return,
        'trades': total_trades,
        'cross_asset_consistency': consistency,
        'holdout_sharpe': holdout_sharpe,
        'assets_evaluated': len(train_results),
        'asset_details': asset_details,
    }
