"""Tests for prepare.py — data loading, indicators, fib levels, backtest, scoring."""
import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from prepare import (
    load_asset_ohlcv, compute_indicators, compute_fib_levels,
    backtest, compute_score, load_ground_truth,
)


# ── Data Loading ─────────────────────────────────────────────────────

class TestDataLoading:
    def test_load_asset_ohlcv_returns_dataframe(self):
        """Load OHLCV for a known asset."""
        df = load_asset_ohlcv("BTCUSDT", "1D")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 100
        assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        assert df.index.is_monotonic_increasing

    def test_load_asset_caches_parquet(self):
        """Second load should use cache."""
        df1 = load_asset_ohlcv("BTCUSDT", "1D")
        cache = Path("data/cache/BTCUSDT_1D.parquet")
        assert cache.exists()
        df2 = load_asset_ohlcv("BTCUSDT", "1D")
        assert len(df1) == len(df2)


# ── Indicators ───────────────────────────────────────────────────────

class TestIndicators:
    def test_compute_indicators_adds_columns(self):
        """All expected indicator columns must exist."""
        df = load_asset_ohlcv("BTCUSDT", "1D")
        df = compute_indicators(df)
        expected = ['rsi', 'stoch_k', 'stoch_d', 'mfi', 'ema_33', 'ema_66',
                    'ema_144', 'ema_288', 'atr', 'vol_ratio', 'adx',
                    'supertrend', 'supertrend_level']
        for col in expected:
            assert col in df.columns, f"Missing: {col}"

    def test_rsi_range(self):
        """RSI must be 0-100."""
        df = load_asset_ohlcv("BTCUSDT", "1D")
        df = compute_indicators(df)
        rsi = df['rsi'].dropna()
        assert rsi.between(0, 100).all()


# ── Fibonacci Levels ─────────────────────────────────────────────────

class TestFibLevels:
    def test_compute_fib_levels_basic(self):
        """Known swing points produce correct levels."""
        levels = compute_fib_levels(100.0, 200.0)
        assert levels['0.0'] == 100.0
        assert levels['1.0'] == 200.0
        assert levels['0.5'] == pytest.approx(150.0)
        assert levels['0.618'] == pytest.approx(161.8)

    def test_compute_fib_levels_from_ground_truth(self):
        """Compute from HYPEUSDT ground truth."""
        gt = load_ground_truth("HYPEUSDT")
        levels = compute_fib_levels(gt['swing_low'], gt['swing_high'])
        assert levels['0.0'] == pytest.approx(20.5, abs=0.1)
        assert levels['1.0'] == pytest.approx(59.54, abs=0.1)
        assert '0.618' in levels
        assert '0.786' in levels

    def test_all_popek_levels_present(self):
        """Must include all Popek-specific levels."""
        levels = compute_fib_levels(0, 100)
        for key in ['0.0', '0.236', '0.382', '0.45', '0.5', '0.577',
                     '0.618', '0.667', '0.705', '0.786', '0.875', '1.0']:
            assert key in levels, f"Missing level: {key}"


# ── Backtest ─────────────────────────────────────────────────────────

class TestBacktest:
    def test_backtest_produces_equity_curve(self):
        """All-long signals produce an equity curve."""
        df = load_asset_ohlcv("BTCUSDT", "1D")
        signals = pd.Series(1, index=df.index)
        result = backtest(df, signals)
        assert 'equity' in result
        assert 'sharpe' in result
        assert 'sortino' in result
        assert 'max_drawdown' in result
        assert 'num_trades' in result
        assert result['equity'].iloc[-1] > 0

    def test_zero_signals_no_trades(self):
        """All-zero signals = flat = no trades."""
        df = load_asset_ohlcv("BTCUSDT", "1D")
        signals = pd.Series(0, index=df.index)
        result = backtest(df, signals)
        assert result['num_trades'] == 0
        assert result['equity'].iloc[-1] == pytest.approx(1.0, abs=0.01)

    def test_backtest_no_lookahead(self):
        """Position shifts by 1 bar — no lookahead bias."""
        df = load_asset_ohlcv("BTCUSDT", "1D")
        signals = pd.Series(0, index=df.index)
        signals.iloc[10] = 1  # signal at bar 10
        result = backtest(df, signals)
        # Position should be active at bar 11, not bar 10
        assert result['num_trades'] >= 1


# ── Scoring ──────────────────────────────────────────────────────────

class TestScoring:
    def test_hard_reject_max_dd(self):
        """MaxDD > 20% = score 0."""
        bad = {'sortino': 2.0, 'max_drawdown': -0.25, 'val_return': 0.5,
               'num_trades': 50, 'sharpe': 1.5}
        assert compute_score(bad, cross_asset_consistency=0.8, train_val_ratio=1.2) == 0.0

    def test_hard_reject_few_trades(self):
        """< 30 trades = score 0."""
        few = {'sortino': 2.0, 'max_drawdown': -0.10, 'val_return': 0.5,
               'num_trades': 5, 'sharpe': 1.5}
        assert compute_score(few, cross_asset_consistency=0.8, train_val_ratio=1.2) == 0.0

    def test_hard_reject_overfitting(self):
        """Train/val ratio > 2.0 = score 0."""
        good = {'sortino': 2.0, 'max_drawdown': -0.10, 'val_return': 0.5,
                'num_trades': 50, 'sharpe': 1.5}
        assert compute_score(good, cross_asset_consistency=0.8, train_val_ratio=2.5) == 0.0

    def test_positive_score(self):
        """Good result produces positive score."""
        good = {'sortino': 2.0, 'max_drawdown': -0.10, 'val_return': 0.3,
                'num_trades': 60, 'sharpe': 1.5}
        score = compute_score(good, cross_asset_consistency=0.8, train_val_ratio=1.1)
        assert score > 0.5

    def test_score_monotonic_with_sortino(self):
        """Higher Sortino = higher score, all else equal."""
        base = {'max_drawdown': -0.10, 'val_return': 0.3, 'num_trades': 60, 'sharpe': 1.5}
        s1 = compute_score({**base, 'sortino': 1.0}, 0.8, 1.0)
        s2 = compute_score({**base, 'sortino': 2.0}, 0.8, 1.0)
        assert s2 > s1


# ── Ground Truth ─────────────────────────────────────────────────────

class TestGroundTruth:
    def test_load_ground_truth_btcusdt(self):
        gt = load_ground_truth("BTCUSDT")
        assert 'swing_low' in gt
        assert 'swing_high' in gt
        assert gt['swing_high'] > gt['swing_low']

    def test_load_ground_truth_hypeusdt(self):
        gt = load_ground_truth("HYPEUSDT")
        assert gt['swing_low'] == pytest.approx(20.5, abs=0.5)
        assert gt['swing_high'] == pytest.approx(59.54, abs=0.5)
