"""Tests for strategy.py — signal generation and full pipeline."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from prepare import load_asset_ohlcv, compute_indicators, compute_fib_levels, evaluate_strategy
from strategy import strategy


class TestStrategy:
    def test_returns_signals_series(self):
        """Strategy must return a Series of -1, 0, or 1."""
        df = load_asset_ohlcv("BTCUSDT", "1D")
        df = compute_indicators(df)
        fib = compute_fib_levels(65056, 76000)
        signals = strategy(df, fib)
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(df)
        assert signals.isin([-1, 0, 1]).all()

    def test_produces_some_trades(self):
        """Strategy must not be all zeros."""
        df = load_asset_ohlcv("BTCUSDT", "1D")
        df = compute_indicators(df)
        fib = compute_fib_levels(65056, 76000)
        signals = strategy(df, fib)
        assert (signals != 0).sum() > 0, "Strategy produced no trades"

    def test_scores_nonnegative(self):
        """Full pipeline: strategy -> backtest -> score >= 0."""
        result = evaluate_strategy(strategy)
        assert result['score'] >= 0.0
        assert result['assets_evaluated'] > 0


class TestIntegration:
    def test_full_pipeline_baseline(self):
        """Full pipeline runs without error and produces parseable output."""
        import subprocess
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent.parent / "strategy.py")],
            capture_output=True, text=True, timeout=120,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0, f"strategy.py failed: {result.stderr[:500]}"

        from orchestrator import parse_strategy_output
        parsed = parse_strategy_output(result.stdout)
        assert 'score' in parsed
        assert 'trades' in parsed
        assert parsed.get('assets_evaluated', 0) > 0
        print(f"\nBASELINE SCORE: {parsed['score']:.4f}")
        print(f"  Sortino:  {parsed.get('sortino', 0):.4f}")
        print(f"  MaxDD:    {parsed.get('max_drawdown', 0):.4f}")
        print(f"  Trades:   {parsed.get('trades', 0)}")
        print(f"  Holdout:  {parsed.get('holdout_sharpe', 0):.4f}")
