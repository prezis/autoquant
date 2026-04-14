"""Tests for orchestrator.py — experiment DB, output parsing, decision logic."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator import ExperimentDB, parse_strategy_output, should_keep


class TestExperimentDB:
    def test_creates_table(self):
        db = ExperimentDB(":memory:")
        tables = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [row[0] for row in tables]
        assert 'experiments' in table_names

    def test_log_and_query(self):
        db = ExperimentDB(":memory:")
        db.log(git_hash="abc123", hypothesis="test RSI threshold",
               mutation="changed RSI < 35 to RSI < 30",
               score=1.5, sortino=2.0, max_drawdown=-0.10,
               consistency=0.8, val_return=0.3, trades=50,
               holdout_sharpe=1.2, status="kept", description="test")
        rows = db.get_best(n=1)
        assert len(rows) == 1
        assert rows[0]['score'] == 1.5

    def test_count(self):
        db = ExperimentDB(":memory:")
        assert db.count() == 0
        db.log(score=1.0, status="kept")
        assert db.count() == 1

    def test_get_recent(self):
        db = ExperimentDB(":memory:")
        for i in range(5):
            db.log(score=float(i), status="kept")
        recent = db.get_recent(3)
        assert len(recent) == 3
        assert recent[0]['score'] == 4.0  # most recent first

    def test_summary(self):
        db = ExperimentDB(":memory:")
        db.log(score=1.5, status="kept")
        db.log(score=0.5, status="reverted")
        s = db.summary()
        assert "2 total" in s
        assert "1 kept" in s
        assert "1.5" in s


class TestOutputParsing:
    def test_parse_full_output(self):
        output = """score:        1.2345
sortino:      2.0000
max_drawdown: -0.1500
val_return:   0.3000
trades:       45
consistency:  0.8000
holdout:      1.1000
assets:       8"""
        result = parse_strategy_output(output)
        assert result['score'] == pytest.approx(1.2345)
        assert result['trades'] == 45
        assert result['max_drawdown'] == pytest.approx(-0.15)
        assert result['assets_evaluated'] == 8

    def test_parse_empty_output(self):
        result = parse_strategy_output("")
        assert result == {}

    def test_parse_partial_output(self):
        result = parse_strategy_output("score:        0.5000\ntrades:       10")
        assert result['score'] == pytest.approx(0.5)
        assert result['trades'] == 10


class TestDecisionLogic:
    def test_keep_better_score(self):
        assert should_keep(new_score=1.5, best_score=1.0) is True

    def test_reject_worse_score(self):
        assert should_keep(new_score=0.5, best_score=1.0) is False

    def test_reject_zero_score(self):
        assert should_keep(new_score=0.0, best_score=0.5) is False

    def test_keep_first_positive(self):
        assert should_keep(new_score=0.1, best_score=0.0) is True
