# orchestrator.py — Experiment loop controller
# ═══════════════════════════════════════════════════════════════════
# Manages: experiment DB, output parsing, git commit/revert, loop control
# READ-ONLY: The autonomous agent must NOT modify this file.
# ═══════════════════════════════════════════════════════════════════

import sqlite3
import subprocess
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


class ExperimentDB:
    """SQLite experiment log — queryable, not a TSV."""

    def __init__(self, db_path: str = "results.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                git_hash TEXT,
                hypothesis TEXT,
                mutation TEXT,
                score REAL NOT NULL,
                sortino REAL,
                max_drawdown REAL,
                consistency REAL,
                val_return REAL,
                trades INTEGER,
                holdout_sharpe REAL,
                status TEXT NOT NULL,
                description TEXT
            )
        """)
        self.conn.commit()

    def log(self, **kwargs) -> int:
        kwargs.setdefault('timestamp', datetime.now(timezone.utc).isoformat())
        cols = ', '.join(kwargs.keys())
        placeholders = ', '.join(['?'] * len(kwargs))
        cur = self.conn.execute(
            f"INSERT INTO experiments ({cols}) VALUES ({placeholders})",
            list(kwargs.values())
        )
        self.conn.commit()
        return cur.lastrowid

    def get_best(self, n: int = 5) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM experiments WHERE status='kept' ORDER BY score DESC LIMIT ?", (n,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_recent(self, n: int = 10) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM experiments ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]

    def summary(self) -> str:
        """Human-readable summary of experiment history."""
        total = self.count()
        best = self.get_best(1)
        best_score = best[0]['score'] if best else 0.0
        kept = self.conn.execute("SELECT COUNT(*) FROM experiments WHERE status='kept'").fetchone()[0]
        return f"Experiments: {total} total, {kept} kept, best score: {best_score:.4f}"


def parse_strategy_output(output: str) -> dict:
    """Parse stdout from strategy.py into a result dict."""
    result = {}
    patterns = {
        'score': r'score:\s+([-\d.]+)',
        'sortino': r'sortino:\s+([-\d.]+)',
        'max_drawdown': r'max_drawdown:\s+([-\d.]+)',
        'val_return': r'val_return:\s+([-\d.]+)',
        'trades': r'trades:\s+(\d+)',
        'consistency': r'consistency:\s+([-\d.]+)',
        'holdout_sharpe': r'holdout:\s+([-\d.]+)',
        'assets_evaluated': r'assets:\s+(\d+)',
    }
    for key, pat in patterns.items():
        m = re.search(pat, output)
        if m:
            val = m.group(1)
            result[key] = int(val) if key in ('trades', 'assets_evaluated') else float(val)
    return result


def should_keep(new_score: float, best_score: float) -> bool:
    """Should we keep this experiment (git commit) or revert?"""
    if new_score == 0.0:
        return False  # hard reject
    return new_score > best_score


def git_hash() -> str:
    """Get current git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def run_strategy(timeout: int = 120) -> dict:
    """Run strategy.py and parse output."""
    try:
        result = subprocess.run(
            [sys.executable, "strategy.py"],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(Path(__file__).parent),
        )
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr[:500]}")
            return {'score': 0.0, 'error': result.stderr[:500]}
        parsed = parse_strategy_output(result.stdout)
        if not parsed:
            print(f"  ERROR: no score in output")
            print(f"  STDOUT: {result.stdout[:500]}")
            return {'score': 0.0, 'error': 'no score parsed'}
        return parsed
    except subprocess.TimeoutExpired:
        return {'score': 0.0, 'error': 'timeout'}


def git_commit(message: str):
    subprocess.run(["git", "add", "strategy.py"], check=True)
    subprocess.run(["git", "commit", "-m", message], check=True)


def git_revert():
    subprocess.run(["git", "checkout", "strategy.py"], check=True)


if __name__ == "__main__":
    # Standalone mode: run strategy once, log result
    db = ExperimentDB()
    exp_num = db.count() + 1
    print(f"═══ Experiment #{exp_num} ═══")
    print(f"{db.summary()}")
    print()

    result = run_strategy()
    score = result.get('score', 0)
    print(f"Score: {score:.4f}")

    best = db.get_best(n=1)
    best_score = best[0]['score'] if best else 0.0

    if should_keep(score, best_score):
        status = 'kept'
        git_commit(f"exp #{exp_num}: score {score:.4f}")
        print(f"  ✓ KEPT (new best: {score:.4f} > {best_score:.4f})")
    else:
        status = 'reverted'
        git_revert()
        print(f"  ✗ REVERTED (score {score:.4f} <= best {best_score:.4f})")

    db.log(
        git_hash=git_hash(),
        score=score,
        sortino=result.get('sortino', 0),
        max_drawdown=result.get('max_drawdown', 0),
        consistency=result.get('consistency', 0),
        val_return=result.get('val_return', 0),
        trades=result.get('trades', 0),
        holdout_sharpe=result.get('holdout_sharpe', 0),
        status=status,
        description=result.get('error', f"score={score:.4f}"),
    )
    print(f"\n{db.summary()}")
