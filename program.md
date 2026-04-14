# Program — Fibonacci Zone Trading Strategy Autoresearch

You are an autonomous research agent optimizing a Fibonacci zone trading strategy.
Your goal: maximize the `score` printed to stdout by `python3 prepare.py`.

## Setup

1. Read these files for full context:
   - `prepare.py` — evaluation harness, data loading, scoring. **READ-ONLY. DO NOT MODIFY.**
   - `strategy.py` — the file you modify. All strategy logic lives here.
   - `results.tsv` — experiment log. Read it to understand what has been tried.
   - This file (`program.md`) — domain knowledge and experiment rules.
2. Verify data exists: check that `pipeline/ground_truth.db` and OHLCV caches are present.
3. Initialize `results.tsv` with the header row if it does not exist.
4. Confirm setup, then begin the experiment loop.

## What You Can Modify

**Only `strategy.py`.** Everything in `prepare.py` is frozen: the asset list, data loading, Sortino-based scoring, MaxDD rejection, and the output format. Do not install new packages. Use only what is already imported or available in the environment (pandas, numpy, ta-lib/ta, smartmoneyconcepts).

## The Metric

`prepare.py` prints a single score to stdout:

```
score: 3.456
```

**Higher is better.** The score is a Sortino-based composite across multiple asset groups. A run that prints no `score:` line (crash or timeout) scores 0.

Hard reject: any asset with MaxDD > 20% zeroes the entire run. This is non-negotiable.

Extract the result:
```bash
grep "^score:" run.log | tail -1
```

## The Experiment Loop

LOOP FOREVER:

1. Read the current `strategy.py` and `results.tsv` to understand where you are.
2. Pick ONE idea from the ranked list below (or invent one). Change ONE thing.
3. Update the `DESCRIPTION` variable in `strategy.py` to describe the change.
4. `git commit -m "exp: <short description>"`
5. Run: `python3 prepare.py > run.log 2>&1`
6. Extract result: `grep "^score:" run.log | tail -1`
7. If grep is empty, the run crashed. Run `tail -n 50 run.log` for the traceback. Fix if trivial, otherwise skip.
8. Log the result to `results.tsv` (do NOT git-track results.tsv).
9. **If score improved**: keep the commit. Update the "Current Record" section below.
10. **If score is equal or worse**: `git reset --hard HEAD~1` to revert.
11. Go to step 1.

**Timeout**: each run should complete in under 5 minutes. Kill and discard anything over 10 minutes.

**Crashes**: fix typos/imports and retry once. If the idea is fundamentally broken, log "crash", revert, move on.

**NEVER STOP**: do not pause for confirmation. The human may be asleep. If you run out of ideas, re-read this file, combine near-misses, try the opposite of what failed, or explore a completely new angle. You run until interrupted.

---

## Domain Knowledge — Popek's Fibonacci Zone Method

### The Core Algorithm (r=0.803 correlation confirmed)

```
RSI OB/OS trigger → trade direction (SHORT/LONG) → draw Fib on impulse → zone entry → candle confirmation
```

1. **RSI is the TRIGGER, not a filter.** RSI overbought on higher TF → SHORT intent. RSI oversold → LONG intent.
2. **Draw Fib on the impulse**: most recent completed impulse (ICT Dealing Range). NOT the largest. NOT the oldest.
3. **Entry zones**:
   - 0.45-0.50 = "safe zone" — 100% historical hit rate across verified assets
   - 0.577-0.618 = "golden zone" — 92% hit rate, better risk/reward (deeper retrace + S/R flip confluence)
   - 0.705 (ICT OTE) = 31% hit rate — only with strong confluence
4. **Confirmation**: shooting star / rejection candle at zone + volume divergence
5. **Direction from RSI intent, NOT trend analysis** — this was the breakthrough insight

### The Silver Rule
Use the structural Higher Low (for UP impulse) or Lower High (for DOWN impulse) as the Fib anchor — not the absolute extreme. The HL/LH is where buying/selling pressure exhausted.

### Swing Detection Hierarchy
1. CHoCH (Change of Character) — marks impulse start
2. BOS (Break of Structure) — confirms impulse continuation
3. Spike fallback — if absolute high > 1.25x structural swing, use absolute extreme
4. Magnitude gate — range must be > 1.5x ATR(14) to qualify

### Asset Groups (strategy must work across ALL groups)

| Group | Assets | Characteristics |
|-------|--------|----------------|
| Crypto majors | BTCUSDT, BTCUSD | Low volatility relative to alts, high liquidity |
| Crypto alts | ZECUSDT, HYPEUSDT, TAOUSDT, ENJUSDT, RENDERUSDT, MONUSD | High volatility, thin books, spike wicks |
| Stocks | MARA, NVDA | Gaps, pre/post market, earnings events |
| Commodities | UKOIL, Gold | Fundamentals-driven, slow trends, geopolitical sensitivity |
| Indices | DAX, NASDAQ, SP500 | Correlated, mean-reverting intraday, trending on daily |

**CRITICAL**: a strategy that only works on crypto will be rejected. The strategy MUST generalize across at least 3 of 5 groups. This is the primary constraint.

### Indicator Toolkit (available in strategy.py)

| Indicator | Usage | Notes |
|-----------|-------|-------|
| RSI(14) | OB/OS trigger (>65 / <35) | THE primary signal — r=0.803 with direction |
| Stochastic %K/%D | OB/OS confirmation | 54% hit at highs, 46% at lows — best addition |
| MFI | Money flow confirmation | Useless at bottoms (0% hit <20). Only use for OB confirmation |
| EMA 33/66/144/288 | Trend stack, S/R levels | EMA33 vs EMA144 for trend, EMAs as dynamic S/R |
| ATR(14) | Volatility filter, stop sizing | Magnitude gate = 1.5x ATR minimum |
| Volume ratio | Institutional activity | Spike volume at zone = institutional interest |
| CVD Z-Score | Cumulative volume delta | Divergence at Fib levels = reversal signal |
| SuperTrend | Trend confirmation | Binary bull/bear, useful for regime filter |
| ADX | Trend strength | >25 = trending, <20 = ranging (Fib works in trends only) |
| BOS/CHoCH | Structure breaks | Via smartmoneyconcepts library |

### Zone Hit Statistics (verified on 11 assets)

| Zone | Hit Rate | Notes |
|------|----------|-------|
| 0.236 | ~40% | Shallow, strong trend only |
| 0.382 | ~65% | First significant level |
| 0.45-0.50 | 100% | Primary safe zone |
| 0.50 | ~85% | Psychological level, often exact reversal |
| 0.577-0.618 | 92% | Golden zone, best R/R |
| 0.667 | ~60% | Between golden and deep |
| 0.705 (OTE) | 31% | ICT optimal trade entry, needs strong confluence |
| 0.786 | ~25% | Deep retrace, often last stand |
| 0.875 | ~15% | Harmonic zone (Gartley/Bat), rare |

---

## Ranked Ideas to Try

### Tier 1 — High Expected Impact (try these first)

1. **RSI-triggered direction instead of trend-based direction** — Use RSI(14) > 65 = SHORT, < 35 = LONG as the primary signal. This is popek's actual method and correlates r=0.803 with correct direction. Most impactful single change.

2. **Dual-zone entry with tiered sizing** — Enter 50% position at 0.45-0.5 zone, remaining 50% at 0.577-0.618 if reached. Average entry improves R/R. The 100% hit rate on 0.45-0.5 means the first tranche almost never fails.

3. **ATR-adaptive stop loss** — Set stop at swing extreme + 0.5x ATR instead of fixed percentage. Volatile assets (crypto alts) need wider stops; stable assets (indices) need tighter ones. Prevents MaxDD blowouts.

4. **Regime filter: ADX > 25 required** — Fib only works in trending markets. Skip signals when ADX < 25 (ranging). Avoids the biggest source of false signals.

5. **Multi-timeframe RSI confirmation** — RSI OB on daily AND 4h (or daily AND weekly) = stronger signal than single TF. Reduces false triggers by ~40%.

6. **Rejection candle confirmation** — Don't enter at zone touch. Wait for a bearish engulfing/shooting star (SHORT) or hammer/bullish engulfing (LONG) at the zone. Reduces premature entries.

7. **Volume spike at zone** — Require volume > 1.5x 20-period average at zone touch. Institutional participation confirms the level. Without volume, zone touches are noise.

### Tier 2 — Medium Expected Impact

8. **Stochastic + RSI double confirmation** — Both RSI and Stochastic in OB/OS simultaneously. Stochastic is the best additional oscillator (54% at highs). Reduces false positives.

9. **EMA stack as trend qualifier** — EMA33 > EMA66 > EMA144 > EMA288 = bullish. All reversed = bearish. Mixed = skip. Only take Fib signals aligned with EMA stack direction.

10. **Adaptive lookback window** — Use ATR-normalized macro_ratio to pick short (10-60 bar) vs long (80-300 bar) window for swing detection. The two-cluster problem (section 8 of autoresearch findings) requires this.

11. **S/R flip confluence scoring** — At the Fib zone, check if there is a prior support level (for SHORT) or resistance level (for LONG) within 1% of the zone price. S/R flip = +1 confluence point. Popek explicitly uses this for 0.577-0.618 entries.

12. **CVD divergence at zone** — Price reaches zone but CVD Z-score diverges (e.g., price up but CVD down at SHORT zone). Divergence = smart money already selling. High-conviction signal.

13. **Partial take-profit at 0.236 / 0.382** — After entering SHORT at golden zone, take 50% profit at 0.382 retrace back, move stop to breakeven. Lock in gains, let runners run.

14. **Sub-cycle detection** — When price is in a new range (dealing range), use the sub-cycle impulse, not the full historical impulse. ZEC: full was 38-750, popek drew 185-562. Use dealing range detection from smartmoneyconcepts.

### Tier 3 — Lower Expected Impact / Exploratory

15. **Fibonacci cluster scoring** — Draw Fib on multiple recent impulses. Where levels overlap within 1% = high-confidence zone. More confluent clusters = stronger entries.

16. **Time-based exit** — If position does not reach TP within N bars, exit at market. Dead trades tie up capital. N = 2x the impulse duration.

17. **Funding rate filter (crypto only)** — Extreme negative funding = overleveraged shorts = likely bounce. Skip SHORT signals when funding < -0.03%. Vice versa for LONG.

18. **Order book wall detection** — Large resting orders at Fib levels = institutional interest. 17.9-sigma wall at BTC 0.577 confirmed in order book data.

19. **Elliott wave context** — If the current move is wave 5 (final wave), Fib retracement is more likely to be deep (0.618+). Wave 3 retracements tend to be shallow (0.382).

20. **Intraday session filter** — For stocks/indices, only enter during high-liquidity sessions (9:30-11:30 ET, 13:00-15:30 ET). Avoid lunch hour and after-hours noise.

21. **Gold/TAO LOE/Quasimodo confluence** — LOE (Level of Entry) and Quasimodo patterns work particularly well on Gold and TAO. Add as bonus confluence for these specific assets.

22. **FRAMA fractal dimension** — Use fractal adaptive moving average to detect whether the market is trending or choppy. FRAMA < 1.3 = trending (take signals), > 1.7 = choppy (skip).

23. **MaxDD circuit breaker** — If drawdown hits 15%, reduce position size by 50% until new equity high. Prevents the hard 20% MaxDD reject.

24. **Correlation filter** — If BTC and the target altcoin have correlation > 0.8, use BTC's Fib levels as additional confluence. High-correlation regimes mean alts follow BTC structure.

25. **Mean reversion overlay for indices** — DAX/NASDAQ/SP500 mean-revert intraday. Use Bollinger Band extremes + Fib zone confluence for index-specific entries.

---

## Anti-Patterns — What NOT to Do

1. **DO NOT optimize per-asset.** A strategy that has `if symbol == 'BTCUSDT': use_special_params()` is overfitting. The strategy must use universal rules parameterized by market regime, not by ticker name.

2. **DO NOT use MFI at bottoms.** MFI has 0% hit rate below 20 at verified swing lows. It is only useful for overbought confirmation (>80).

3. **DO NOT use EMA crossovers for direction.** EMA+CHoCH combined scored 5/15 vs CHoCH alone 7/15. EMAs make direction detection WORSE. Use RSI intent or structure (HH/HL/LH/LL).

4. **DO NOT use fixed percentage stops.** A 5% stop on BTC is reasonable; on MARA (which gaps 10% regularly) it is a guaranteed stop-out. ATR-based stops adapt to each asset.

5. **DO NOT increase lookback window beyond 300 bars.** More data does not help — it includes stale structure from old regimes. 300 bars on 1D = ~14 months, sufficient for any impulse.

6. **DO NOT trade against the RSI trigger.** If RSI says OB (SHORT), do not take LONG entries even if trend is bullish. RSI trigger = direction. Trend = bias, not override.

7. **DO NOT change multiple things at once.** Each experiment changes ONE parameter, ONE indicator, ONE rule. If you change two things and score improves, you do not know which one helped. Strict isolation.

8. **DO NOT trust VLM/vision-extracted prices.** Always use OHLCV data. A VLM once said BTC was at 73k when it was at 58.55. Pipeline believed it.

9. **DO NOT target very short horizons (< 24 bars on 1h, < 1 bar on 1D).** The scoring metric explodes due to compounding artifacts. 2000+ trades/year with 0.1% fees = 200% annual costs.

10. **DO NOT remove the MaxDD constraint.** It exists for survival. A strategy with 50% drawdown is unusable regardless of returns. The 20% hard cap is the reality constraint.

11. **DO NOT use dropout > 0.4 on models.** If using ML components, dropout 0.4+ kills signal for stable assets (ETH went 96% flat at dropout 0.5).

12. **DO NOT use ensemble of per-asset models.** Single cross-asset model outperforms (score 3.45 vs 1.60 for per-asset ensemble). Cross-asset correlations matter.

---

## Experiment Tracking

### results.tsv Format (tab-separated)

```
commit	score	maxdd	status	description
```

| Column | Description |
|--------|-------------|
| commit | git short hash (7 chars) |
| score | score from stdout (0.000 for crashes) |
| maxdd | worst single-asset max drawdown % (0.0 for crashes) |
| status | `keep`, `discard`, or `crash` |
| description | what this experiment tried (1 line) |

Example:
```
commit	score	maxdd	status	description
a1b2c3d	2.100	14.2	keep	baseline
b2c3d4e	2.340	12.8	keep	add RSI trigger for direction
c3d4e5f	1.890	18.5	discard	remove ATR stop (worse DD)
d4e5f6g	0.000	0.0	crash	syntax error in zone detection
```

### Current Record

**Score: [TO BE FILLED AFTER BASELINE]**
**Config: [TO BE FILLED AFTER BASELINE]**

Update this section after each new high score.

---

## One-Change Discipline

Before each experiment, write down:
1. **Hypothesis**: "I expect [change X] to improve score because [reason Y]."
2. **Variable**: exactly ONE thing changed.
3. **Prediction**: "Score should go from A to approximately B."
4. **Rollback plan**: the change is in a single git commit that can be reverted.

After each experiment, record:
1. **Actual score**: what happened.
2. **Verdict**: keep / discard / crash.
3. **Learning**: what did this teach about the strategy?

---

## Key Constraints Summary

- Strategy must generalize across 3+ of 5 asset groups (crypto major, crypto alt, stocks, commodities, indices)
- MaxDD > 20% on ANY single asset = entire run rejected (score 0)
- Only modify `strategy.py` — `prepare.py` is frozen
- No new dependencies
- One change per experiment
- Log every experiment to `results.tsv`
- Never stop — run until interrupted
