"""
strategy.py — Ichimoku + RSI + MACD + ATR trailing stop + barometry (4H)
Agent modyfikuje TEN plik. prepare.py jest read-only.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
pd.set_option('future.no_silent_downcasting', True)
from prepare import evaluate, plot_equity

RESULTS_FILE = Path(__file__).parent / "results.tsv"
OPIS = "Ichimoku+RSI+MACD+ATR_stop 4H proporcjonalne"


# ─── Wskaźniki ───────────────────────────────────────────────────

def ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26,
             senkou_b: int = 52) -> dict[str, pd.Series]:
    """Ichimoku Kinko Hyo — chmura i linie bazowe."""
    high, low, close = df["high"], df["low"], df["close"]

    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    senkou_span_b = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)

    return {
        "tenkan": tenkan_sen,
        "kijun": kijun_sen,
        "senkou_a": senkou_span_a,
        "senkou_b": senkou_span_b,
    }


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI — Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> dict[str, pd.Series]:
    """MACD — Moving Average Convergence Divergence."""
    ema_fast = series.ewm(span=fast, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return {"macd": macd_line, "signal": signal_line, "hist": histogram}


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR — Average True Range (zmienność)."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, min_periods=period).mean()


def atr_trailing_stop(close: pd.Series, atr_values: pd.Series,
                      positions: pd.Series, multiplier: float = 2.0) -> pd.Series:
    """
    ATR trailing stop — zamyka pozycję gdy cena spadnie o multiplier×ATR
    od szczytu (long) lub wzrośnie od dołka (short).

    Zwraca zmodyfikowane pozycje z wyzerowanymi stopami.
    """
    result = positions.copy()
    trail_price = np.nan
    direction = 0  # 1=long, -1=short

    for i in range(len(close)):
        pos = result.iloc[i]
        price = close.iloc[i]
        atr_val = atr_values.iloc[i]

        if np.isnan(atr_val):
            continue

        stop_dist = multiplier * atr_val

        if pos > 0:  # Long
            if direction != 1:
                # Nowy long — ustaw trailing stop
                trail_price = price - stop_dist
                direction = 1
            else:
                # Aktualizuj trailing stop w górę
                new_trail = price - stop_dist
                trail_price = max(trail_price, new_trail)

                # Sprawdź czy cena spadła poniżej trailing stop
                if price < trail_price:
                    result.iloc[i] = 0
                    direction = 0

        elif pos < 0:  # Short
            if direction != -1:
                # Nowy short — ustaw trailing stop
                trail_price = price + stop_dist
                direction = -1
            else:
                # Aktualizuj trailing stop w dół
                new_trail = price + stop_dist
                trail_price = min(trail_price, new_trail)

                # Sprawdź czy cena wzrosła powyżej trailing stop
                if price > trail_price:
                    result.iloc[i] = 0
                    direction = 0

        else:  # Flat
            direction = 0

    return result


# ─── Strategia ───────────────────────────────────────────────────

def strategy(df: pd.DataFrame, context: dict[str, pd.DataFrame]) -> pd.Series:
    """
    Ichimoku + RSI + MACD + ATR trailing stop + barometry.

    Ulepszenia vs poprzednia wersja:
    1. MACD jako filtr momentum — potwierdza siłę ruchu
    2. ATR trailing stop — chroni zyski, zamyka gdy 2×ATR od szczytu
    3. Proporcjonalne pozycje — siła sygnału 0.5 lub 1.0

    Logika long:
        - Cena > chmury + tenkan > kijun + RSI > 45
        - MACD histogram > 0 → pełny long (1.0)
        - MACD histogram < 0 → częściowy long (0.5)
        - ATR trailing stop zamyka pozycję

    Logika short (ostrożna):
        - Cena < chmury + tenkan < kijun + RSI < 40
        - Makro bearish (SPY spadkowy LUB DXY rośnie)
        - MACD histogram < 0 → pełny short (-1.0)
        - MACD histogram > 0 → brak shorta
    """
    close = df["close"]

    # ─── Ichimoku ───
    ichi = ichimoku(df)
    tenkan = ichi["tenkan"]
    kijun = ichi["kijun"]
    span_a = ichi["senkou_a"]
    span_b = ichi["senkou_b"]

    cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
    cloud_bottom = pd.concat([span_a, span_b], axis=1).min(axis=1)

    # ─── RSI ───
    rsi_val = rsi(close, period=14)

    # ─── MACD ───
    macd_data = macd(close)
    macd_hist = macd_data["hist"]
    macd_bullish = macd_hist > 0
    macd_bearish = macd_hist < 0

    # ─── ATR ───
    atr_val = atr(df, period=14)

    # ─── Barometry ───
    spy_bearish = pd.Series(False, index=df.index)
    dxy_rising = pd.Series(False, index=df.index)

    if "SPY" in context and len(context["SPY"]) > 200:
        spy = context["SPY"]["close"]
        spy_sma50 = spy.rolling(50).mean()
        spy_sma200 = spy.rolling(200).mean()
        spy_bearish = (~(spy_sma50 > spy_sma200)).reindex(df.index, method="ffill").fillna(False)

    if "UUP" in context and len(context["UUP"]) > 50:
        uup = context["UUP"]["close"]
        uup_sma20 = uup.rolling(20).mean()
        uup_sma50 = uup.rolling(50).mean()
        dxy_rising = (uup_sma20 > uup_sma50).reindex(df.index, method="ffill").fillna(False)

    macro_bearish = spy_bearish | dxy_rising

    # ─── Sygnały z siłą ───
    signals = pd.Series(0.0, index=df.index)

    # Warunek bazowy long: cena nad chmurą + tenkan > kijun + RSI > 45
    long_base = (
        (close > cloud_top)
        & (tenkan > kijun)
        & (rsi_val > 45)
    )

    # Long pełny (1.0): MACD potwierdza momentum
    signals[long_base & macd_bullish] = 1.0
    # Long częściowy (0.5): Ichimoku+RSI OK, ale MACD nie potwierdza
    signals[long_base & ~macd_bullish] = 0.5

    # Short (ostrożny): wymaga makro bearish + MACD bearish
    short_cond = (
        (close < cloud_bottom)
        & (tenkan < kijun)
        & (rsi_val < 40)
        & macro_bearish
        & macd_bearish
    )
    signals[short_cond] = -1.0

    # ─── ATR trailing stop ───
    signals = atr_trailing_stop(close, atr_val, signals, multiplier=2.0)

    return signals


# ─── Runner ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("AUTOQUANT — Ichimoku+RSI+MACD+ATR stop (4H, long/short)")
    print("=" * 60)
    print()

    results = evaluate(strategy, timeframe="4h")

    avg_score = results["_avg_score"]

    print()
    print("=" * 60)
    print(f"WYNIK KOŃCOWY (avg score): {avg_score}")
    print("=" * 60)

    plot_equity(results)

    # ─── Zapis do results.tsv ───
    assets = [k for k in results if not k.startswith("_")]

    if not RESULTS_FILE.exists():
        with open(RESULTS_FILE, "w") as f:
            f.write("nr\tdata\tscore\tsharpe_train\tsharpe_val\t"
                    "return_train\treturn_val\tmax_dd_val\t"
                    "trades_val\topis\n")

    with open(RESULTS_FILE, "r") as f:
        nr = max(len(f.readlines()) - 1, 0) + 1

    val_sharpes = [results[a]["val"]["sharpe"] for a in assets]
    train_sharpes = [results[a]["train"]["sharpe"] for a in assets]
    val_returns = [results[a]["val"]["total_return"] for a in assets]
    train_returns = [results[a]["train"]["total_return"] for a in assets]
    val_dds = [results[a]["val"]["max_drawdown"] for a in assets]
    val_trades = [results[a]["val"]["num_trades"] for a in assets]

    row = (
        f"{nr}\t"
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}\t"
        f"{avg_score:.4f}\t"
        f"{np.mean(train_sharpes):.3f}\t"
        f"{np.mean(val_sharpes):.3f}\t"
        f"{np.mean(train_returns):.2%}\t"
        f"{np.mean(val_returns):.2%}\t"
        f"{np.mean(val_dds):.2%}\t"
        f"{int(np.mean(val_trades))}\t"
        f"{OPIS}"
    )

    with open(RESULTS_FILE, "a") as f:
        f.write(row + "\n")

    print(f"\n📊 Zapisano wynik #{nr} → {RESULTS_FILE}")
