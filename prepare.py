"""
prepare.py — pobieranie danych + silnik backtestu + scoring
READ-ONLY dla agenta. Modyfikuje tylko człowiek.

Źródła danych:
  - Krypto (BTC, ETH, XMR, SOL, TAO) → ccxt/Binance (1h świece, pełna historia)
  - Barometry (SPY, QQQ, UUP)        → Alpha Vantage (1h intraday, klucz premium)

Barometry NIE są handlowane — służą jako kontekst makro dla strategii.
Krypto jest handlowane — long (+1), short (-1), flat (0).
"""

import os
import time
import numpy as np
import pandas as pd
import ccxt
import requests
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv

# Załaduj klucz API z .env
load_dotenv(Path(__file__).parent / ".env")

# ─── Konfiguracja ───────────────────────────────────────────────

# Aktywa handlowane (krypto, ccxt)
# Domyślnie Binance, wyjątki w EXCHANGE_OVERRIDES
CRYPTO_ASSETS = ["BTC/USDT", "ETH/USDT", "XMR/USDT", "SOL/USDT", "TAO/USDT"]

# Giełdy per asset (domyślnie Binance, tu wyjątki)
EXCHANGE_OVERRIDES = {
    "XMR/USDT": "bitfinex",  # Binance zdelisował XMR, Bitfinex ma pełną historię
    # TAO/USDT — Binance (od 2024-04), ~11 mies. train + 1 rok val
}

# Barometry rynku (kontekst makro, Alpha Vantage)
BAROMETER_ASSETS = ["SPY", "QQQ", "UUP"]

INTERVAL_1H = "1h"
CACHE_DIR = Path.home() / ".cache" / "autoquant" / "data"

# Okresy (train / validation)
# Okresy: 3 lata wstecz od 2026-03-17
TRAIN_START = "2023-03-17"
TRAIN_END   = "2025-03-17"
VAL_START   = "2025-03-17"
VAL_END     = "2026-03-17"

# Koszty transakcji (spot Binance)
COMMISSION = 0.001   # 0.1% taker fee
SLIPPAGE   = 0.0003  # 0.03%

# Alpha Vantage
AV_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
AV_BASE_URL = "https://www.alphavantage.co/query"


# ─── Pobieranie danych krypto (ccxt, wiele giełd) ────────────────

def _get_exchange(symbol: str):
    """Zwraca instancję giełdy ccxt dla danego symbolu."""
    exchange_id = EXCHANGE_OVERRIDES.get(symbol, "binance")
    exchange_cls = getattr(ccxt, exchange_id)
    return exchange_cls({"enableRateLimit": True}), exchange_id


def _fetch_crypto_ohlcv(symbol: str, timeframe: str = "1h",
                        since: str = "2017-01-01") -> pd.DataFrame:
    """
    Pobiera pełną historię OHLCV przez ccxt.
    Giełda wybierana automatycznie z EXCHANGE_OVERRIDES.
    """
    exchange, exchange_id = _get_exchange(symbol)
    since_ts = exchange.parse8601(f"{since}T00:00:00Z")
    limit = 1000

    all_candles = []
    print(f"    Pobieram {symbol} {timeframe} od {since} ({exchange_id})...")

    while True:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=since_ts, limit=limit)
        except Exception as e:
            if "ratelimit" in str(e).lower() or "rate" in str(e).lower():
                print(f"    ⏳ Rate limit, czekam 60s...")
                time.sleep(60)
                continue
            raise

        if not candles:
            break

        all_candles.extend(candles)
        since_ts = candles[-1][0] + 1

        if len(candles) < limit:
            break

        if len(all_candles) % 10000 < limit:
            dt = pd.Timestamp(candles[-1][0], unit="ms")
            print(f"    ... {len(all_candles)} świec, ostatnia: {dt}")

        # Pauza między requestami (unikanie rate limit)
        time.sleep(0.5)

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp")
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    return df


def download_crypto(symbol: str, force: bool = False) -> pd.DataFrame:
    """Pobiera dane krypto 1h z odpowiedniej giełdy, cache jako parquet."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _, exchange_id = _get_exchange(symbol)
    safe_name = symbol.replace("/", "_")
    cache_path = CACHE_DIR / f"{safe_name}_{exchange_id}_1h.parquet"

    if cache_path.exists() and not force:
        print(f"  {symbol}: wczytano z cache ({cache_path})")
        return pd.read_parquet(cache_path)

    print(f"  {symbol}: pobieram 1h z {exchange_id}...")
    df = _fetch_crypto_ohlcv(symbol, INTERVAL_1H)

    if df.empty:
        raise RuntimeError(f"Brak danych dla {symbol}")

    df.to_parquet(cache_path)
    print(f"  {symbol}: zapisano {len(df)} świec 1h → {cache_path}")
    return df


# ─── Pobieranie barometrów (Alpha Vantage) ──────────────────────

def _fetch_av_intraday(symbol: str, interval: str = "60min") -> pd.DataFrame:
    """
    Pobiera dane intraday z Alpha Vantage (TIME_SERIES_INTRADAY).
    Premium: outputsize=full daje pełną historię (do ~2 lat 1h danych).
    """
    if not AV_API_KEY:
        raise RuntimeError(
            "Brak klucza Alpha Vantage! Ustaw ALPHA_VANTAGE_API_KEY w .env"
        )

    print(f"    Pobieram {symbol} intraday ({interval}) z Alpha Vantage...")

    # AV z premium i extended_hours + pełna historia wymaga month-by-month
    # Ale z outputsize=full zwraca maksymalną dostępną historię
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "outputsize": "full",
        "datatype": "json",
        "apikey": AV_API_KEY,
    }

    all_frames = []

    # AV premium: pobieramy miesiąc po miesiącu dla pełnej historii
    # Generujemy listę miesięcy od 2019-01 do teraz
    months = pd.date_range(start="2019-01", end=pd.Timestamp.now(), freq="MS")

    for month in months:
        month_str = month.strftime("%Y-%m")
        params_month = {**params, "month": month_str}

        resp = requests.get(AV_BASE_URL, params=params_month, timeout=30)
        data = resp.json()

        ts_key = f"Time Series ({interval})"
        if ts_key not in data:
            # Może być limit API albo brak danych dla tego miesiąca
            if "Note" in data or "Information" in data:
                msg = data.get("Note", data.get("Information", ""))
                print(f"    ⚠ AV limit dla {symbol} {month_str}: {msg[:80]}")
                time.sleep(12)  # Czekaj na reset limitu
                continue
            # Brak danych — pomijamy miesiąc
            continue

        ts_data = data[ts_key]
        rows = []
        for dt_str, vals in ts_data.items():
            rows.append({
                "timestamp": pd.Timestamp(dt_str),
                "open": float(vals["1. open"]),
                "high": float(vals["2. high"]),
                "low": float(vals["3. low"]),
                "close": float(vals["4. close"]),
                "volume": float(vals["5. volume"]),
            })

        if rows:
            all_frames.append(pd.DataFrame(rows))

        # Progres
        if len(all_frames) % 12 == 0 and all_frames:
            print(f"    ... pobrano {len(all_frames)} miesięcy danych")

        # Rate limit: 75 calls/min na premium, ale lepiej nie ryzykować
        time.sleep(0.9)

    if not all_frames:
        raise RuntimeError(f"Brak danych AV dla {symbol}")

    df = pd.concat(all_frames, ignore_index=True)
    df = df.set_index("timestamp")
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    return df


def download_barometer(symbol: str, force: bool = False) -> pd.DataFrame:
    """Pobiera dane barometru 1h z Alpha Vantage, cache jako parquet."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{symbol}_1h.parquet"

    if cache_path.exists() and not force:
        print(f"  {symbol}: wczytano z cache ({cache_path})")
        return pd.read_parquet(cache_path)

    print(f"  {symbol}: pobieram 1h z Alpha Vantage...")
    df = _fetch_av_intraday(symbol, interval="60min")

    if df.empty:
        raise RuntimeError(f"Brak danych dla {symbol}")

    df.to_parquet(cache_path)
    print(f"  {symbol}: zapisano {len(df)} świec 1h → {cache_path}")
    return df


# ─── Agregacja do 4H ────────────────────────────────────────────

def resample_to_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Agreguje dane 1h do świec 4h."""
    df_4h = df.resample("4h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    return df_4h


# ─── Ładowanie wszystkich danych ─────────────────────────────────

def load_all_data(timeframe: str = "1h") -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """
    Pobiera/wczytuje dane dla krypto + barometrów.

    Args:
        timeframe: "1h" lub "4h"

    Returns:
        (crypto_data, barometer_data) — dwa słowniki DataFrames
    """
    print(f"Ładowanie danych ({timeframe})...")

    crypto_data = {}
    for symbol in CRYPTO_ASSETS:
        try:
            df = download_crypto(symbol)
            if timeframe == "4h":
                df = resample_to_4h(df)
            crypto_data[symbol] = df
        except Exception as e:
            print(f"  ⚠ {symbol}: błąd pobierania — {e}")

    barometer_data = {}
    for symbol in BAROMETER_ASSETS:
        try:
            df = download_barometer(symbol)
            if timeframe == "4h":
                df = resample_to_4h(df)
            barometer_data[symbol] = df
        except Exception as e:
            print(f"  ⚠ {symbol}: błąd pobierania — {e}")

    print(f"\nZaładowano: {len(crypto_data)} krypto, {len(barometer_data)} barometrów\n")
    return crypto_data, barometer_data


def split_periods(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dzieli dane na train i validation."""
    train = df.loc[TRAIN_START:TRAIN_END].copy()
    val = df.loc[VAL_START:VAL_END].copy()
    return train, val


# ─── Silnik backtestu (long + short) ────────────────────────────

def backtest(df: pd.DataFrame, signals: pd.Series) -> dict:
    """
    Wektoryzowany backtest — long/short/flat.

    Args:
        df: DataFrame z kolumnami open/high/low/close/volume
        signals: Series z sygnałami:
                 +1 = long, -1 = short, 0 = flat

    Returns:
        dict z metrykami
    """
    # Wyrównaj sygnały z danymi (shift o 1 — sygnał z poprzedniej świecy)
    signals = signals.reindex(df.index).fillna(0).shift(1).fillna(0)

    # Clamp do [-1, 1]
    signals = signals.clip(lower=-1, upper=1)

    # Zwroty per-świeca
    returns = df["close"].pct_change().fillna(0)

    # Koszty transakcji przy zmianie pozycji
    position_changes = signals.diff().abs().fillna(0)
    costs = position_changes * (COMMISSION + SLIPPAGE)

    # Zwroty strategii (long: +1*return, short: -1*return)
    strategy_returns = signals * returns - costs

    # Krzywa equity
    equity = (1 + strategy_returns).cumprod()

    # ─── Metryki ───
    periods_per_year = 365 * 24  # 1h świece, krypto 24/7
    if _detect_timeframe(df) == "4h":
        periods_per_year = 365 * 6  # 6 świec 4h dziennie

    mean_ret = strategy_returns.mean()
    std_ret = strategy_returns.std()

    # Sharpe annualizowany
    sharpe = (mean_ret / std_ret * np.sqrt(periods_per_year)) if std_ret > 0 else 0.0

    # Sortino (tylko ujemna zmienność)
    downside = strategy_returns[strategy_returns < 0]
    downside_std = downside.std() if len(downside) > 0 else 1e-9
    sortino = (mean_ret / downside_std * np.sqrt(periods_per_year)) if downside_std > 0 else 0.0

    # Max drawdown
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_drawdown = drawdown.min()

    # Total return
    total_return = equity.iloc[-1] / equity.iloc[0] - 1 if len(equity) > 0 else 0.0

    # Transakcje i win rate
    trades = signals.diff().fillna(0)
    trade_entries = trades[trades != 0]
    num_trades = len(trade_entries)

    winning = (strategy_returns > 0).sum()
    active = (strategy_returns != 0).sum()
    win_rate = winning / active if active > 0 else 0.0

    # Buy & hold dla porównania
    bh_return = df["close"].iloc[-1] / df["close"].iloc[0] - 1 if len(df) > 0 else 0.0

    # Statystyki long/short
    long_signals = (signals > 0).sum()
    short_signals = (signals < 0).sum()
    flat_signals = (signals == 0).sum()
    total_bars = len(signals)

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown": round(max_drawdown, 4),
        "total_return": round(total_return, 4),
        "win_rate": round(win_rate, 4),
        "num_trades": num_trades,
        "buy_hold_return": round(bh_return, 4),
        "long_pct": round(long_signals / total_bars, 3) if total_bars > 0 else 0,
        "short_pct": round(short_signals / total_bars, 3) if total_bars > 0 else 0,
        "flat_pct": round(flat_signals / total_bars, 3) if total_bars > 0 else 0,
        "equity_curve": equity,
    }


def _detect_timeframe(df: pd.DataFrame) -> str:
    """Wykrywa timeframe na podstawie mediany odstępów między świecami."""
    if len(df) < 2:
        return "1h"
    diffs = df.index.to_series().diff().dropna()
    median_hours = diffs.median().total_seconds() / 3600
    return "4h" if median_hours > 2.5 else "1h"


# ─── Scoring ────────────────────────────────────────────────────

def compute_score(train_metrics: dict, val_metrics: dict) -> float:
    """
    Composite score — wyższy = lepiej.

    Składniki:
    - 35% Sharpe (val)
    - 20% Sortino (val)
    - 20% (1 + max_drawdown) — kara za drawdown
    - 15% total return (val)
    - 10% win rate (val)
    × trade_penalty — min 50 transakcji na val
    × consistency — podobieństwo Sharpe train vs val
    """
    v = val_metrics

    raw = (
        0.35 * v["sharpe"]
        + 0.20 * v["sortino"]
        + 0.20 * (1 + v["max_drawdown"])
        + 0.15 * v["total_return"]
        + 0.10 * v["win_rate"]
    )

    # Kara za zbyt mało transakcji
    trade_penalty = min(v["num_trades"] / 50.0, 1.0)

    # Spójność train vs val
    t_sharpe = train_metrics["sharpe"]
    v_sharpe = v["sharpe"]
    if abs(t_sharpe) > 0.01:
        ratio = v_sharpe / t_sharpe
        consistency = max(0.0, min(1.0, 1.0 - abs(1.0 - ratio) * 0.5))
    else:
        consistency = 0.5

    score = raw * trade_penalty * consistency
    return round(score, 4)


# ─── Ewaluacja pełna ────────────────────────────────────────────

def evaluate(strategy_fn, timeframe: str = "1h") -> dict:
    """
    Uruchamia strategię na wszystkich krypto z kontekstem barometrów.

    Args:
        strategy_fn: funkcja strategy(df, context) → pd.Series sygnałów
            df — dane aktywa (krypto)
            context — dict z DataFrames barometrów (SPY, QQQ, UUP)
        timeframe: "1h" lub "4h"

    Returns:
        dict z wynikami per-asset + uśrednionym score
    """
    crypto_data, barometer_data = load_all_data(timeframe)
    results = {}
    scores = []

    for symbol, df in crypto_data.items():
        train_df, val_df = split_periods(df)

        if len(train_df) < 100 or len(val_df) < 100:
            print(f"  {symbol}: za mało danych "
                  f"({len(train_df)} train, {len(val_df)} val), pomijam")
            continue

        # Przygotuj kontekst barometrów (ten sam split czasowy)
        train_context = {}
        val_context = {}
        for baro_name, baro_df in barometer_data.items():
            baro_train, baro_val = split_periods(baro_df)
            train_context[baro_name] = baro_train
            val_context[baro_name] = baro_val

        # Generuj sygnały — strategia dostaje dane krypto + kontekst makro
        train_signals = strategy_fn(train_df, train_context)
        val_signals = strategy_fn(val_df, val_context)

        # Backtest
        train_metrics = backtest(train_df, train_signals)
        val_metrics = backtest(val_df, val_signals)

        # Score
        score = compute_score(train_metrics, val_metrics)
        scores.append(score)

        results[symbol] = {
            "train": train_metrics,
            "val": val_metrics,
            "score": score,
        }

        print(f"  {symbol}:")
        print(f"    Train — Sharpe: {train_metrics['sharpe']:>7.3f}  "
              f"Return: {train_metrics['total_return']:>8.2%}  "
              f"MaxDD: {train_metrics['max_drawdown']:>7.2%}  "
              f"Trades: {train_metrics['num_trades']}  "
              f"Long/Short/Flat: {train_metrics['long_pct']:.0%}/"
              f"{train_metrics['short_pct']:.0%}/"
              f"{train_metrics['flat_pct']:.0%}")
        print(f"    Val   — Sharpe: {val_metrics['sharpe']:>7.3f}  "
              f"Return: {val_metrics['total_return']:>8.2%}  "
              f"MaxDD: {val_metrics['max_drawdown']:>7.2%}  "
              f"Trades: {val_metrics['num_trades']}  "
              f"Long/Short/Flat: {val_metrics['long_pct']:.0%}/"
              f"{val_metrics['short_pct']:.0%}/"
              f"{val_metrics['flat_pct']:.0%}")
        print(f"    B&H:  Train {train_metrics['buy_hold_return']:>8.2%}  "
              f"Val {val_metrics['buy_hold_return']:>8.2%}")
        print(f"    Score: {score}")
        print()

    avg_score = round(np.mean(scores), 4) if scores else 0.0

    # Output grep-owalny (dla agenta)
    print(f"score:        {avg_score}")
    print(f"assets:       {len(scores)}")

    results["_avg_score"] = avg_score
    return results


# ─── Wizualizacja ───────────────────────────────────────────────

def plot_equity(results: dict, save_path: str = "equity.png"):
    """Rysuje krzywe equity vs buy & hold (train + val) per krypto."""
    assets = [k for k in results if not k.startswith("_")]
    if not assets:
        print("Brak danych do wykresu.")
        return

    fig, axes = plt.subplots(len(assets), 1, figsize=(14, 5 * len(assets)))
    if len(assets) == 1:
        axes = [axes]

    for ax, symbol in zip(axes, assets):
        r = results[symbol]

        for period, label, color in [("train", "Train", "#2196F3"), ("val", "Val", "#FF9800")]:
            eq = r[period]["equity_curve"]
            ax.plot(eq.index, eq.values, label=f"Strategia ({label})", color=color)

        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"{symbol} — Score: {r['score']}")
        ax.set_ylabel("Equity")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    print(f"\nWykres zapisano → {save_path}")
    plt.close()


if __name__ == "__main__":
    # Tryb standalone: pobierz dane i wyświetl info
    crypto_data, baro_data = load_all_data("1h")

    print("─── KRYPTO (handlowane) ───")
    for symbol, df in crypto_data.items():
        train, val = split_periods(df)
        print(f"  {symbol}: {len(df)} świec 1h, {len(train)} train, {len(val)} val")
        print(f"    Zakres: {df.index[0]} → {df.index[-1]}")
        print(f"    Cena: {df['close'].iloc[0]:.2f} → {df['close'].iloc[-1]:.2f}")

    print("\n─── BAROMETRY (kontekst) ───")
    for symbol, df in baro_data.items():
        train, val = split_periods(df)
        print(f"  {symbol}: {len(df)} świec 1h, {len(train)} train, {len(val)} val")
        print(f"    Zakres: {df.index[0]} → {df.index[-1]}")
        print(f"    Cena: {df['close'].iloc[0]:.2f} → {df['close'].iloc[-1]:.2f}")
