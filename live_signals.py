"""
live_signals.py — generator sygnałów tradingowych w czasie rzeczywistym

Pobiera świeże dane, uruchamia strategię LSTM, wyświetla sygnały.

Uruchomienie jednorazowe:
    python3 live_signals.py

Uruchomienie w pętli co godzinę:
    python3 live_signals.py --loop

Z powiadomieniami Telegram:
    python3 live_signals.py --loop --telegram

Konfiguracja Telegram (opcjonalnie):
    Ustaw w .env:
    TELEGRAM_BOT_TOKEN=twoj_token
    TELEGRAM_CHAT_ID=twoj_chat_id
"""

import sys
import time
import os
import argparse
import requests as req
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# Załaduj .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# Import strategii i narzędzi z prepare/strategy
from prepare import (
    CRYPTO_ASSETS, BAROMETER_ASSETS, EXCHANGE_OVERRIDES,
    MACRO_INDICATORS, FUTURES_ASSETS,
    download_crypto, download_barometer, download_macro,
    download_funding_rate, resample_to_4h, _get_exchange
)
from strategy import strategy, atr, LOOKBACK

# ─── Konfiguracja ────────────────────────────────────────────────

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

BEST_MODEL_DIR = Path.home() / ".cache" / "autoquant" / "best_model"

# Kolory ANSI
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
BG_GREEN = "\033[42;30m"
BG_RED = "\033[41;97m"
BG_YELLOW = "\033[43;30m"


# ─── Pobieranie świeżych danych ─────────────────────────────────

def fetch_fresh_crypto(symbol: str, limit: int = 500) -> pd.DataFrame:
    """Pobiera najnowsze świece 1H z giełdy (bez cache)."""
    exchange, exchange_id = _get_exchange(symbol)

    candles = exchange.fetch_ohlcv(symbol, "1h", limit=limit)
    if not candles:
        return pd.DataFrame()

    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp")
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    return df


def load_historical_and_fresh(symbol: str) -> pd.DataFrame:
    """
    Łączy dane historyczne (cache) z najnowszymi świecami.
    Dzięki temu LSTM ma pełną historię do treningu + najnowszy sygnał.
    """
    # Historyczne z cache
    try:
        hist = download_crypto(symbol)
    except Exception:
        hist = pd.DataFrame()

    # Świeże (ostatnie 500 świec)
    try:
        fresh = fetch_fresh_crypto(symbol, limit=500)
    except Exception as e:
        print(f"  ⚠ Nie można pobrać świeżych danych {symbol}: {e}")
        return hist

    if hist.empty:
        return fresh
    if fresh.empty:
        return hist

    # Połącz — świeże nadpisują stare
    combined = pd.concat([hist, fresh])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    return combined


def load_barometers() -> dict[str, pd.DataFrame]:
    """Ładuje barometry z cache (ETF + makro + funding rate)."""
    baro = {}

    for symbol in BAROMETER_ASSETS:
        try:
            baro[symbol] = download_barometer(symbol)
        except Exception:
            pass

    for name, config in MACRO_INDICATORS.items():
        try:
            df = download_macro(name, config)
            baro[name] = pd.DataFrame({"close": df["value"]})
        except Exception:
            pass

    for symbol in FUTURES_ASSETS:
        try:
            fr_df = download_funding_rate(symbol)
            safe = symbol.replace("/", "_").replace("USDT", "")
            if not fr_df.empty:
                baro[f"FR_{safe}"] = pd.DataFrame({"close": fr_df["funding_rate"]})
        except Exception:
            pass

    return baro


# ─── Interpretacja sygnału ───────────────────────────────────────

def interpret_signal(signal: float) -> tuple[str, str, str]:
    """
    Zamienia surowy sygnał (-1 do +1) na czytelną rekomendację.
    Zwraca: (akcja, opis, kolor)
    """
    if signal >= 0.75:
        return "LONG PEŁNY", "Silny sygnał kupna — pełna pozycja", BG_GREEN
    elif signal >= 0.5:
        return "LONG ¾", "Umiarkowany sygnał kupna — ¾ pozycji", GREEN
    elif signal >= 0.25:
        return "LONG ½", "Słaby sygnał kupna — ½ pozycji", GREEN
    elif signal > 0.05:
        return "LONG LEKKI", "Minimalny sygnał kupna — mała pozycja", DIM + GREEN
    elif signal > -0.05:
        return "CZEKAJ", "Brak wyraźnego sygnału — nie wchodź", YELLOW
    elif signal > -0.25:
        return "SHORT LEKKI", "Minimalny sygnał sprzedaży", DIM + RED
    elif signal > -0.5:
        return "SHORT ½", "Umiarkowany sygnał sprzedaży — ½ pozycji", RED
    elif signal > -0.75:
        return "SHORT ¾", "Silniejszy sygnał sprzedaży — ¾ pozycji", RED
    else:
        return "SHORT PEŁNY", "Silny sygnał sprzedaży — pełna pozycja", BG_RED


# ─── Wyświetlanie ───────────────────────────────────────────────

def display_signals(signals_data: list[dict]):
    """Wyświetla czytelny panel sygnałów."""
    W = 80
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    print()
    print(f"{BOLD}{CYAN}{'═' * W}{RESET}")
    print(f"{BOLD}{CYAN}  AUTOQUANT — SYGNAŁY LIVE   {now}{RESET}")
    print(f"{BOLD}{CYAN}{'═' * W}{RESET}")
    print()

    for s in signals_data:
        action, desc, color = interpret_signal(s["signal"])
        price = s["price"]
        change_1h = s.get("change_1h", 0)
        change_24h = s.get("change_24h", 0)
        atr_val = s.get("atr", 0)

        ch1_color = GREEN if change_1h >= 0 else RED
        ch24_color = GREEN if change_24h >= 0 else RED

        print(f"  {BOLD}{s['symbol']:<12}{RESET} "
              f"Cena: {BOLD}${price:>10,.2f}{RESET}  "
              f"1h: {ch1_color}{change_1h:>+6.2f}%{RESET}  "
              f"24h: {ch24_color}{change_24h:>+6.2f}%{RESET}")

        print(f"  {'':12} "
              f"{color} {action:^14} {RESET}  "
              f"{DIM}{desc}{RESET}")

        # Stop loss / take profit
        if abs(s["signal"]) > 0.05 and atr_val > 0:
            if s["signal"] > 0:
                sl = price - 1.9 * atr_val
                tp = price + 3.0 * atr_val
                print(f"  {'':12} "
                      f"{DIM}Stop Loss: ${sl:,.2f} ({(sl/price-1)*100:+.1f}%)  "
                      f"Take Profit: ${tp:,.2f} ({(tp/price-1)*100:+.1f}%){RESET}")
            elif s["signal"] < 0:
                sl = price + 1.9 * atr_val
                tp = price - 3.0 * atr_val
                print(f"  {'':12} "
                      f"{DIM}Stop Loss: ${sl:,.2f} ({(sl/price-1)*100:+.1f}%)  "
                      f"Take Profit: ${tp:,.2f} ({(tp/price-1)*100:+.1f}%){RESET}")

        print(f"  {'─' * (W - 4)}")

    # Podsumowanie
    longs = sum(1 for s in signals_data if s["signal"] > 0.05)
    shorts = sum(1 for s in signals_data if s["signal"] < -0.05)
    waiting = len(signals_data) - longs - shorts

    print()
    print(f"  {BOLD}Podsumowanie:{RESET}  "
          f"{GREEN}▲ {longs} long{RESET}  "
          f"{RED}▼ {shorts} short{RESET}  "
          f"{YELLOW}─ {waiting} czekaj{RESET}")
    print()

    # Ogólny sentyment
    avg_signal = np.mean([s["signal"] for s in signals_data])
    if avg_signal > 0.2:
        mood = f"{GREEN}BULLISH — rynek sprzyja longom{RESET}"
    elif avg_signal > 0.05:
        mood = f"{GREEN}LEKKO BULLISH{RESET}"
    elif avg_signal > -0.05:
        mood = f"{YELLOW}NEUTRALNY — brak wyraźnego kierunku{RESET}"
    elif avg_signal > -0.2:
        mood = f"{RED}LEKKO BEARISH{RESET}"
    else:
        mood = f"{RED}BEARISH — rynek sprzyja shortom{RESET}"

    print(f"  {BOLD}Sentyment rynku:{RESET} {mood}")
    print(f"{BOLD}{CYAN}{'═' * W}{RESET}")
    print(f"  {DIM}Następna aktualizacja za ~60 minut{RESET}")
    print(f"  {DIM}UWAGA: To nie jest porada inwestycyjna. Handluj na własne ryzyko.{RESET}")
    print()


# ─── Telegram ────────────────────────────────────────────────────

def send_telegram(signals_data: list[dict]):
    """Wysyła sygnały na Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    now = datetime.now().strftime("%H:%M")
    all_flat = all(abs(s["signal"]) <= 0.05 for s in signals_data)

    if all_flat:
        # Krótkie podsumowanie gdy same FLAT
        prices = "  ".join(
            f"{s['symbol']}: ${s['price']:,.0f} ({s['change_24h']:+.1f}%)"
            for s in signals_data
        )
        text = f"⏳ *AUTOQUANT {now}* — wszystkie FLAT\n`{prices}`"
    else:
        lines = [f"🤖 *AUTOQUANT — {now}*\n"]

        for s in signals_data:
            action, desc, _ = interpret_signal(s["signal"])
            emoji = "🟢" if s["signal"] > 0.05 else "🔴" if s["signal"] < -0.05 else "⏳"
            price = s["price"]

            line = f"{emoji} *{s['symbol']}* — {action}\n"
            line += f"   `Cena:` ${price:,.2f}"

            if abs(s["signal"]) > 0.05 and s.get("atr", 0) > 0:
                atr_val = s["atr"]
                if s["signal"] > 0:
                    sl = price - 1.9 * atr_val
                    tp = price + 3.0 * atr_val
                else:
                    sl = price + 1.9 * atr_val
                    tp = price - 3.0 * atr_val
                line += f"\n   `SL:` ${sl:,.2f} ({(sl/price-1)*100:+.1f}%)"
                line += f"\n   `TP:` ${tp:,.2f} ({(tp/price-1)*100:+.1f}%)"

            lines.append(line)

        avg = np.mean([s["signal"] for s in signals_data])
        if avg > 0.1:
            lines.append("\n📈 Sentyment: BULLISH")
        elif avg < -0.1:
            lines.append("\n📉 Sentyment: BEARISH")
        else:
            lines.append("\n➡️ Sentyment: NEUTRALNY")

        text = "\n".join(lines)

    try:
        req.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "Markdown",
            },
            timeout=10,
        )
        print(f"  {DIM}Telegram: wysłano ✓{RESET}")
    except Exception as e:
        print(f"  ⚠ Telegram error: {e}")


# ─── Główna pętla ────────────────────────────────────────────────

def generate_signals(send_tg: bool = False) -> list[dict]:
    """Generuje sygnały dla wszystkich assetów."""
    print(f"\n{DIM}Ładowanie danych...{RESET}")

    # Załaduj barometry (z cache)
    barometers = load_barometers()

    signals_data = []

    for symbol in CRYPTO_ASSETS:
        try:
            print(f"  {DIM}Przetwarzam {symbol}...{RESET}", end=" ", flush=True)

            # Pobierz dane (historyczne + świeże)
            df = load_historical_and_fresh(symbol)
            if df.empty or len(df) < 500:
                print(f"{RED}za mało danych{RESET}")
                continue

            # Przygotuj kontekst barometrów
            context = {}
            for baro_name, baro_df in barometers.items():
                context[baro_name] = baro_df

            # Uruchom strategię z najlepszymi modelami agenta (trenuje tylko jeśli brak plików)
            signals = strategy(df, context, model_cache_dir=BEST_MODEL_DIR,
                               model_retrain_hours=99999)

            # Ostatni sygnał = aktualny
            last_signal = signals.iloc[-1]
            last_price = df["close"].iloc[-1]

            # Zmiany cenowe
            change_1h = (df["close"].iloc[-1] / df["close"].iloc[-2] - 1) * 100 if len(df) > 1 else 0
            change_24h = (df["close"].iloc[-1] / df["close"].iloc[-25] - 1) * 100 if len(df) > 25 else 0

            # ATR dla stop loss / take profit
            atr_val = atr(df, 14).iloc[-1]

            signals_data.append({
                "symbol": symbol.replace("/USDT", ""),
                "signal": float(last_signal),
                "price": float(last_price),
                "change_1h": float(change_1h),
                "change_24h": float(change_24h),
                "atr": float(atr_val),
            })

            action, _, _ = interpret_signal(float(last_signal))
            print(f"{action}")

        except Exception as e:
            print(f"{RED}błąd: {e}{RESET}")

    # Wyświetl
    if signals_data:
        display_signals(signals_data)

        if send_tg:
            send_telegram(signals_data)

    return signals_data


def main():
    parser = argparse.ArgumentParser(description="Autoquant — generator sygnałów live")
    parser.add_argument("--loop", action="store_true", help="Uruchom w pętli co godzinę")
    parser.add_argument("--telegram", action="store_true", help="Wysyłaj powiadomienia Telegram")
    parser.add_argument("--interval", type=int, default=3600, help="Interwał w sekundach (domyślnie 3600)")
    args = parser.parse_args()

    print(f"{BOLD}{CYAN}AUTOQUANT — Generator sygnałów{RESET}")
    print(f"{DIM}Strategia: LSTM Hybrid + ATR trailing stop{RESET}")
    print(f"{DIM}Assetów: {len(CRYPTO_ASSETS)} | Timeframe: 1H{RESET}")

    if args.loop:
        print(f"{DIM}Tryb pętli: co {args.interval//60} minut{RESET}")
        while True:
            try:
                generate_signals(send_tg=args.telegram)

                # Czekaj do następnej pełnej godziny
                now = datetime.now()
                seconds_to_next = args.interval - (now.minute * 60 + now.second) % args.interval
                print(f"\n{DIM}Następna aktualizacja za {seconds_to_next//60} min...{RESET}")
                time.sleep(seconds_to_next)

            except KeyboardInterrupt:
                print(f"\n{YELLOW}Zatrzymano.{RESET}")
                break
            except Exception as e:
                print(f"\n{RED}Błąd: {e}{RESET}")
                time.sleep(60)
    else:
        generate_signals(send_tg=args.telegram)


if __name__ == "__main__":
    main()
