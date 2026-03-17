# Program agenta — Autoquant (Faza 2)

Jesteś autonomicznym agentem optymalizującym strategie tradingowe na krypto.

## Twoje zadanie

Modyfikuj `strategy.py` aby maksymalizować metrykę `score` (wyższy = lepiej).
Po każdej modyfikacji uruchom `python3 strategy.py` i sprawdź wynik.

## Zasady

1. Modyfikuj **wyłącznie** `strategy.py` — `prepare.py` jest read-only
2. Nie dodawaj nowych zależności (pip install). Dostępne: torch, pandas, numpy
3. Każdy eksperyment musi się zakończyć w ~2 min
4. Wyniki zapisują się automatycznie do `results.tsv`
5. Zmień zmienną `OPIS` w strategy.py na krótki opis co zmieniłeś

## Protokół eksperymentu

1. Przeczytaj aktualny `strategy.py` i `results.tsv`
2. Wymyśl ulepszenie
3. Zmodyfikuj `strategy.py`
4. Uruchom: `python3 strategy.py`
5. Sprawdź `score:` w outputcie
6. Jeśli score lepszy niż poprzedni najlepszy → zachowaj, idź do kroku 2
7. Jeśli score gorszy → cofnij zmiany (`git checkout strategy.py`), idź do kroku 2
8. Powtarzaj w nieskończoność

## Obecny baseline (po Fazie 1 — 43 eksperymenty)

- Score: **0.4082**
- Strategia: Dual MACD (12/26 + 8/17) + Ichimoku + EMA200 + ATR 1.9 trailing stop + cooldown 6 + profit target 3×ATR + chikou konfirmacja
- Timeframe: 4H
- Aktywa: BTC, ETH, XMR, SOL, TAO (long/short)
- Barometry ETF: SPY, QQQ, UUP, GLD (złoto), VIXY (VIX/strach)
- Makro: FED_RATE (stopa Fed), CPI (inflacja), TREASURY_10Y, TREASURY_2Y
- Wszystkie dostępne w `context` dict — używaj ich!

### Per-asset wyniki (problem!):
- **SOL: 0.975** — doskonały, nie psuj
- **ETH: 0.642** — dobry, nie psuj
- **XMR: 0.176** — słaby, -87% na train
- **TAO: 0.154** — słaby, overfit
- **BTC: 0.095** — BARDZO SŁABY, strategia nie łapie bull runów BTC

### Co zadziałało w Fazie 1:
- EMA200 filter, ATR 1.9, Dual MACD, cooldown 6, profit target 3×ATR, chikou

### Co NIE zadziałało w Fazie 1:
- Stochastic, ADX, Bollinger, volume filter, crypto Ichimoku (7/22/44), long-only, ensemble voting, QQQ macro

## FAZA 2 — PLAN ATAKU (wykonuj w tej kolejności!)

### Krok 1: Per-asset parametry (szybki zysk)

BTC ma score 0.095 — to OGROMNY potencjał. Strategia działa świetnie na SOL/ETH ale słabo na BTC.

Rozwiązanie: **różne parametry per asset**. W `strategy()` sprawdzaj jaki to asset (po cenach — BTC > 10000, SOL < 500, itd. lub dodaj parametr) i stosuj inne:
- BTC: może potrzebuje dłuższych MA, łagodniejszego ATR stop, innego MACD
- XMR: specyficzny asset, Monero ma inną dynamikę (privacy coin)
- TAO: AI token, bardzo młody, mała historia — może uproszczona strategia

Nie psuj SOL i ETH! Testuj zmiany per-asset.

### Krok 2: Sieci neuronowe PyTorch (GPU RTX 4090 24GB)

Po wyczerpaniu per-asset tuningu, przejdź do modeli neuronowych.

Architektura hybrydowa:
1. Wskaźniki techniczne jako **features** (close_pct, rsi, macd_hist, atr_norm, volume_ratio, cloud_dist, tenkan_kijun_diff, spy_trend, uup_trend)
2. **Mały model neuronowy** jako decydent — uczy się na train, generuje sygnały na val
3. ATR trailing stop nadal chroni pozycje

Modele do wypróbowania:
- **LSTM/GRU** (lookback 20-50 świec) — klasyka dla time series
- **1D CNN** — szybki, łapie lokalne wzorce
- **Transformer** — attention, dobre na dłuższe zależności
- **Ensemble** — kilka małych modeli głosuje

Implementacja:
```python
import torch
import torch.nn as nn
device = torch.device("cuda")  # RTX 4090

# Mały model — 1-3 warstwy, 32-128 neuronów
# Regularyzacja: dropout=0.3, weight_decay=1e-4, early stopping
# Normalizacja: StandardScaler fit na train, transform na val (BEZ data leakage!)
# Output: sygnał [-1, 1] (continuous), threshold na pozycje
# Trenuj PER-ASSET (każdy asset ma swój model)
```

### Nowe dane makro w context (UŻYWAJ ICH!)

Agent Fazy 1 nie miał tych danych. Ty masz. Klucze w `context`:

| Klucz | Typ | Co to | Jak użyć |
|-------|-----|-------|----------|
| `GLD` | ETF 1h | Złoto | Korelacja z BTC (store of value). GLD rośnie = pozytywne dla krypto |
| `VIXY` | ETF 1h | VIX proxy | Strach. VIXY rośnie = risk-off = krypto spada. VIXY spada = risk-on = krypto rośnie |
| `FED_RATE` | miesięczny | Stopa Fed | Rosnąca = hawkish = krypto spada. Malejąca = dovish = krypto rośnie. Reindex ffill do 4H |
| `CPI` | miesięczny | Inflacja | Rosnąca = Fed zaostrza = źle dla krypto. Reindex ffill do 4H |
| `TREASURY_10Y` | dzienny | Yield 10Y | Rosnący yield = presja na ryzykowne aktywa. Kluczowy driver! |
| `TREASURY_2Y` | dzienny | Yield 2Y | Yield curve (10Y-2Y). Inwersja = recesja = risk-off |

Przykład użycia makro:
```python
if "TREASURY_10Y" in context and len(context["TREASURY_10Y"]) > 50:
    t10y = context["TREASURY_10Y"]["close"]
    t10y = t10y.reindex(df.index, method="ffill")  # forward-fill do 4H
    t10y_rising = t10y.diff(20) > 0  # yieldy rosną w ostatnich 20 barach
    # Gdy yieldy rosną → osłab longi lub wzmocnij shorty
```

### Krok 3: Multi-timeframe (jeśli zostanie czas)

Łączenie sygnałów z 1H (precyzyjne wejścia) i 4H (kierunek trendu).

## Czego NIE zmieniać

- Sygnatura `strategy(df, context) -> pd.Series`
- Import z prepare.py
- Sekcja Runner (if __name__ == "__main__")
- Zapis do results.tsv

## Metryka score

Composite: 35% Sharpe + 20% Sortino + 20% (1-drawdown) + 15% return + 10% win_rate
× trade_penalty (min 50 transakcji) × consistency (train vs val)

Ważne: score musi być dobry na VALIDATION (2025-03 do 2026-03), nie tylko train.
Unikaj overfittingu — duża różnica train vs val to zły znak.
