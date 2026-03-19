# Program agenta — Autoquant (Faza 3)

Jesteś autonomicznym agentem optymalizującym strategie tradingowe na krypto.

## Twoje zadanie

Modyfikuj `strategy.py` aby maksymalizować metrykę `score` (wyższy = lepiej).
Po każdej modyfikacji uruchom `python3 strategy.py` i sprawdź wynik.

## Zasady

1. Modyfikuj **wyłącznie** `strategy.py` — `prepare.py` jest read-only
2. Nie dodawaj nowych zależności (pip install). Dostępne: torch, pandas, numpy
3. Każdy eksperyment: `python3 strategy.py > logi/run_NNN.log 2>&1` (NNN = numer z results.tsv)
4. Wyniki zapisują się automatycznie do `results.tsv`
5. Zmień zmienną `OPIS` w strategy.py na krótki opis co zmieniłeś

## Protokół eksperymentu

1. Przeczytaj `strategy.py`, `results.tsv` i NINIEJSZY plik (`program.md`)
2. Wymyśl ulepszenie (NIE powtarzaj tego co jest w sekcji "Nie powtarzać")
3. Zmodyfikuj `strategy.py` i zmień `OPIS`
4. Uruchom: `python3 strategy.py`
5. Sprawdź `score:` w outputcie
6. Jeśli score > aktualny rekord (3.456) → zachowaj, zaktualizuj `program.md`
7. Jeśli score gorszy → cofnij zmiany, idź do kroku 2
8. Powtarzaj

---

## Aktualny rekord — exp #109 (2026-03-19)

**Score: 3.4564**
**Konfiguracja:** `LSTM_h384_3L_drop03_target24h_discrete_funding_1H`

```
Model:    LSTM 3-warstwowy (hidden=384) + BatchNorm + dropout=0.3
Target:   close.pct_change(24).shift(-24)   ← 24h forward return
Lookback: 168 świec (7 dni na 1H)
Trening:  300 epok, lr=0.002, AdamW wd=0.02, CosineAnnealingLR, batch=512
Features: 23 (21 technicznych + market_funding + vixy_trend)
Predykcja: predict_lstm_confidence() — TYLKO train_valid_idx (ostatnie 20% train)
Sygnały:  dyskretne progi: >0.15→0.5, >0.35→0.75, >0.55→1.0
          short: <-0.15→-0.5, <-0.55→-1.0
ATR stop: multiplier=1.9, cooldown=24, profit_target_atr=3.0
Seed:     SINGLE_SEED=42, okno train=80%
```

**Per-asset (exp #109):**

| Asset | Val Sharpe | Val Return | Score |
|-------|-----------|-----------|-------|
| BTC   | 3.415 | +74% | 1.79 |
| ETH   | 5.559 | +368% | 3.79 |
| XMR   | 7.707 | +1402% | 3.64 |
| SOL   | 4.147 | +254% | 2.41 |
| TAO   | 6.976 | +1702% | 5.66 |

BTC jest najsłabszy — val period 2025-03→2026-03 był bessą dla BTC (B&H = -10.80%).

---

## ⚠️ Ostrzeżenie: metryka score i krótkie horyzonty

Score NIE jest ograniczony górnie — składnik `15% × return` eksploduje przy krótkich targetach:
- target=12h → score 27.4, ~2000 trades/asset/rok
- target=6h → score **4982**, SOL train zwrot **141 bilionów %**

To artefakt bezkosztowego backtestu + compounding. W produkcji: 3000 trades × 0.1% fee = 300% rocznych opłat → niezgrywalny.

**Granica sensowności: target ≥ 24h (≤ 1200 trades/asset/rok)**

---

## 🚫 NIE POWTARZAJ — zbadane i odrzucone

### Architektura modelu

| Zmiana | Wyniki exp | Dlaczego nie |
|--------|-----------|-------------|
| h=512 (2L lub 3L) | #102, #104, #113 | Zawsze destabilizuje ≥1 asset. ETH/XMR collapse lub overfit |
| 4 warstwy (h=384 4L) | #114, score 3.425 | Marginalnie gorszy od 3L |
| dropout=0.5 | #106, score 1.129 | ETH idzie flat (4% long, 96% flat) |
| dropout=0.4 przy target=24h | #108 vs #109 | 0.3 lepsze — krótszy target to naturalna regularyzacja |
| Per-asset ensemble (różne seedy per asset) | #94, score 1.600 | Niszczy cross-asset korelacje |
| Ensemble 3 seedów (przy h=384 3L target=24h) | #112, score 3.447 | Brak poprawy vs single seed, 3× koszt GPU |

### Hiperparametry treningu

| Zmiana | Wyniki exp | Dlaczego nie |
|--------|-----------|-------------|
| lr=0.001 | #115, score 2.236 | Underfitting — model nie converge w 300 epokach |
| 500 epok | #116, score 3.207 | Overfit — gorszy niż 300 |
| wd=0.05 | #118, score 3.381 | Marginalna różnica vs 0.02 |

### Dane i sygnały

| Zmiana | Wyniki exp | Dlaczego nie |
|--------|-----------|-------------|
| LOOKBACK=96 | #107, score 1.474 | BTC overfit — train Sharpe 3.77 vs val 0.27 |
| LOOKBACK=240 | #119, score 3.086 | SOL/XMR słabiej, gorsze niż 168 |
| target=12h | #110, score 27.4 | Artefakt metryki (nierealne) |
| target=6h | #111, score 4982 | Całkowicie nierealny |
| target=48h | #93–#107 | 24h dramatycznie lepsze (+58%) |
| Progi 0.10/0.30/0.50 | #117, score 3.076 | Więcej false trades, BTC spada |
| Walkforward (70%+80% okna) | #97, score 1.479 | Słabszy model 70% + oscylacje progów → 2-3× więcej trades |
| predict_on_data() (pełne OOS) | #98, score 1.526 | Noisy predykcje w OOS → za dużo trades |
| Ciągłe pozycjonowanie (bez progów) | #95, 3949 trades | 5× więcej trades vs dyskretne progi |
| Per-asset funding (3 osobne features) | #100, score 1.465 | XMR nie ma własnych futures → szum dla XMR/TAO |

---

## ✅ Co warto próbować dalej

### Architektura (priorytet wysoki)
- **BiLSTM** — dwukierunkowy LSTM, może lepiej uchwycić długoterminowe zależności
- **GRU** zamiast LSTM — mniej parametrów, często porównywalny wynik
- **h=256 3L** — mniejszy model, bardziej regularny niż h=384

### Features (priorytet średni)
- **OBV** (On-Balance Volume) — wolumen-momentum
- **Stochastic RSI** — bardziej czuły oscylator
- **Spread 10Y-2Y** yield curve — `context['TREASURY_10Y'] - context['TREASURY_2Y']`
- **NEWS sentiment** — `context['NEWS_BTC']`, `context['NEWS_ETH']` (daily)
- **Williams %R** — momentum

### ATR i zarządzanie pozycją (priorytet średni)
- **multiplier=2.5** — luźniejszy stop, więcej przestrzeni
- **profit_target_atr=4.0 lub 5.0** — dłuższe trzymanie zwycięzców
- **cooldown=12** — szybszy re-entry

### Target i timing (priorytet niski)
- **target=36h** — między 24h a 48h, może sweet spot
- **Inny seed niż 42** — sprawdzenie czy 42 jest globalnym optimum

---

## Ewolucja wyników (historia)

```
Faza 1 (rule-based):   0.408
MLP (sieć neuronowa):  1.064
LSTM 2L h=256 lb=168:  1.903  (#93, 2026-03-17)
LSTM h=384 2L:         1.948  (#101, 2026-03-18)
LSTM h=384 3L:         2.116  (#103, 2026-03-19 00:18)
+dropout=0.4:          2.188  (#105, 2026-03-19 01:36)
+target=24h:           3.341  (#108, 2026-03-19 02:54)
+dropout=0.3:          3.456  (#109, 2026-03-19 03:29)  ← REKORD
```

---

## Kluczowe odkrycia techniczne

1. **BatchNorm obowiązkowy** — skok 0.60→0.80, bez niego LSTM nie stabilizuje się
2. **LSTM >> MLP** na time series krypto (+79%)
3. **1H >> 4H** — 4× więcej danych treningowych, lepszy signal-to-noise
4. **predict_lstm_confidence() >> predict_on_data()** — OOS predykcje są noisy
5. **Interakcja dropout×target**: krótszy target → mniej regularizacji potrzeba
6. **h=384 sweet spot** — h=256 za mały, h=512 niestabilny dla wieloassetowego modelu
7. **market_funding** (avg FR_BTC+ETH+SOL) >> per-asset funding (XMR/TAO nie mają własnych futures)
8. **_strip_tz() konieczny** — funding rate ma UTC timezone, OHLCV nie ma TZ

---

## Kontekst danych w strategy(df, context)

| Klucz | Typ | Użycie |
|-------|-----|--------|
| `FR_BTC_`, `FR_ETH_`, `FR_SOL_` | co 8h, kolumna "close" | market_funding (avg) |
| `VIXY` | ETF 1h | vixy_trend = odchylenie od MA20 |
| `SPY`, `QQQ`, `UUP`, `GLD` | ETF 1h | kontekst makro |
| `FED_RATE`, `CPI` | miesięczny | kontekst makro |
| `TREASURY_10Y`, `TREASURY_2Y` | dzienny | yield curve |
| `NEWS_BTC`, `NEWS_ETH` | dzienny | sentyment (dotąd nieużywany w features) |

**Uwaga TZ:** funding rate ma UTC timezone → użyj `_strip_tz()` przed reindex.

---

## Ograniczenia systemu

- Modyfikuj **wyłącznie** `strategy.py` (prepare.py jest read-only)
- Nie dodawaj nowych zależności
- Sygnatura: `strategy(df, context) -> pd.Series` (+1 long, -1 short, 0 flat)
- Wyniki logowane automatycznie do `results.tsv`
- Wszystkie pliki i komentarze w języku polskim
- evaluate() wywołuje strategy() **10 razy** (5 assetów × 2 okresy: train i val)
- Logi z eksperymentów w folderze `logi/` (format: `logi/run_NNN.log`)

## live_signals.py

Generuje sygnały live bez ponownego treningu — ładuje modele z dysku.

```bash
python3 live_signals.py              # jednorazowo
python3 live_signals.py --loop       # pętla co godzinę
python3 live_signals.py --loop --telegram  # + powiadomienia Telegram
```

Modele cache: `~/.cache/autoquant/best_model/lstm_{asset}_s42.pt` (5 plików ~12MB każdy).
Po każdym `python3 strategy.py` modele są automatycznie aktualizowane.

**Uwaga `_asset_id()`:** SOL w val period (bessa) ma vol < XMR — identyfikacja po vol+cenie:
`vol < 0.011 AND median > 155` → xmr | `median > 150` → tao | reszta → sol
