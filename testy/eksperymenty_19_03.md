# Eksperymenty autoquant — 2026-03-19 (sesja ~11h, GPU RTX 4090)

## Punkt startowy

- **Rekord wejściowy:** 2.116 (exp #103, LSTM h=384 3L, target 48h, dropout=0.3)
- **Rekord wyjściowy:** **3.456** (exp #109, target=24h)
- **Łączny czas GPU:** ~11 godzin (00:18–11:18 UTC)

---

## Tabela wszystkich eksperymentów

| # | Score | Opis zmiany | Wynik |
|---|-------|-------------|-------|
| 103 | 2.116 | LSTM h=384 3L drop=0.3 target=48h | ← baseline tej sesji |
| 104 | 1.623 | h=512 3L (test szerszego modelu) | ❌ SOL val Sharpe 0.533 (overfit) |
| 105 | 2.188 | dropout 0.3→**0.4** przy h=384 3L | ✅ naprawił SOL (val Sharpe 0.533→3.783) |
| 106 | 1.129 | dropout 0.4→**0.5** | ❌ za dużo: ETH idzie flat (4% long, 96% flat) |
| 107 | 1.474 | LOOKBACK 168→**96** | ❌ BTC overfit (train 3.766 vs val 0.271) |
| 108 | 3.341 | target 48h→**24h** (drop=0.4) | ✅✅ OGROMNY SKOK +1.15 |
| **109** | **3.456** | target=24h + dropout 0.4→**0.3** | ✅ **NOWY REKORD** |
| 110 | 27.4 | target **12h** | ⚠️ artefakt metryki (~2000 trades/asset) |
| 111 | 4982 | target **6h** | ⚠️⚠️ SOL train: 141 bilionów% — całkowicie nierealny |
| 112 | 3.447 | ensemble **3 seedów** (42/137/999) przy target=24h | ≈ brak poprawy, 3× dłużej |
| 113 | 2.867 | h=**512** 3L target=24h | ❌ ETH train Sharpe -0.49 (collapse) |
| 114 | 3.425 | h=384 **4 warstwy** target=24h | ≈ marginalnie gorszy od 3L |
| 115 | 2.236 | LR **0.001** (zamiast 0.002) | ❌ underfitting — model nie converge w 300 ep |
| 116 | 3.207 | **500 epok** (zamiast 300) | ❌ overfit — gorszy niż 300 |
| 117 | 3.076 | progi **0.10/0.30/0.50** (zamiast 0.15/0.35/0.55) | ❌ więcej false trades, BTC spada |
| 118 | 3.381 | wd=**0.05** (zamiast 0.02) | ≈ marginalna różnica |
| 119 | 3.086 | LOOKBACK=**240** (10 dni) | ❌ SOL/XMR słabiej |

---

## Kluczowe odkrycia

### 1. Target horizon to najważniejszy parametr
- 48h → 24h: skok z 2.116 → 3.341 (+58%)
- Krótszy target → model uczy się bardziej lokalnych, powtarzalnych wzorców
- Generalizacja train→val dramatycznie lepsza

### 2. Interakcja dropout × target
- Przy target=48h: dropout=0.4 >> 0.3 (SOL się overfitował przy 0.3)
- Przy target=24h: dropout=0.3 >> 0.4 (krótszy horyzont to naturalna regularyzacja)
- Logika: krótszy target = łatwiejszy problem = mniej regularizacji potrzeba

### 3. h=384 to definitywny sweet spot
- h=512 przy 2L → XMR consistency=0 (train Sharpe 1.39 vs val 5.09)
- h=512 przy 3L → SOL overfit (score 1.623 vs 2.116 przy h=384)
- h=512 przy 3L target=24h → ETH train Sharpe -0.49 (collapse)
- h=384 nigdy nie zawodziło, h=512 zawsze powodowało problemy z co najmniej jednym assetem

### 4. 3 warstwy to sweet spot głębokości
- 2L → 3L: +0.20 (2.116 vs 1.948, potwierdzony wcześniej)
- 3L → 4L: -0.03 (marginalny regres)
- h=384 3L: stabilna architektura dla wszystkich 5 assetów jednocześnie

### 5. Krótkie horyzonty (6h, 12h) to artefakt metryki
- target=12h: 27.4 score, ~2000 trades/asset/rok
- target=6h: 4982 score, ~3000 trades, SOL train zwrot 141 BILIONÓW %
- Przyczyna: 0 kosztów transakcji w backteście + compounding 3000+ małych zysków = astronomia
- W produkcji: 3000 trades × 0.1% fee = 300% opłat rocznie → niezgrywalny
- **Granica sensowności: target ≥ 24h (≤ 1200 trades/asset/rok)**

### 6. Ensemble nie pomaga (przy optymalnej konfiguracji)
- 3 seedy: 3.447 vs single seed: 3.456 (brak poprawy)
- 3× dłuższy czas treningu
- Wcześniej ensemble pomagał przy niestabilnym XMR (#93 era), ale przy h=384 3L XMR jest stabilny

### 7. LR i epoki — optima
- lr=0.002: lepsze niż 0.001 przy 300 epokach (CosineAnnealing dobrze converge)
- lr=0.001: underfitting — potrzeba by ~500 epok żeby uzyskać ten sam efekt
- 300 epok: sweet spot
- 500 epok: overfit (CosineAnnealing dosięga bardzo niskiego LR, model wpada w lokalne minimum)

---

## Sweet spot (exp #109) — konfiguracja do kopiowania

```python
OPIS = "LSTM_h384_3L_drop03_target24h_discrete_funding_1H"
LOOKBACK = 168          # 7 dni na 1H
SINGLE_SEED = 42

# W train_lstm():
model = SignalLSTM(n_features=n_features, hidden=384, n_layers=3, dropout=0.3)
opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.02)
# CosineAnnealingLR, n_epochs=300, batch_size=min(512, len(Xt))

# W strategy():
fwd_ret = close.pct_change(24).shift(-24)   # target 24h
te = int(n * 0.80)                           # pojedyncze okno 80%
nn_pred = predict_lstm_confidence(model_info, features)  # tylko train_valid_idx

# Progi sygnałów:
signals[nn_pred > 0.15] = 0.5
signals[nn_pred > 0.35] = 0.75
signals[nn_pred > 0.55] = 1.0
signals[nn_pred < -0.15] = -0.5
signals[nn_pred < -0.55] = -1.0

# ATR stop:
atr_trailing_stop(close, atr_val, signals, multiplier=1.9, cooldown=24, profit_target_atr=3.0)
```

---

## Co NIE jest warte powtarzania

| Kierunek | Co testowano | Dlaczego nie warto |
|----------|-------------|-------------------|
| h=512 (jakakolwiek konfiguracja) | #102, #104, #113 | Zawsze destabilizuje co najmniej jeden asset |
| LOOKBACK=96 | #107 | BTC overfit train/val |
| LOOKBACK=240+ | #119 | Regresja vs 168 |
| dropout=0.5 | #106 | ETH idzie flat |
| dropout=0.4 przy target=24h | #108 vs #109 | 0.3 lepsze przy krótszym targecie |
| Ensemble 3 seedów | #112 | Brak poprawy, 3× koszt |
| Per-asset ensemble | #94 (stara sesja) | Niszczy cross-asset korelacje |
| target=12h, 6h | #110, #111 | Artefakt metryki, nierealny |
| target=48h | #93-#103 | 24h dramatycznie lepsze |
| LR=0.001 | #115 | Underfitting przy 300 epokach |
| 500 epok | #116 | Overfit |
| 4 warstwy | #114 | Marginalna regresja vs 3L |
| Progi 0.10/0.30/0.50 | #117 | Więcej false trades |
| wd=0.05 | #118 | Brak istotnej zmiany vs 0.02 |
| Walkforward (70%/80%) | #97 | Słabszy model 70% + oscylacje progów → 2-3× więcej trades |
| predict_on_data() (pełne OOS) | #98 | Noisy predykcje OOS → za dużo trades |
| Ciągłe pozycjonowanie | #95 | 3949 trades vs 785 przy dyskretnych progach |

---

## Co zostało do zbadania (nie testowano w tej sesji)

**Architektura:**
- BiLSTM (dwukierunkowy LSTM) — może lepiej uchwycić długoterminowe zależności
- GRU zamiast LSTM — mniej parametrów, często porównywalny
- Mniejszy model h=256 3L — może być bardziej regularny

**Features:**
- OBV (On-Balance Volume) — wolumen jako feature
- Stochastic RSI — bardziej czuły RSI
- Williams %R — momentum oscillator
- Spread 10Y-2Y (yield curve) — z context['TREASURY_10Y'] i context['TREASURY_2Y']
- NEWS sentiment — context['NEWS_BTC'], context['NEWS_ETH']

**ATR i zarządzanie pozycją:**
- multiplier=2.5 (luźniejszy stop, większa przestrzeń dla trades)
- profit_target_atr=4.0-5.0 (dłuższe trzymanie zwycięskich pozycji)
- cooldown=12 (krótszy cooldown — szybszy re-entry)

**Target:**
- target=36h — między 24h a 48h, może sweet spot
- target=24h z normalizacją log-return (zamiast pct_change) — stabilniejszy target

**Inne:**
- Różny seed niż 42 — sprawdzenie czy 42 jest globalnym czy lokalnym optimum dla tej konfiguracji
- wd=0.01 (mniej L2, odwrotny kierunek od 0.05)

---

## Wyniki per-asset (exp #109 — rekord)

| Asset | Train Sharpe | Val Sharpe | Val Return | Score |
|-------|-------------|-----------|-----------|-------|
| BTC   | 5.925 | 3.415 | +74% | 1.7905 |
| ETH   | 5.867 | 5.559 | +368% | 3.7864 |
| XMR   | 3.997 | 7.707 | +1402% | 3.6355 |
| SOL   | 6.195 | 4.147 | +254% | 2.4066 |
| TAO   | 5.347 | 6.976 | +1702% | 5.6629 |
| **AVG** | — | — | — | **3.4564** |

BTC jest najsłabszy (val period 2025-03→2026-03 był bessą dla BTC: B&H = -10.80%).
XMR i TAO mają val Sharpe > train — model dobrze generalizuje.
