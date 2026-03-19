# Opis strategii autoquant — stan na 2026-03-19

## Ogólna idea — w jednym zdaniu

Sieć neuronowa (LSTM) patrzy na ostatnie **7 dni** wykresu godzinowego każdej kryptowaluty, analizuje dziesiątki wskaźników jednocześnie, i mówi: *"za 24 godziny cena będzie wyżej czy niżej?"*. Na podstawie tej odpowiedzi otwieramy lub zamykamy pozycję.

---

## Krok 1 — Co sieć "widzi" (23 wskaźniki)

Dla każdej godziny strategia oblicza 23 liczby opisujące stan rynku:

**Trend i momentum:**
- Odległość ceny od EMA50 i EMA200 — czy cena jest powyżej/poniżej długoterminowego trendu
- Wskaźniki Ichimoku (chmura, tenkan/kijun) — japońska analiza techniczna
- MACD × 2 (standardowy 12/26 + szybszy 8/17) — siła i kierunek trendu
- RSI — czy rynek jest wykupiony/wyprzedany
- Chikou span — porównanie dzisiejszej ceny z ceną sprzed 26 godzin

**Zmienność i wolumen:**
- ATR — jak bardzo cena "skacze" (miara ryzyka)
- Szerokość wstęg Bollingera — jak szeroki jest zakres cen
- Wolumen vs. średnia z 20 dni — czy handel jest aktywniejszy niż normalnie

**Momentum wielookresowe:**
- Zwroty za ostatnie 1h, 3h, 6h, 12h, 24h — skąd przyszliśmy

**Makro (rynek globalny):**
- SPY trend — czy giełda USA rośnie czy spada
- UUP trend — siła dolara (silny dolar = słabe krypto)
- Market funding rate — średnia z kontraktów BTC/ETH/SOL futures. Gdy wszyscy płacą dużo za lewarowanie pozycji long → rynek przegrzany → potencjalny spadek
- VIXY trend — indeks strachu (VIX). Gdy strach rośnie → rynki sprzedają ryzykowne aktywa

---

## Krok 2 — Jak uczy się sieć

Sieć LSTM (Long Short-Term Memory) to rodzaj sieci neuronowej zaprojektowanej specjalnie do **sekwencji w czasie** — czyta dane godzina po godzinie tak jak człowiek czyta tekst słowo po słowie, pamiętając kontekst z poprzednich kroków.

Trening wygląda tak:
1. Bierzemy dane z okresu **marzec 2023 → marzec 2025** (2 lata, ~17 500 godzin na każdy asset)
2. Pierwszych **80%** to dane treningowe, pozostałe **20%** to wewnętrzna walidacja
3. Sieć dostaje: *"oto 168 godzin (7 dni) wskaźników → ile wynosiło pct_change za kolejne 24h?"*
4. Przez 300 cykli (epok) sieć poprawia swoje "domysły" metodą gradientów
5. Wynik: sieć zwraca liczbę od -1 do +1 — *"jak mocno wierzę, że cena wzrośnie"*

**Kluczowy detal:** sieć generuje sygnały **tylko na ostatnich 20% danych treningowych** (czyli tylko tam gdzie nie trenowała). Na "świeżych" danych walidacyjnych (marzec–grudzień 2025) działa jak black box.

---

## Krok 3 — Jak sygnał zamienia się w pozycję

Sieć zwraca liczbę (np. 0.42 = "dość pewny wzrost"). Mapujemy to na pozycję:

```
Sieć mówi > 0.55  →  LONG z pełną siłą (100% kapitału)
Sieć mówi > 0.35  →  LONG z 75% kapitału
Sieć mówi > 0.15  →  LONG z 50% kapitału

Sieć mówi < -0.55 →  SHORT z pełną siłą
Sieć mówi < -0.15 →  SHORT z 50% kapitału

Pomiędzy -0.15 a +0.15 → siedzimy w gotówce (flat)
```

---

## Krok 4 — Stop loss i ochrona zysku (ATR trailing stop)

Na każdej otwartej pozycji działa automatyczny stop:
- **Stop podąża za ceną** — gdy cena rośnie, stop też podnosi się (ale nigdy nie spada)
- **Szerokość stopu:** 1.9 × ATR (= 1.9 × typowy zakres ceny na godzinę)
- **Profit target:** gdy zysk = 3× ATR → zmniejszamy pozycję o połowę (realizacja zysku)
- **Cooldown 24h po stopie** — po wybiciu stopu czekamy 24 godziny zanim ponownie wejdziemy

Dzięki temu strategia **szybko ucina straty, ale pozwala rosnąć zyskom**.

---

## Wyniki per kryptowaluta (eksperyment #109, walidacja marzec 2025 – marzec 2026)

| Krypto | Sharpe | Zwrot (val.) | Buy & Hold | Ocena |
|--------|--------|-------------|-----------|-------|
| **XMR** (Monero) | 7.7 | +1402% | +76% | Rewelacyjny |
| **TAO** (Bittensor) | 7.0 | +1702% | +12% | Rewelacyjny |
| **ETH** (Ethereum) | 5.6 | +368% | +22% | Bardzo dobry |
| **SOL** (Solana) | 4.1 | +254% | -27% | Dobry |
| **BTC** (Bitcoin) | 3.4 | +74% | -11% | Przyzwoity |

**Sharpe ratio** to miara jakości — im wyżej, tym lepiej (3+ to już bardzo dobre, 5+ to wybitne).

**Ważna uwaga:** to backtest bez kosztów transakcji. Realne wyniki będą niższe — każda transakcja kosztuje ok. 0.1% na Binance. Strategia robi ~1100 transakcji rocznie na asset, więc ok. 110% w opłatach. Przy XMR +1402% to wciąż ogromny zysk, ale trzeba to uwzględnić przy ocenie realnej dochodowości.

---

## Dlaczego XMR i TAO są najlepsze, a BTC najgorszy?

**XMR i TAO** — mniej płynne, bardziej "techniczne" rynki. Wzorce cenowe są bardziej powtarzalne, LSTM je łapie dobrze. XMR w szczególności ma ciekawą dynamikę — nawet gdy BTC spada, XMR zachowuje się inaczej (privacy coin, niszowy popyt).

**BTC** — to najbardziej "efektywny" rynek, największa płynność na świecie. Najtrudniej przewidzieć. Dodatkowo w naszym okresie walidacyjnym (2025-2026) BTC był w bessie (-11% buy & hold), więc i tak wycisnęliśmy +74% tam gdzie "trzymać" dawało stratę.

---

## Gdzie jesteśmy w projekcie

```
Score ogólny:     3.46  (był 0.41 przy prostej strategii — wzrost 8×)
Najsłabszy asset: BTC  1.79
Najmocniejszy:    TAO  5.66
```

Dotarliśmy do **lokalnego plateau** przy obecnej architekturze. Kolejne potencjalne kroki:
- Nowe cechy: wolumen (OBV), yield curve, sentiment newsów
- Inna architektura sieci: BiLSTM, GRU
- Inne parametry stop-lossa (multiplier, profit target)

Szczegółowa historia eksperymentów i lista tego co nie działa: `testy/eksperymenty_19_03.md`

---

## Parametry techniczne (dla dewelopera)

| Parametr | Wartość |
|----------|---------|
| Model | LSTM 3-warstwowy, hidden=384 |
| Dropout | 0.3 |
| Lookback | 168 świec (7 dni × 24h) |
| Target | pct_change(24h) |
| Trening | 300 epok, lr=0.002, AdamW wd=0.02 |
| Timeframe | 1H |
| Train split | 80% (2023-03 → ~2025-01) |
| Val split | ostatnie 20% train + pełne val (2025-03 → 2026-03) |
| Seed | 42 |
| ATR multiplier | 1.9 |
| ATR profit target | 3.0× |
| ATR cooldown | 24h |
| Transakcje/rok | ~1100 per asset |
