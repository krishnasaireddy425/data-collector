# V3 Exit Classifier — Report

Generated: 2026-04-20 09:24:45 UTC

## Split

- train: 611 entries, epochs (1776157500, 1776437700)
- val: 204 entries, epochs (1776438000, 1776542100)
- test: 204 entries, epochs (1776542400, 1776641100)

## Class balance

- train: losers=8231/69943 (11.8%)
- val: losers=1927/20381 (9.5%)
- test: losers=2681/24405 (11.0%)

## Baseline — V3 hold-to-settlement on TEST

- Entries: 201
- Losers: 19  Winners: 182
- Total hold PnL: **$+24.80**
- Per entry: $+0.123

## Exit-classifier results on TEST (walk-forward)

| Target precision | Threshold (val) | Test precision | Test recall | # exits | TP (losers caught) | FP (winners exited) | Avg exit t (TP) | Avg savings/TP | Total savings | FP cost | **Net PnL vs hold** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.99 | 0.9950 | 81.8% | 47.4% | 11 | 9 | 2 | 225.0 | $+1.90 | $+17.10 | $-9.40 | **$+7.70**  (+31% vs baseline) |
| 0.98 | 0.9950 | 81.8% | 47.4% | 11 | 9 | 2 | 225.0 | $+1.90 | $+17.10 | $-9.40 | **$+7.70**  (+31% vs baseline) |
| 0.95 | 0.9950 | 81.8% | 47.4% | 11 | 9 | 2 | 225.0 | $+1.90 | $+17.10 | $-9.40 | **$+7.70**  (+31% vs baseline) |
| 0.90 | 0.9950 | 81.8% | 47.4% | 11 | 9 | 2 | 225.0 | $+1.90 | $+17.10 | $-9.40 | **$+7.70**  (+31% vs baseline) |
