# Methodology

This document is the technical companion to the top-level `README.md`. It
specifies the modelling decisions in enough detail that an external reader
can audit the design without access to the full (private) code.

## 1. Problem statement

Let `r_{t+1} = ln(P_{t+1} / P_t)` be the next-day log return of TSLA.
Let `A_t` denote the set of Tesla-relevant news articles published in the
30 calendar days prior to trading day `t`. We model the conditional mean

    E[r_{t+1} | F_t]  =  f( S(A_t),  X_t,  Z_t )

where

  * `S(A_t)` is a vector of **sentiment features** derived from `A_t` via
    the four-stage pipeline of Figure 1,
  * `X_t` is a vector of standard market controls
    (market return, VIX, lagged return, volume change),
  * `Z_t` is a vector of optional **regime indicators** (high-VIX dummy,
    earnings-window dummy), used only when testing H4.

The contribution is in the **structure of `S(A_t)`**, not in `f`.
We deliberately keep `f` linear (OLS with HAC standard errors) so that
the empirical claims attach to the feature design rather than to a
flexible non-linear estimator that could be fit to noise.

## 2. Pre-processing (DONE)

Implemented in `src/preprocessing.py`. Two consequential choices:

1. **Trailing-source stripping.**  About 6% of headlines in the corpus
   end with a redundant `" - <Publisher>"` suffix that injects publisher
   names into the lexicon-based sentiment score. We strip these via a
   conservative regex before scoring. This change shifts the corpus
   sentiment-mean by ≈0.012 — small but systematically positive (publisher
   names are mostly neutral, but neutral content dilutes high-magnitude
   negative scores into less-negative averages).

2. **Date normalisation.** The raw `date` field is parsed to a calendar
   date; the separate `publishedAt` ISO field (when present) is preserved
   for the trading-day alignment step. Articles with unparseable dates
   (≈0.13%) are dropped rather than imputed.

After cleaning the corpus contains **28,620 articles** spanning
**2021-01-01 → 2025-06-01** from seven sources (see `results/eda_summary.json`).

## 3. Sentiment scoring

### 3.1 Tier 1 — VADER (DONE)

VADER is a lexicon-based rule system with **zero learnable parameters**;
this is a feature, not a bug, for a baseline. The output `compound ∈ [-1, 1]`
is computed deterministically given the lexicon snapshot, which we pin
to the version shipped with `vaderSentiment==3.3.2`.

On the cleaned corpus VADER produces:

| metric | value |
|---|---|
| mean compound score | 0.0021 |
| std | 0.352 |
| share positive (>0.05) | 30.2% |
| share negative (<−0.05) | 29.0% |
| share neutral | 40.8% |

The near-zero mean is reassuring — a strongly asymmetric mean would
suggest either a lexicon artefact or a corpus collection bias. The
roughly balanced positive/negative shares confirm the corpus is not
crowded with bullish PR releases.

### 3.2 Tier 2 — FinBERT (WIP)

Implementation track: `src/sentiment.py::score_corpus_finbert`. The plan is

  1. Filter `A_t` to entity-relevant articles via a lightweight
     Tesla-mention regex tuned on a 500-headline annotated subset.
  2. Run ProsusAI/finbert in batched mode (batch=32 on CPU, batch=128 on
     A100) over the relevant subset.
  3. Convert the softmax `(p_pos, p_neu, p_neg)` to a signed score
     `finbert = p_pos − p_neg`.
  4. Cross-validate against VADER via `calibration_agreement(...)` —
     target ≥85% sign agreement; disagreements feed the audit queue.

The reason for this two-tier stack rather than going straight to
FinBERT is auditability. A lexicon score is interpretable token-by-token,
which matters when reviewers ask "why did 2024-04-17 spike?" If only
the FinBERT score is shipped, every spike requires a manual click-through.

## 4. Temporal aggregation (DONE — `src/aggregation.py`)

Daily sentiment is decomposed into three non-overlapping windows
(proposal §3.3):

  * **Recent** `S_recent[t]` over `(t−7, t−1)`  — captures fast information flow
  * **Medium** `S_medium[t]` over `(t−15, t−8)` — captures intermediate horizon
  * **Distant** `S_distant[t]` over `(t−30, t−16)` — captures slow / structural

Each window optionally has:

  * an **asymmetric split** `S+ = max(S, 0)`, `S− = max(−S, 0)`,
    motivated by prospect-theory loss aversion (Kahneman & Tversky 1979);
  * a **dispersion** `σ_S` (std of per-article scores within the window),
    motivated by divergence-of-opinion (Miller 1977).

This is *the* feature-design step the project contributes. The
combination — multi-window + asymmetric + dispersion — is, to our
knowledge, not jointly explored in prior Tesla-sentiment work; the
closest prior is Tetlock (2007) which uses a single window and a
symmetric mean.

### Look-ahead bias guard

The target is `r_{t+1}`, and **all** sentiment features for day `t` are
computed strictly over `(t−30, t−1)` — the day-`t` window is excluded
to prevent same-day after-hours news from leaking into the predictor.
This matters because Tesla earnings calls (the highest-information
events in the corpus) are released after the 16:00 ET close; including
day-`t` news would make the regression "predict" returns it has
already observed.

## 5. Regression layer (WIP — `src/model.py`)

The four hypotheses (proposal §5) map onto four nested specifications.
HAC standard errors (Newey-West, lag=5) are used throughout to handle
the residual autocorrelation typical of overlapping-window features.

| Spec | Feature set (in addition to controls) | Hypothesis tested |
|---|---|---|
| M1 | `S_recent`, `S_medium`, `S_distant` | H1a, H1b: time-decay magnitude |
| M2 | `S±_recent`, `S±_medium`, `S±_distant` | H2a, H2b: pos/neg asymmetry |
| M3 | `S_recent`, `σ_S_recent` | H3a, H3b: dispersion → return / vol |
| M4 | `S_recent`, `S_recent × VIX_dummy` | H4: regime moderation |

For each spec we report (i) coefficient and HAC-t, (ii) incremental R²
over the controls-only baseline (`r_t`, `r_mkt`, `VIX`, `vol_change`),
(iii) when applicable a Wald test on the relevant equality
(e.g. `|b_neg| = |b_pos|`).

## 6. Baseline result reproduced in this repo

A single-feature baseline (daily-mean VADER `S_daily` regressed on
contemporaneous and next-day TSLA log returns) yields:

| pair | Pearson r | p | Spearman ρ | p |
|---|---:|---:|---:|---:|
| `S_daily` vs `r_t` (same day) | **0.084** | **0.005** | **0.082** | **0.006** |
| `S_daily` vs `r_{t+1}` (next day) | 0.012 | 0.685 | 0.000 | 0.996 |

The same-day correlation is statistically significant; the next-day
correlation is not. This is the motivating gap the multi-window /
asymmetric framework is designed to close — the question is not whether
news sentiment *correlates* with returns (it does, contemporaneously)
but whether a more carefully *structured* feature set produces returns
predictability above the controls baseline. This is what M1–M4 test.

Reproduce with:

```bash
python scripts/run_vader_baseline.py
cat results/baseline_correlation.json
```

The numbers in this table are read directly from
`results/baseline_correlation.json`, which is regenerated end-to-end
on every run.

## 7. Limitations the project is explicit about

  * **Single-ticker scope.** The corpus and prices are Tesla-only. The
    framework is portable but cross-firm transfer is not part of this
    project's scope.
  * **English-only.** Chinese-language financial reports (sometimes
    high-information for the China-export thread of Tesla's story) are
    not in the corpus.
  * **Headline ≠ article.** VADER and FinBERT are run on headlines only
    in the current pipeline. The proposal lists body-text scoring as a
    Phase-2 extension.
  * **No causal claim.** Every result is a conditional correlation;
    instrumental-variable arguments are out of scope.

## References

  1. Tetlock, P. C. (2007). *Giving content to investor sentiment.*
     Journal of Finance, 62(3), 1139–1168.
  2. Kahneman, D., & Tversky, A. (1979). *Prospect theory: An analysis
     of decision under risk.* Econometrica, 47(2), 263–291.
  3. Miller, E. M. (1977). *Risk, uncertainty, and divergence of opinion.*
     Journal of Finance, 32(4), 1151–1168.
  4. Lo, A. W. (2004). *The adaptive markets hypothesis.* Journal of
     Portfolio Management, 30(5), 15–29.
  5. Araci, D. (2019). *FinBERT: Financial sentiment analysis with
     pre-trained language models.* arXiv:1908.10063.
