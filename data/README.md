# Data

The repository code expects the following inputs at run time. Raw data
is intentionally **not committed** — the corpus is collected from
third-party APIs whose terms of service do not permit redistribution.

## Files

| Path | Source | Description |
|---|---|---|
| `data/raw/tesla_news.csv` | News API + scrapers | Headlines, descriptions, publish dates for Tesla-mentioning articles, 2021-01-01 → 2025-06-01 |
| (fetched at runtime) | Yahoo Finance via `yfinance` | TSLA daily OHLCV |
| (planned) | CBOE | VIX index, daily close |
| (planned) | SEC EDGAR / Tesla IR | Quarterly fundamentals, earnings dates |

## Corpus snapshot

The current corpus has **28,657 raw articles**, dropping to **28,620**
after deterministic cleaning (deduplicate, drop placeholder
descriptions, normalise whitespace, require title length ≥10).

Source distribution (from `results/eda_summary.json`):

| Source | # articles |
|---|---:|
| Business Insider | 13,235 |
| Bloomberg | 4,546 |
| TechCrunch | 4,431 |
| ABC News | 3,571 |
| BBC News | 1,180 |
| Wired | 1,029 |
| The Wall Street Journal | 664 |

## Why the data is gitignored

  * Most sources' ToS permit research use but not redistribution.
  * Working with a 17 MB CSV on a git server is wasteful when
    `scripts/run_eda.py` and `scripts/run_vader_baseline.py` are fully
    deterministic w.r.t. the input file.
  * If you want a synthetic stand-in to test the pipeline end-to-end,
    `scripts/run_vader_baseline.py` falls back to a deterministic
    random-walk price series when `yfinance` is unavailable; a similar
    synthetic news generator can be added on request.
