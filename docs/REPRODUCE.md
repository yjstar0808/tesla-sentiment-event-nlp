# Reproducing the numbers in this repo

The repo ships scripts that regenerate **every number and every figure**
referenced in the README from the raw news corpus and Yahoo Finance.
No pre-computed result is hard-coded into the documentation.

## 0. Get the corpus

The corpus (`tesla_news.csv`, ≈17 MB, 28k articles) is **not** committed
to git for license-cleanliness reasons. Place it at
`data/raw/tesla_news.csv` or pass an explicit path to the scripts below.

If you do not have the corpus and want to reproduce the figures on your
own data, the scripts only require the columns
`['date', 'title', 'description', 'source']`.

## 1. Environment

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# or: source .venv/bin/activate # macOS / Linux
pip install -r requirements.txt
```

Python 3.10+ is required.

## 2. Run the EDA

```bash
python scripts/run_eda.py
```

Produces:
- `figures/eda_articles_per_month.png`
- `figures/eda_source_distribution.png`
- `figures/eda_title_length_hist.png`
- `results/eda_summary.json`

## 3. Run the VADER baseline + price join

```bash
python scripts/run_vader_baseline.py
```

This step:
1. cleans the corpus (see §2 of `docs/methodology.md`),
2. scores every headline with VADER,
3. fetches TSLA daily prices via `yfinance` (falls back to a clearly-marked
   placeholder if offline — only the placeholder run prints
   `price_source: placeholder` in the JSON),
4. aggregates sentiment to the trading day, joins with returns,
5. reports the Pearson / Spearman correlations.

Outputs:
- `results/vader_per_article.parquet`
- `results/sentiment_daily.csv`
- `results/baseline_correlation.json`   ← the table in README §3
- `figures/sentiment_vs_return.png`
- `figures/sentiment_daily_timeseries.png`

The whole pipeline runs in ≈30 s on a laptop CPU.

## 4. Rebuild the architecture figure

```bash
python figures/make_architecture.py
```

## 5. Run the multi-window regressions  (WIP)

The four hypothesis-aligned regressions (M1–M4 in `docs/methodology.md`
§5) are implemented in `src/model.py`; the driver script that wires
them up to the daily panel is on track to land in the next iteration.
Until then the equivalent results can be reproduced manually:

```python
import pandas as pd
from src.aggregation import aggregate_per_day
from src.features import build_design_matrix, feature_columns_for_hypothesis
from src.model import fit_ols_with_hac

panel = aggregate_per_day(scored_articles=..., trading_days=...)
design = build_design_matrix(panel, tsla_prices=..., market_prices=..., vix=...)
result = fit_ols_with_hac(design["r_next"], design[feature_columns_for_hypothesis("H1")])
print(result.to_markdown())
```

The driver script `scripts/run_hypothesis_tests.py` is the next planned
addition — see the Roadmap in the README.
