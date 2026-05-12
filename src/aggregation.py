"""Temporal aggregation of per-article sentiment into per-day features.

Implements the multi-window decomposition described in the research
proposal (Section 3.3):

    S_recent[t]   = mean sentiment over (t-7, t-1)
    S_medium[t]   = mean sentiment over (t-15, t-8)
    S_distant[t]  = mean sentiment over (t-30, t-16)

and their signed (asymmetric) split

    S+[t] = max(S[t], 0)
    S-[t] = max(-S[t], 0)

plus the cross-sectional sentiment dispersion sigma_S[t] (standard
deviation of per-article sentiment within the window).

This module is the bridge between the article-level scoring layer
and the regression layer (src/features.py / src/model.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowSpec:
    name: str
    lo: int  # inclusive number of days back, lower bound
    hi: int  # inclusive number of days back, upper bound
    # so 'recent' is lo=1, hi=7  (yesterday back to a week ago)


DEFAULT_WINDOWS: tuple[WindowSpec, ...] = (
    WindowSpec("recent", 1, 7),
    WindowSpec("medium", 8, 15),
    WindowSpec("distant", 16, 30),
)


@dataclass(frozen=True)
class AggregationConfig:
    sentiment_col: str = "vader"
    date_col: str = "date"
    windows: tuple[WindowSpec, ...] = DEFAULT_WINDOWS
    decay_lambda: float | None = None
    # if not None, apply exponential time-decay weights exp(-lambda * d)
    # within each window before averaging. lambda=0 -> uniform mean.
    asymmetric: bool = True
    include_dispersion: bool = True


def _window_indices(target_day: pd.Timestamp, spec: WindowSpec) -> tuple[pd.Timestamp, pd.Timestamp]:
    lower = target_day - pd.Timedelta(days=spec.hi)
    upper = target_day - pd.Timedelta(days=spec.lo)
    return lower, upper


def aggregate_per_day(
    scored_articles: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
    cfg: AggregationConfig | None = None,
) -> pd.DataFrame:
    """Produce a (n_trading_days, n_features) panel.

    For each trading day t and each window w, compute:
      * S_w[t]            : mean sentiment of articles published in [t-hi, t-lo]
      * S+_w[t], S-_w[t]  : asymmetric split (if cfg.asymmetric)
      * sigma_S_w[t]      : within-window dispersion (if cfg.include_dispersion)

    Trading days that fall outside the news corpus span (no articles in
    any window) are dropped to avoid silently-zero rows that would bias
    downstream regression coefficients.
    """
    cfg = cfg or AggregationConfig()
    sa = scored_articles.copy()
    sa[cfg.date_col] = pd.to_datetime(sa[cfg.date_col])

    rows = []
    for t in trading_days:
        row = {"date": t}
        any_obs = False
        for w in cfg.windows:
            lo, hi = _window_indices(t, w)
            mask = (sa[cfg.date_col] >= lo) & (sa[cfg.date_col] <= hi)
            window_scores = sa.loc[mask, cfg.sentiment_col].to_numpy(dtype=np.float64)

            if window_scores.size == 0:
                row[f"S_{w.name}"] = np.nan
                if cfg.asymmetric:
                    row[f"Spos_{w.name}"] = np.nan
                    row[f"Sneg_{w.name}"] = np.nan
                if cfg.include_dispersion:
                    row[f"sigmaS_{w.name}"] = np.nan
                continue

            any_obs = True
            mean = float(window_scores.mean())
            row[f"S_{w.name}"] = mean
            if cfg.asymmetric:
                row[f"Spos_{w.name}"] = max(mean, 0.0)
                row[f"Sneg_{w.name}"] = max(-mean, 0.0)
            if cfg.include_dispersion:
                row[f"sigmaS_{w.name}"] = float(window_scores.std(ddof=1)) if window_scores.size > 1 else 0.0

        if any_obs:
            rows.append(row)

    return pd.DataFrame(rows).set_index("date").sort_index()


def exponential_decay_weights(n: int, lam: float) -> np.ndarray:
    """w_d = exp(-lam * d) / sum, d = 0..n-1."""
    d = np.arange(n, dtype=np.float64)
    w = np.exp(-lam * d)
    return w / w.sum()
