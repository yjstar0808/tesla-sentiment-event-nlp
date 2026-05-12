"""Join the per-day sentiment panel with market-side control variables.

The dependent variable is the log return of TSLA: r_{t+1} = ln(P_{t+1} / P_t).
Controls (from the research proposal §3.3.3): market return r_mkt (SPY),
implied volatility (VIX), lagged return r_t, and volume change Vol.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    target: str = "r_next"      # next-day log return (avoids look-ahead)
    market_col: str = "r_mkt"
    vix_col: str = "vix"
    lag_return_col: str = "r_t"
    volume_change_col: str = "vol_change"


def log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices).diff()


def build_design_matrix(
    sentiment_panel: pd.DataFrame,
    tsla_prices: pd.DataFrame,
    market_prices: pd.DataFrame,
    vix: pd.Series,
    cfg: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Stitch sentiment + price + market data into a single regression frame.

    NOTE on look-ahead bias. The target is r_{t+1}; all features must be
    measurable strictly before the start of trading on day t+1. The
    sentiment panel is windowed on (t-30, t-1) so this holds by construction;
    the controls (r_mkt, vix, vol_change) use day-t values that are also
    measurable by end-of-day t. See docs/methodology.md for the discussion
    of borderline cases (after-hours news, weekends, holidays).
    """
    cfg = cfg or FeatureConfig()

    tsla_prices = tsla_prices.copy()
    tsla_prices["r_t"] = log_returns(tsla_prices["close"])
    tsla_prices["r_next"] = tsla_prices["r_t"].shift(-1)
    tsla_prices["vol_change"] = tsla_prices["volume"].pct_change()

    market_prices = market_prices.copy()
    market_prices["r_mkt"] = log_returns(market_prices["close"])

    df = sentiment_panel.join(tsla_prices[["r_t", "r_next", "vol_change"]], how="inner")
    df = df.join(market_prices[["r_mkt"]], how="left")
    df["vix"] = vix.reindex(df.index)

    df = df.dropna(subset=[cfg.target, cfg.market_col, cfg.vix_col, cfg.lag_return_col])
    return df


def feature_columns_for_hypothesis(hypothesis: str) -> list[str]:
    """Map H1 / H2 / H3 / H4 to the relevant feature subset (see §5)."""
    base_controls = ["r_mkt", "vix", "r_t", "vol_change"]
    table = {
        "H1": ["S_recent", "S_medium", "S_distant"] + base_controls,
        "H2": ["Spos_recent", "Sneg_recent",
               "Spos_medium", "Sneg_medium",
               "Spos_distant", "Sneg_distant"] + base_controls,
        "H3": ["S_recent", "sigmaS_recent"] + base_controls,
        "H4": ["S_recent"] + base_controls,   # H4 is an interaction model — see model.py
    }
    if hypothesis not in table:
        raise KeyError(f"unknown hypothesis: {hypothesis}; expected one of {list(table)}")
    return table[hypothesis]
