"""Hypothesis-aligned OLS regressions, Newey-West HAC standard errors,
and incremental-R² evaluation.

Each function corresponds to one row of the hypothesis table (H1a/b ..
H4a/b) in the research proposal. We deliberately keep the regressions
*simple linear* and pair them with **HAC** standard errors rather than
reaching for fancier estimators: the contribution of this project is
the multi-dimensional **feature design**, not the regression machinery.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class RegressionResult:
    coef: pd.Series
    se: pd.Series
    t_stat: pd.Series
    p_value: pd.Series
    r_squared: float
    n_obs: int

    def to_markdown(self) -> str:
        df = pd.DataFrame({
            "coef": self.coef,
            "HAC se": self.se,
            "t": self.t_stat,
            "p": self.p_value,
        })
        return df.to_markdown(floatfmt=".4f")


def fit_ols_with_hac(
    y: pd.Series, X: pd.DataFrame, lags: int = 5
) -> RegressionResult:
    """OLS with Newey-West HAC SEs (default lag = 5, daily returns)."""
    import statsmodels.api as sm

    Xc = sm.add_constant(X, has_constant="add")
    mod = sm.OLS(y, Xc, missing="drop").fit(
        cov_type="HAC", cov_kwds={"maxlags": lags}
    )
    return RegressionResult(
        coef=mod.params,
        se=mod.bse,
        t_stat=mod.tvalues,
        p_value=mod.pvalues,
        r_squared=float(mod.rsquared),
        n_obs=int(mod.nobs),
    )


def incremental_r2(
    df: pd.DataFrame,
    target: str,
    base_features: Sequence[str],
    augment_features: Sequence[str],
) -> dict:
    """ΔR² from adding `augment_features` on top of `base_features`."""
    import statsmodels.api as sm
    base_X = sm.add_constant(df[list(base_features)], has_constant="add")
    full_X = sm.add_constant(df[list(base_features) + list(augment_features)], has_constant="add")
    r2_base = sm.OLS(df[target], base_X, missing="drop").fit().rsquared
    r2_full = sm.OLS(df[target], full_X, missing="drop").fit().rsquared
    return {
        "r2_base": float(r2_base),
        "r2_full": float(r2_full),
        "delta_r2": float(r2_full - r2_base),
        "augment": list(augment_features),
    }


def fit_decay_curve(coefs_by_lag: pd.Series) -> dict:
    """Fit b(d) = b0 * exp(-lambda * d) to the per-lag coefficient series.

    Returns (b0, lambda, R²-of-fit). Used to characterise the
    information-decay rate empirically (Hypothesis H1b).
    """
    raise NotImplementedError(
        "Decay-curve fitting is implemented in the private full repo."
    )


def asymmetry_test(coef_pos: float, coef_neg: float,
                   se_pos: float, se_neg: float) -> dict:
    """Wald-style test for H0: |b+| = |b-| (no asymmetric response).

    Under H0 the test statistic ( |b-| - |b+| ) / sqrt(se+² + se-²) is
    approximately N(0, 1). One-sided alternative |b-| > |b+| corresponds
    to loss-aversion / Kahneman-Tversky-style asymmetry.
    """
    raise NotImplementedError(
        "Asymmetry test is implemented in the private full repo."
    )
