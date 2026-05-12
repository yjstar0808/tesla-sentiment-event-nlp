"""
Run VADER on every headline in the Tesla news corpus, fetch TSLA prices,
and produce the *baseline* table that the proposed multi-dimensional
framework will be compared against. All numbers in the output JSON / CSV
are produced by this script — nothing here is hard-coded.

Outputs:
  results/vader_per_article.parquet
  results/sentiment_daily.csv
  results/baseline_correlation.json
  figures/sentiment_vs_return.png
  figures/sentiment_daily_timeseries.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing import clean_corpus, PreprocessConfig
from src.sentiment import score_corpus_vader

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)


def _load_tsla_prices() -> pd.DataFrame:
    """Try yfinance; fall back to a deterministic placeholder if offline.

    The placeholder uses a synthetic random-walk so the script always
    completes; the README is explicit that the baseline correlation
    reported in the docs uses real yfinance data, and prints
    a clear marker when the placeholder is in use.
    """
    try:
        import yfinance as yf
        df = yf.download("TSLA", start="2021-01-01", end="2025-06-30",
                         progress=False, auto_adjust=False)
        if df is None or len(df) == 0:
            raise RuntimeError("yfinance returned empty")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.rename(columns={"Close": "close", "Volume": "volume"})
        df.index = pd.to_datetime(df.index).date
        df.index.name = "date"
        df["source"] = "yfinance"
        return df[["close", "volume", "source"]]
    except Exception as e:
        print(f"[warn] yfinance unavailable ({e}); using placeholder", file=sys.stderr)
        idx = pd.bdate_range("2021-01-01", "2025-06-01").date
        rng = np.random.default_rng(20260512)
        ret = rng.normal(0, 0.03, len(idx))
        close = 200.0 * np.exp(np.cumsum(ret))
        vol = rng.lognormal(17, 0.4, len(idx))
        df = pd.DataFrame({"close": close, "volume": vol, "source": "placeholder"},
                          index=pd.Index(idx, name="date"))
        return df


def main(corpus_path: str) -> None:
    print("[1/5] loading + cleaning corpus...")
    raw = pd.read_csv(corpus_path)
    df = clean_corpus(raw, PreprocessConfig())
    print(f"      kept {len(df):,} of {len(raw):,} articles after cleaning")

    print("[2/5] scoring with VADER (compound score on headlines)...")
    df = score_corpus_vader(df, text_col="title")
    df[["doc_id", "date", "source", "title", "vader"]].to_parquet(
        RES / "vader_per_article.parquet", index=False
    )

    print("[3/5] aggregating to daily sentiment...")
    daily = (
        df.groupby("date")["vader"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "S_daily", "std": "sigmaS_daily", "count": "n_articles"})
    )
    daily.index = pd.to_datetime(daily.index).date
    daily.to_csv(RES / "sentiment_daily.csv")

    print("[4/5] loading TSLA prices + computing returns...")
    prices = _load_tsla_prices()
    prices["r_t"] = np.log(prices["close"]).diff()
    prices["r_next"] = prices["r_t"].shift(-1)

    joined = daily.join(prices[["r_t", "r_next"]], how="inner").dropna()
    print(f"      {len(joined):,} aligned trading-day observations")

    print("[5/5] computing baseline correlations...")
    pearson_same = stats.pearsonr(joined["S_daily"], joined["r_t"])
    pearson_next = stats.pearsonr(joined["S_daily"], joined["r_next"])
    spearman_same = stats.spearmanr(joined["S_daily"], joined["r_t"])
    spearman_next = stats.spearmanr(joined["S_daily"], joined["r_next"])

    out = {
        "n_articles_after_cleaning": int(len(df)),
        "n_trading_days": int(len(joined)),
        "price_source": str(prices["source"].iloc[0]),
        "vader_mean": float(df["vader"].mean()),
        "vader_std": float(df["vader"].std()),
        "vader_share_positive": float((df["vader"] > 0.05).mean()),
        "vader_share_negative": float((df["vader"] < -0.05).mean()),
        "vader_share_neutral": float(df["vader"].between(-0.05, 0.05, inclusive="both").mean()),
        "pearson_S_vs_rt": {"r": float(pearson_same.statistic), "p": float(pearson_same.pvalue)},
        "pearson_S_vs_r_next": {"r": float(pearson_next.statistic), "p": float(pearson_next.pvalue)},
        "spearman_S_vs_rt": {"rho": float(spearman_same.statistic), "p": float(spearman_same.pvalue)},
        "spearman_S_vs_r_next": {"rho": float(spearman_next.statistic), "p": float(spearman_next.pvalue)},
    }
    (RES / "baseline_correlation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    # --- figures ---
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    ax.scatter(joined["S_daily"], joined["r_next"], s=6, alpha=0.35, color="#2F5496")
    ax.axhline(0, color="#999", lw=0.6); ax.axvline(0, color="#999", lw=0.6)
    ax.set_xlabel("Daily mean VADER sentiment $S_t$")
    ax.set_ylabel("Next-day log return $r_{t+1}$")
    ax.set_title(f"Same-day sentiment vs next-day return  (n={len(joined):,})")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "sentiment_vs_return.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.4), dpi=150)
    rolling = daily["S_daily"].rolling(20, min_periods=5).mean()
    ax.plot(pd.to_datetime(daily.index), rolling, lw=1.0, color="#2F5496",
            label="VADER mean, 20-day rolling")
    ax.axhline(0, color="#999", lw=0.6)
    ax.set_title("Daily Tesla news sentiment (rolling 20-day mean)")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "sentiment_daily_timeseries.png", bbox_inches="tight")
    plt.close(fig)

    print("\nDone. See results/ and figures/.")


if __name__ == "__main__":
    default = r"C:\Users\Administrator\Desktop\nlp_baha\nlp_baha\tesla_news.csv"
    main(sys.argv[1] if len(sys.argv) > 1 else default)
