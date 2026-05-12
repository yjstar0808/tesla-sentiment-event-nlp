"""
Run reproducible exploratory data analysis on the Tesla news corpus.

Inputs:
  data/raw/tesla_news.csv  (not shipped in repo; see data/README.md)

Outputs:
  figures/eda_articles_per_month.png
  figures/eda_source_distribution.png
  figures/eda_title_length_hist.png
  results/eda_summary.json

This script is fully deterministic (no model calls, no randomness) so
the figures it produces are pinned to the corpus snapshot specified in
configs/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)


def _load(corpus_path: Path) -> pd.DataFrame:
    df = pd.read_csv(corpus_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])
    df["title"] = df["title"].fillna("").astype(str)
    return df


def plot_articles_per_month(df: pd.DataFrame) -> None:
    by_month = (
        pd.to_datetime(df["date"]).dt.to_period("M").value_counts().sort_index()
    )
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=150)
    ax.bar(by_month.index.astype(str), by_month.values, color="#2F5496", edgecolor="white")
    ax.set_title("Tesla news articles per month  (corpus: 2021-01 → 2025-06)")
    ax.set_ylabel("# articles")
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "eda_articles_per_month.png", bbox_inches="tight")
    plt.close(fig)


def plot_source_distribution(df: pd.DataFrame) -> None:
    counts = df["source"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.barh(counts.index[::-1], counts.values[::-1], color="#548235", edgecolor="white")
    ax.set_title("Articles by source")
    ax.set_xlabel("# articles")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "eda_source_distribution.png", bbox_inches="tight")
    plt.close(fig)


def plot_title_length(df: pd.DataFrame) -> None:
    lengths = df["title"].str.len()
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
    ax.hist(lengths, bins=60, color="#C55A11", edgecolor="white")
    ax.set_title("Headline length distribution")
    ax.set_xlabel("characters")
    ax.set_ylabel("# articles")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "eda_title_length_hist.png", bbox_inches="tight")
    plt.close(fig)


def summarise(df: pd.DataFrame) -> dict:
    return {
        "n_articles": int(len(df)),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "n_sources": int(df["source"].nunique()),
        "top_sources": df["source"].value_counts().head(7).to_dict(),
        "title_len_mean": float(df["title"].str.len().mean()),
        "title_len_median": float(df["title"].str.len().median()),
    }


def main(corpus_path: str) -> None:
    df = _load(Path(corpus_path))
    plot_articles_per_month(df)
    plot_source_distribution(df)
    plot_title_length(df)
    summary = summarise(df)
    (RES / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    default = r"C:\Users\Administrator\Desktop\nlp_baha\nlp_baha\tesla_news.csv"
    main(sys.argv[1] if len(sys.argv) > 1 else default)
