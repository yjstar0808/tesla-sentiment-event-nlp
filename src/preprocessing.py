"""Text and date pre-processing for the Tesla news corpus.

The corpus is heterogeneous: some articles in the early years carry the
publisher name as a trailing suffix ("... - The Wall Street Journal"),
others have generic phrases like "[Removed]" instead of full content.
This module performs the deterministic cleaning that all downstream
sentiment/aggregation steps depend on, so it is run once and cached.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

_TRAILING_SOURCE_RE = re.compile(r"\s+-\s+[A-Z][A-Za-z .']+$")
_BRACKETED_PLACEHOLDER_RE = re.compile(r"^\s*\[[A-Za-z ]+\]\s*$")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class PreprocessConfig:
    """Frozen pre-processing configuration."""

    min_title_len: int = 10
    drop_placeholder_descriptions: bool = True
    drop_languages_other_than: tuple[str, ...] = ("en",)
    deduplicate_on: tuple[str, ...] = ("title", "date")


def strip_trailing_source(title: str) -> str:
    """'Tesla stock falls - Reuters' -> 'Tesla stock falls'."""
    if not isinstance(title, str):
        return ""
    return _TRAILING_SOURCE_RE.sub("", title).strip()


def is_placeholder(text: str) -> bool:
    return isinstance(text, str) and bool(_BRACKETED_PLACEHOLDER_RE.match(text))


def normalize_whitespace(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return _WHITESPACE_RE.sub(" ", s).strip()


def clean_corpus(df: pd.DataFrame, cfg: PreprocessConfig | None = None) -> pd.DataFrame:
    """Apply deterministic cleaning to the news dataframe.

    Returns a copy with normalised title/description columns and a stable
    'doc_id' column suitable for joining with downstream sentiment scores.
    """
    cfg = cfg or PreprocessConfig()
    out = df.copy()

    out["title"] = out["title"].fillna("").map(strip_trailing_source).map(normalize_whitespace)
    out["description"] = out["description"].fillna("").map(normalize_whitespace)

    if cfg.drop_placeholder_descriptions:
        out = out[~out["description"].map(is_placeholder)]

    out = out[out["title"].str.len() >= cfg.min_title_len]

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out = out.dropna(subset=["date"])

    out = out.drop_duplicates(subset=list(cfg.deduplicate_on)).reset_index(drop=True)
    out["doc_id"] = out.index.astype("int64")
    return out


def trading_day_aligned(news_dates: Iterable, trading_days: Iterable):
    """Map each news date to the *next* trading day.

    Critical for avoiding look-ahead bias: a news article published after
    16:00 ET should be associated with the next day's open-to-close return,
    not the current day's. The naive same-day join is the single most
    common pitfall in news-to-return studies.
    """
    raise NotImplementedError(
        "Trading-day alignment is implemented in the private full repo. "
        "The public preview ships only the deterministic text-side preprocessing."
    )
