"""Sentiment scoring layer.

The framework uses a two-tier sentiment stack:

  Tier 1 (lexicon-based)  : VADER, run on every headline as a fast
                            and fully-reproducible baseline. Used both
                            as the primary feature in baseline models
                            and as a calibration anchor for Tier 2.

  Tier 2 (transformer)    : FinBERT (ProsusAI/finbert), run on the
                            subset of headlines that pass an entity-
                            relevance filter. Produces per-headline
                            (positive, neutral, negative) probabilities.

The public preview ships Tier 1 only. Tier 2 weights and the entity-
relevance filter live in the private full repo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@dataclass(frozen=True)
class SentimentConfig:
    use_finbert: bool = False
    finbert_model: str = "ProsusAI/finbert"
    finbert_batch_size: int = 32
    headline_only: bool = True  # title is more diagnostic than body for news
    device: str = "cpu"


class VaderScorer:
    """Thin wrapper that yields a single compound score in [-1, 1]."""

    def __init__(self) -> None:
        self._analyser = SentimentIntensityAnalyzer()

    def score(self, text: str) -> float:
        if not isinstance(text, str) or not text:
            return 0.0
        return float(self._analyser.polarity_scores(text)["compound"])

    def score_batch(self, texts: Sequence[str]) -> np.ndarray:
        return np.array([self.score(t) for t in texts], dtype=np.float32)


def score_corpus_vader(df: pd.DataFrame, text_col: str = "title") -> pd.DataFrame:
    """Append a 'vader' column with compound scores in [-1, 1]."""
    scorer = VaderScorer()
    out = df.copy()
    out["vader"] = scorer.score_batch(out[text_col].tolist())
    return out


def score_corpus_finbert(df: pd.DataFrame, text_col: str = "title",
                         cfg: SentimentConfig | None = None) -> pd.DataFrame:
    """Append a 'finbert' column (signed score in [-1, 1]).

    Stubbed in the public preview. The private repo runs ProsusAI/finbert
    via HuggingFace Transformers with batched inference and converts the
    softmax output (p_pos, p_neu, p_neg) into a signed score
    ``finbert = p_pos - p_neg``.
    """
    raise NotImplementedError(
        "FinBERT scoring is implemented in the private full repo."
    )


def calibration_agreement(vader: np.ndarray, finbert: np.ndarray) -> dict:
    """Compute agreement between two sentiment streams on the same docs.

    Used as a sanity check: VADER and FinBERT should agree on sign for
    the large majority of headlines that are unambiguous; disagreements
    flag candidates for manual audit.
    """
    sign_agree = float(np.mean(np.sign(vader) == np.sign(finbert)))
    pearson = float(np.corrcoef(vader, finbert)[0, 1])
    return {
        "n": int(len(vader)),
        "sign_agreement_rate": sign_agree,
        "pearson_r": pearson,
    }
