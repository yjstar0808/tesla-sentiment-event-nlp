"""
Generates figures/architecture_overview.png — the headline figure
referenced from README.md.

Four-stage pipeline:
  News corpus -> Sentiment scoring -> Multi-window aggregation -> Hypothesis-aligned regressions

Each block is colour-coded by completion state:
  blue  = done       (baseline implemented + numbers reproduced in this repo)
  amber = in progress (architecture finalized, implementation underway)
  grey  = planned    (interface defined, work scheduled)

This colour coding matches the Status table in README.md so a reader
can audit progress at a glance.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent

COL_DONE = "#DCEAF7"
EDG_DONE = "#2F5496"
COL_WIP = "#FDE9CF"
EDG_WIP = "#C55A11"
COL_PLAN = "#EDEDED"
EDG_PLAN = "#6F6F6F"


def box(ax, x, y, w, h, label, sub, face, edge, title_fs=11, sub_fs=8.5):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.05",
        linewidth=1.5, edgecolor=edge, facecolor=face,
    ))
    ax.text(x + w / 2, y + h * 0.68, label, ha="center", va="center",
            fontsize=title_fs, fontweight="bold", color="#1F3864")
    ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center",
            fontsize=sub_fs, color="#404040", style="italic")


def arrow(ax, x1, y1, x2, y2, color="#404040", lw=1.2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
        color=color, lw=lw, connectionstyle="arc3,rad=0.0",
        shrinkA=2, shrinkB=2,
    ))


def main():
    fig, ax = plt.subplots(figsize=(12.5, 7.6), dpi=160)
    ax.set_xlim(0, 13); ax.set_ylim(0, 8.5)
    ax.set_aspect("equal"); ax.axis("off")

    ax.text(6.5, 8.15, "Tesla Sentiment → Return: four-stage pipeline",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#1F3864")

    # Stage 1
    box(ax, 0.3, 6.4, 12.4, 1.1,
        "Stage 1   ▸   News corpus  (n = 28,620 cleaned articles, 2021-01 → 2025-06)",
        "Business Insider · Bloomberg · TechCrunch · ABC · BBC · Wired · WSJ",
        COL_DONE, EDG_DONE, title_fs=11.5)

    # Stage 2
    box(ax, 0.3, 4.8, 6.0, 1.2,
        "Stage 2a   ▸   VADER baseline",
        "lexicon · headline-level · compound ∈ [-1, 1]   (DONE)",
        COL_DONE, EDG_DONE)
    box(ax, 6.7, 4.8, 6.0, 1.2,
        "Stage 2b   ▸   FinBERT scoring",
        "ProsusAI/finbert · transformer · pos/neu/neg   (WIP)",
        COL_WIP, EDG_WIP)

    # Stage 3
    box(ax, 0.3, 3.0, 12.4, 1.4,
        "Stage 3   ▸   Multi-window temporal aggregation",
        "S_recent (1-7d)   ·   S_medium (8-15d)   ·   S_distant (16-30d)   ·   "
        "asymmetric split S+ / S-   ·   dispersion σ_S",
        COL_DONE, EDG_DONE, title_fs=11.5)

    # Stage 4 — split into four hypothesis cells
    cell_y = 1.1; cell_h = 1.5; cell_w = 2.9; gap = 0.15
    xs = [0.3 + i * (cell_w + gap) for i in range(4)]
    hyps = [
        ("H1: time-decay",        "|b_recent| > |b_distant|", COL_WIP,  EDG_WIP),
        ("H2: asymmetry",         "|b_neg| > |b_pos|",        COL_WIP,  EDG_WIP),
        ("H3: dispersion",        "σ_S → σ_r positive",       COL_PLAN, EDG_PLAN),
        ("H4: regime moderation", "VIX × S interaction",      COL_PLAN, EDG_PLAN),
    ]
    for x, (lab, sub, face, edge) in zip(xs, hyps):
        box(ax, x, cell_y, cell_w, cell_h, lab, sub, face, edge, title_fs=10.5, sub_fs=8)

    # Arrows
    arrow(ax, 6.5, 6.38, 3.3, 6.02)
    arrow(ax, 6.5, 6.38, 9.7, 6.02)
    arrow(ax, 3.3, 4.78, 6.5, 4.42)
    arrow(ax, 9.7, 4.78, 6.5, 4.42)
    for x in xs:
        arrow(ax, x + cell_w / 2, 2.98, x + cell_w / 2, 2.62)

    # Legend
    handles = [
        mpatches.Patch(facecolor=COL_DONE, edgecolor=EDG_DONE, label="Implemented & reproduced in this repo"),
        mpatches.Patch(facecolor=COL_WIP, edgecolor=EDG_WIP, label="In progress (architecture finalized)"),
        mpatches.Patch(facecolor=COL_PLAN, edgecolor=EDG_PLAN, label="Planned (interface defined)"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04),
              ncol=3, frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "architecture_overview.png", dpi=200, bbox_inches="tight")
    print(f"saved: {OUT / 'architecture_overview.png'}")


if __name__ == "__main__":
    main()
