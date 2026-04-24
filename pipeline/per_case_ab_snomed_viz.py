#!/usr/bin/env python3
"""Figures from ``per_case_ab_snomed_summary.csv`` (+ optional cross-concept CSV).

Reads outputs of ``per_case_snomed_ab_comparison`` and writes:
  - a six-panel dashboard (distributions + scatter + mean composition), and
  - a two-panel figure for capped is-a cross pairs (depth gap + who-is-broader).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from .paths import (
    SNOMED_PER_CASE_AB_CROSS,
    SNOMED_PER_CASE_AB_CROSS_FIG,
    SNOMED_PER_CASE_AB_DASHBOARD,
    SNOMED_PER_CASE_AB_SUMMARY,
    ensure_results,
)


def _f(x: str) -> float | None:
    s = (x or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _i(x: str) -> int | None:
    s = (x or "").strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def load_summary(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_cross(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_dashboard(rows: list[dict[str, str]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    inter = [_i(r["n_concepts_intersection"]) for r in rows]
    inter = [x for x in inter if x is not None]
    jac = [_f(r["jaccard_concepts_a_b"]) for r in rows]
    jac = [x for x in jac if x is not None]
    ddiff = [_f(r["mean_depth_weighted_b_minus_a"]) for r in rows]
    ddiff = [x for x in ddiff if x is not None]
    surf = [_i(r["n_string_surface_intersection"]) for r in rows]
    surf = [x for x in surf if x is not None]
    only_a = [_i(r["n_concepts_a_only"]) for r in rows]
    only_a = [x for x in only_a if x is not None]
    only_b = [_i(r["n_concepts_b_only"]) for r in rows]
    only_b = [x for x in only_b if x is not None]

    jac_all = [_f(r["jaccard_concepts_a_b"]) for r in rows]
    ddiff_all = [_f(r["mean_depth_weighted_b_minus_a"]) for r in rows]
    inter_all = [_i(r["n_concepts_intersection"]) for r in rows]
    surf_all = [_i(r["n_string_surface_intersection"]) for r in rows]

    xs_j, ys_d = [], []
    xs_surf, ys_inter = [], []
    for j, d, s, it in zip(jac_all, ddiff_all, surf_all, inter_all):
        if j is not None and d is not None:
            xs_j.append(j)
            ys_d.append(d)
        if s is not None and it is not None:
            xs_surf.append(s)
            ys_inter.append(it)

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2), constrained_layout=True, facecolor="white")
    fig.suptitle(
        "Per-case SNOMED comparison (Corpus A vs B): NER → concepts",
        fontsize=13,
        fontweight="bold",
        color="#1a1a1a",
    )

    ax = axes[0, 0]
    if inter:
        hi = max(inter) + 1
        bins = min(hi, 20)
        ax.hist(inter, bins=bins, color="#37474f", edgecolor="white", alpha=0.88)
    ax.set_title("Shared concepts per case\n(|A∩B| mapped SNOMED IDs)")
    ax.set_xlabel("Count")
    ax.set_ylabel("Cases")

    ax = axes[0, 1]
    if jac:
        ax.hist(jac, bins=18, color="#1565c0", edgecolor="white", alpha=0.88)
    ax.set_title("Jaccard overlap on concepts\n|A∩B| / |A∪B|")
    ax.set_xlabel("Jaccard")
    ax.set_ylabel("Cases")

    ax = axes[0, 2]
    if ddiff:
        ax.hist(ddiff, bins=22, color="#6a1b9a", edgecolor="white", alpha=0.88)
        ax.axvline(0.0, color="#c62828", linestyle="--", linewidth=1.0, label="B−A = 0")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Mention-weighted mean depth\n(model − human)")
    ax.set_xlabel("Δ depth")
    ax.set_ylabel("Cases")

    ax = axes[1, 0]
    if xs_j and ys_d:
        ax.scatter(xs_j, ys_d, s=28, alpha=0.45, c="#00695c", edgecolors="none")
    ax.set_title("Overlap vs depth gap")
    ax.set_xlabel("Jaccard (concepts)")
    ax.set_ylabel("Mean depth B − A")

    ax = axes[1, 1]
    if xs_surf and ys_inter:
        ax.scatter(xs_surf, ys_inter, s=28, alpha=0.45, c="#e65100", edgecolors="none")
    ax.set_title("Surface vs concept overlap")
    ax.set_xlabel("Same surface string count (both sides)")
    ax.set_ylabel("Shared SNOMED concepts")

    ax = axes[1, 2]
    ma = sum(only_a) / len(only_a) if only_a else 0.0
    mb = sum(only_b) / len(only_b) if only_b else 0.0
    mi = sum(inter) / len(inter) if inter else 0.0
    tot = ma + mb + mi
    if tot > 0:
        labels = ["A-only\nconcepts", "Intersection", "B-only\nconcepts"]
        sizes = [ma / tot, mi / tot, mb / tot]
        colors = ["#0d47a1", "#2e7d32", "#b71c1c"]
        ax.barh([0], [sizes[0]], left=0, color=colors[0], height=0.45, label=labels[0])
        ax.barh([0], [sizes[1]], left=sizes[0], color=colors[1], height=0.45, label=labels[1])
        ax.barh([0], [sizes[2]], left=sizes[0] + sizes[1], color=colors[2], height=0.45, label=labels[2])
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Mean share of concept counts across cases")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=8)
    ax.set_title("Typical concept mass (means)")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)


def write_cross_fig(cross_rows: list[dict[str, str]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not cross_rows:
        return
    gaps = []
    for r in cross_rows:
        g = _i(r.get("depth_gap_abs", ""))
        if g is not None:
            gaps.append(g)
    c: Counter[str] = Counter()
    for r in cross_rows:
        s = (r.get("isa_broader_side") or "").strip()
        if s:
            c[s] += 1

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), constrained_layout=True, facecolor="white")
    fig.suptitle("Cross-side is-a pairs (capped per case)", fontsize=12, fontweight="bold")

    ax = axes[0]
    if gaps:
        ax.hist(gaps, bins=min(20, max(5, len(set(gaps)))), color="#455a64", edgecolor="white", alpha=0.9)
    ax.set_xlabel("|depth_A − depth_B|")
    ax.set_ylabel("Pair rows (all cases)")

    ax = axes[1]
    labels = list(c.keys())
    vals = [c[k] for k in labels]
    if labels:
        ax.barh(labels[::-1], vals[::-1], color="#5d4037", alpha=0.85)
    ax.set_xlabel("Count")
    ax.set_title("Which side is broader on RF2 is-a")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)


def main() -> None:
    ensure_results()
    p = argparse.ArgumentParser(description="Plot per-case A vs B SNOMED summary (+ cross pairs).")
    p.add_argument("--summary-csv", type=Path, default=SNOMED_PER_CASE_AB_SUMMARY)
    p.add_argument("--cross-csv", type=Path, default=SNOMED_PER_CASE_AB_CROSS)
    p.add_argument("--dashboard-png", type=Path, default=SNOMED_PER_CASE_AB_DASHBOARD)
    p.add_argument("--cross-png", type=Path, default=SNOMED_PER_CASE_AB_CROSS_FIG)
    p.add_argument("--no-cross-figure", action="store_true")
    args = p.parse_args()

    if not args.summary_csv.is_file():
        raise SystemExit(f"Missing {args.summary_csv}\nRun: python -m pipeline.per_case_snomed_ab_comparison")

    rows = load_summary(args.summary_csv)
    if not rows:
        raise SystemExit("Summary CSV is empty.")

    write_dashboard(rows, args.dashboard_png)
    print(f"Wrote {args.dashboard_png}", file=sys.stderr)

    if not args.no_cross_figure:
        cross = load_cross(args.cross_csv)
        if cross:
            write_cross_fig(cross, args.cross_png)
            print(f"Wrote {args.cross_png}", file=sys.stderr)
        else:
            print(f"No cross rows at {args.cross_csv}; skip cross figure.", file=sys.stderr)


if __name__ == "__main__":
    main()
