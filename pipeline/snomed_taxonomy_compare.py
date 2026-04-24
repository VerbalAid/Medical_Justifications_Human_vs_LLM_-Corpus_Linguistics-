#!/usr/bin/env python3
"""
One picture per SNOMED top-level category, plus one overview strip (all JPEG).

Where the rows come from
------------------------
``branch_pairs`` builds *many* rows: every NER entity from human texts (A)
paired with every NER entity from model texts (B) that share a least common
ancestor (LCA) on the SNOMED is-a tree within a hop limit. The strings do
**not** have to match (that is a different script: ``pair_depths``).

So "drug" on the human side and "rt-PA" on the model side can appear together
because they are one A×B pair from the CSV that had a large depth difference
for that category — not because the pipeline thinks they are "the same"
clinical mention.

What this script picks
----------------------
For each top-level hierarchy (Substance, Procedure, …), it keeps **one** row:
the pair with the largest |depth_a − depth_b|. That is only a **summary
example** per category, not a full alignment of corpora.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import textwrap
from pathlib import Path

from .branch_pairs import ancestor_closure, hierarchy_category_for_concept
from .paths import (
    DEFAULT_RF2_ROOT,
    DIR_TAXONOMY_JPEG,
    SNOMED_BRANCH_PAIRS,
    SNOMED_TAXONOMY_OVERVIEW,
    ensure_results,
)
from .rf2 import (
    RF2_FSN_TYPE_ID,
    RF2_SYNONYM_TYPE_ID,
    discover_rf2_description_and_relationship_files,
    format_rf2_root_missing_error,
    load_rf2_isa_parent_map,
)

DEFAULT_BRANCH_PAIRS = SNOMED_BRANCH_PAIRS
DEFAULT_JPEG_DIR = DIR_TAXONOMY_JPEG
OUT_OVERVIEW_JPEG = SNOMED_TAXONOMY_OVERVIEW

# JPEG is smaller than PNG; quality 90 is usually fine for slides/prints.
JPEG_KWARGS = {"quality": 90, "optimize": True}


def _csv_field_limit() -> None:
    try:
        csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    except OverflowError:
        csv.field_size_limit(2**31 - 1)


def load_rf2_labels(description_paths: list[Path], want: set[str]) -> dict[str, str]:
    if not want:
        return {}
    _csv_field_limit()
    best: dict[str, tuple[str, int]] = {}
    rank_fsn, rank_syn = 0, 1
    for path in description_paths:
        with path.open(encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            try:
                header = next(reader)
            except StopIteration:
                continue
            col = {name: i for i, name in enumerate(header)}
            need = ("conceptId", "languageCode", "typeId", "term", "active")
            if not all(k in col for k in need):
                continue
            ci, lc, ti, te, ac = (col[k] for k in need)
            for row in reader:
                if max(ci, lc, ti, te, ac) >= len(row):
                    continue
                if row[ac] != "1":
                    continue
                cid = row[ci]
                if cid not in want:
                    continue
                if not row[lc].strip().lower().startswith("en"):
                    continue
                typ = row[ti]
                if typ not in (RF2_FSN_TYPE_ID, RF2_SYNONYM_TYPE_ID):
                    continue
                key = row[te].strip().lower()
                if not key:
                    continue
                rank = rank_fsn if typ == RF2_FSN_TYPE_ID else rank_syn
                prev = best.get(cid)
                if prev is None or rank < prev[1]:
                    best[cid] = (key, rank)
    return {cid: t for cid, (t, _) in best.items()}


def shortest_path_up(
    start: str,
    goal: str,
    parents: dict[str, list[str]],
    max_len: int = 48,
) -> list[str] | None:
    from collections import deque

    q: deque[tuple[str, list[str]]] = deque([(start, [start])])
    seen = {start}
    while q:
        node, path = q.popleft()
        if node == goal:
            return path
        if len(path) > max_len:
            continue
        for p in parents.get(node, []):
            if p in seen:
                continue
            seen.add(p)
            npath = path + [p]
            if p == goal:
                return npath
            q.append((p, npath))
    return None


def category_slug(cat: str) -> str:
    s = cat.lower().replace("/", " ").replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return (s[:60] if s else "category")


def category_for_lca(
    lca: str,
    parents: dict[str, list[str]],
    anc_cache: dict[str, frozenset[str]],
) -> str:
    if lca not in anc_cache:
        anc_cache[lca] = ancestor_closure(lca, parents)
    return hierarchy_category_for_concept(lca, anc_cache[lca], parents)


def pick_divergent_pairs_per_category(
    branch_csv: Path,
    parents: dict[str, list[str]],
    min_abs_diff: int,
    max_rows: int | None = None,
) -> list[dict[str, str]]:
    """
    One row per SNOMED top-level group: the row with the biggest |depth_a-depth_b|.
    """
    best: dict[str, dict] = {}
    anc_cache: dict[str, frozenset[str]] = {}
    with branch_csv.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for lineno, row in enumerate(reader):
            if max_rows is not None and lineno >= max_rows:
                break
            try:
                da = int(row["depth_a"])
                db = int(row["depth_b"])
            except (KeyError, ValueError):
                continue
            ca = (row.get("concept_a") or "").strip()
            cb = (row.get("concept_b") or "").strip()
            lca = (row.get("lca_concept") or "").strip()
            if not ca or not cb or not lca or ca == cb:
                continue
            if da == db:
                continue
            if abs(da - db) < min_abs_diff:
                continue
            cat = category_for_lca(lca, parents, anc_cache)
            score = abs(da - db)
            prev = best.get(cat)
            if prev is None or score > int(prev["_score"]):
                row["_score"] = str(score)
                row["_category"] = cat
                best[cat] = row
    return sorted(best.values(), key=lambda r: r["_category"])


def compact_top_down(path_td: list[str]) -> list[tuple[str, str | None, int | None]]:
    k = len(path_td)
    if k == 0:
        return []
    if k <= 6:
        out: list[tuple[str, str | None, int | None]] = []
        for i, cid in enumerate(path_td):
            if i == 0:
                out.append(("lca", cid, None))
            elif i == k - 1:
                out.append(("focal", cid, None))
            else:
                out.append(("anc", cid, None))
        return out
    skip_lo, skip_hi = 2, k - 4
    n_skip = skip_hi - skip_lo + 1 if skip_hi >= skip_lo else 0
    out = [("lca", path_td[0], None), ("anc", path_td[1], None)]
    if n_skip > 0:
        out.append(("ellipsis", None, n_skip))
    out.append(("anc", path_td[k - 3], None))
    out.append(("anc", path_td[k - 2], None))
    out.append(("focal", path_td[k - 1], None))
    return out


def _trunc(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[: n - 1] + "\u2026"


def comparison_label(da: int, db: int) -> str:
    diff = da - db
    if diff > 0:
        return f"A more specific by {diff} levels"
    if diff < 0:
        return f"B more specific by {-diff} levels"
    return "Same depth"


def compute_pair_visual(
    r: dict[str, str],
    labels: dict[str, str],
    parents: dict[str, list[str]],
) -> dict[str, object] | None:
    ca, cb = r["concept_a"].strip(), r["concept_b"].strip()
    lca = r["lca_concept"].strip()
    da, db = int(r["depth_a"]), int(r["depth_b"])
    path_a = shortest_path_up(ca, lca, parents)
    path_b = shortest_path_up(cb, lca, parents)
    if not path_a or not path_b:
        return None
    top_a = list(reversed(path_a))
    top_b = list(reversed(path_b))
    items_a = compact_top_down(top_a)
    items_b = compact_top_down(top_b)
    ea, eb = r.get("entity_a", ""), r.get("entity_b", "")
    cat = r.get("_category", "")
    title = f'{cat}: "{ea}" (human) vs "{eb}" (model)'
    mid = comparison_label(da, db)
    return {
        "category": cat,
        "title": title,
        "mid": mid,
        "items_a": items_a,
        "items_b": items_b,
        "ea": ea,
        "eb": eb,
        "da": da,
        "db": db,
    }


def write_category_jpeg_figure(vis: dict[str, object], labels: dict[str, str], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    items_a = vis["items_a"]
    items_b = vis["items_b"]
    assert isinstance(items_a, list) and isinstance(items_b, list)

    da, db = int(vis["da"]), int(vis["db"])
    depth_band = (
        f"SNOMED is-a depth: human {da} | model {db} | {str(vis['mid'])}"
    )

    ctx_a = str(vis.get("context_a") or "").strip()
    ctx_b = str(vis.get("context_b") or "").strip()
    has_ctx = bool(ctx_a or ctx_b)
    fig = plt.figure(figsize=(10.8, 7.4 if has_ctx else 7.0), facecolor="white")
    fig.subplots_adjust(
        top=0.86, left=0.06, right=0.94, bottom=0.16 if has_ctx else 0.06, wspace=0.12
    )
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0])
    ax_l = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])

    fig.suptitle(str(vis["title"]), fontsize=11, fontweight="600", color="#111111", y=0.96)
    fig.text(0.5, 0.905, depth_band, ha="center", va="top", fontsize=10, color="#424242")

    def _column(
        ax,
        items: list[tuple[str, str | None, int | None]],
        side: str,
        entity: str,
        depth: int,
    ) -> None:
        ax.set_facecolor("#fafafa")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        col = "#1565c0" if side == "A" else "#c62828"
        banner = f"{'Human (A)' if side == 'A' else 'Model (B)'} — SNOMED depth {depth}"
        ax.text(0.5, 0.97, banner, ha="center", va="top", fontsize=9.8, fontweight="bold", color=col)
        y = 0.90
        gap = 0.017
        hmap = {"lca": 0.11, "ellipsis": 0.085, "anc": 0.095, "focal": 0.24}
        for kind, cid, n_skip in items:
            h = hmap.get(kind, 0.1)
            y_bot = y - h
            if y_bot < 0.02:
                break
            if kind == "lca":
                raw = labels.get(str(cid or ""), "") or "Common ancestor"
                lab = _trunc(raw, 48)
                fc, ec, lw = "#e6e6e6", "#9e9e9e", 0.9
                fs = 8.6
            elif kind == "ellipsis":
                lab = f"... {n_skip} intermediate levels ..."
                fc, ec, lw = "#f2f2f2", "#bdbdbd", 0.8
                fs = 8.0
            elif kind == "anc":
                assert cid is not None
                lab = _trunc(labels.get(cid, cid), 46)
                fc, ec, lw = "#ffffff", "#bdbdbd", 0.85
                fs = 8.5
            else:
                assert cid is not None
                snomed_line = _trunc(labels.get(cid, cid), 46)
                lab = f"\u201c{entity}\u201d\n{snomed_line}"
                fc = "#e3f2fd" if side == "A" else "#ffebee"
                ec = "#1565c0" if side == "A" else "#c62828"
                lw = 1.35
                fs = 8.4
            patch = FancyBboxPatch(
                (0.05, y_bot),
                0.9,
                h,
                boxstyle="round,pad=0.006,rounding_size=0.018",
                linewidth=lw,
                edgecolor=ec,
                facecolor=fc,
                transform=ax.transAxes,
            )
            ax.add_patch(patch)
            ax.text(
                0.5,
                y_bot + h * 0.5,
                lab,
                ha="center",
                va="center",
                fontsize=fs,
                color="#212121",
                transform=ax.transAxes,
                linespacing=1.2,
            )
            y = y_bot - gap

    _column(ax_l, items_a, "A", str(vis["ea"]), da)
    _column(ax_r, items_b, "B", str(vis["eb"]), db)
    if has_ctx:
        ca = _trunc(f"Human context: {ctx_a}", 200) if ctx_a else ""
        cb = _trunc(f"Model context: {ctx_b}", 200) if ctx_b else ""
        y0 = 0.078 if (ctx_a and ctx_b) else 0.055
        if ca:
            fig.text(
                0.5,
                y0,
                ca,
                ha="center",
                va="bottom",
                fontsize=7.1,
                color="#37474f",
            )
        if cb:
            fig.text(
                0.5,
                0.028 if ca else y0,
                cb,
                ha="center",
                va="bottom",
                fontsize=7.1,
                color="#37474f",
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path,
        format="jpeg",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        pil_kwargs=JPEG_KWARGS,
    )
    plt.close(fig)


def _anc_three(cid: str, parents: dict[str, list[str]], labels: dict[str, str]) -> tuple[str, str, str]:
    chain = [cid]
    cur = cid
    seen = {cid}
    for _ in range(30):
        ps = parents.get(cur)
        if not ps:
            break
        nxt = ps[0]
        if nxt in seen:
            break
        seen.add(nxt)
        chain.append(nxt)
        cur = nxt
    a1 = labels.get(chain[1], chain[1]) if len(chain) > 1 else "\u2014"
    a2 = labels.get(chain[2], chain[2]) if len(chain) > 2 else "\u2014"
    a3 = labels.get(chain[3], chain[3]) if len(chain) > 3 else "\u2014"
    return a1, a2, a3


def _print_text_summary(
    pairs: list[dict[str, str]],
    parents: dict[str, list[str]],
    labels: dict[str, str],
) -> None:
    for r in pairs:
        cat = r.get("_category", "")
        ea, eb = r.get("entity_a", ""), r.get("entity_b", "")
        ca, cb = r["concept_a"].strip(), r["concept_b"].strip()
        da, db = int(r["depth_a"]), int(r["depth_b"])
        lca = r["lca_concept"].strip()
        la = labels.get(ca, ca)
        lb = labels.get(cb, cb)
        aa1, aa2, aa3 = _anc_three(ca, parents, labels)
        ba1, ba2, ba3 = _anc_three(cb, parents, labels)
        print(f"\n[{cat}]")
        print(f"  A  entity={ea!r}  concept={la!r}  depth={da}")
        print(f"      ancestors (up to 3): {aa1} \u2190 {aa2} \u2190 {aa3}")
        print(f"  B  entity={eb!r}  concept={lb!r}  depth={db}")
        print(f"      ancestors (up to 3): {ba1} \u2190 {ba2} \u2190 {ba3}")
        print(f"  LCA: {labels.get(lca, lca)}  |  depth diff (A\u2212B) = {da - db}")


def write_overview_jpeg(
    pairs: list[dict[str, str]],
    out_path: Path,
    *,
    intro_lines: list[str] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n = max(1, len(pairs))
    has_any_ctx = any(
        (str(r.get("context_a") or "").strip() or str(r.get("context_b") or "").strip())
        for r in pairs
    )
    intro_lines = [ln for ln in (intro_lines or []) if (ln or "").strip()]
    intro_wrapped_lines: list[str] = []
    intro_wrap_width = 96
    for para in intro_lines:
        intro_wrapped_lines.extend(
            textwrap.wrap(
                para.strip(),
                width=intro_wrap_width,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
    # When the intro already defines SUBSTITUTE / INCOMPARABLE, skip repeating long hints per row.
    compact_row_legend = bool(intro_wrapped_lines)
    # Two-column context layout needs a bit more vertical room than stacked-centre snippets.
    row_h = 3.85 if has_any_ctx else 2.35
    intro_h = 0.0
    if intro_wrapped_lines:
        nl = max(1, len(intro_wrapped_lines))
        intro_h = min(3.65, 0.30 + 0.175 * nl)
    fig_h = intro_h + row_h * n
    if intro_wrapped_lines:
        fig, axes = plt.subplots(
            n + 1,
            1,
            figsize=(12.2, fig_h),
            height_ratios=[intro_h / row_h] + [1.0] * n,
            facecolor="white",
        )
        intro_ax = axes[0]
        intro_ax.axis("off")
        intro_ax.set_xlim(0, 1)
        intro_ax.set_ylim(0, 1)
        intro_ax.set_facecolor("#f7f5f0")
        intro_nl = len(intro_wrapped_lines)
        intro_fs = 8.35 if intro_nl > 10 else (8.65 if intro_nl > 7 else 9.0)
        intro_ax.add_patch(
            Rectangle(
                (0.012, 0.04),
                0.976,
                0.92,
                transform=intro_ax.transAxes,
                fill=True,
                facecolor="#faf8f4",
                edgecolor="#c9bfb0",
                linewidth=1.0,
                zorder=0,
            )
        )
        intro_ax.text(
            0.028,
            0.94,
            "\n".join(intro_wrapped_lines),
            fontsize=intro_fs,
            color="#2e2216",
            transform=intro_ax.transAxes,
            va="top",
            ha="left",
            linespacing=1.42,
            zorder=1,
        )
        panel_axes = list(axes[1:])
    else:
        fig, ax_all = plt.subplots(n, 1, figsize=(12.2, row_h * n), facecolor="white")
        panel_axes = [ax_all] if n == 1 else list(ax_all)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(hspace=0.22, top=0.98, bottom=0.02)

    hints = {
        "INCOMPARABLE": (
            "INCOMPARABLE: the reviewer judged these two highlighted spans an unfair clinical pairing "
            "(e.g. different angles on the case). It does not mean broken extraction or invalid SNOMED IDs."
        ),
        "SUBSTITUTE": (
            "SUBSTITUTE: the reviewer replaced the surface strings so SNOMED depths compare the intended concepts."
        ),
        "KEEP": "KEEP: the original span pair was accepted for comparison.",
    }

    for ax, r in zip(panel_axes, pairs, strict=False):
        cat = (r.get("_category") or "").strip()
        ea, eb = r.get("entity_a", ""), r.get("entity_b", "")
        da, db = int(r["depth_a"]), int(r["depth_b"])
        diff = da - db
        arrow = f"A deeper by {diff}" if diff > 0 else (f"B deeper by {-diff}" if diff < 0 else "Same depth")
        ctx_a = str(r.get("context_a") or "").strip()
        ctx_b = str(r.get("context_b") or "").strip()
        ax.set_facecolor("#f5f5f5")
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        band = f"Depth: human {da}  ·  model {db}  ·  {arrow}"
        dec = (r.get("_llm_decision") or "").strip().upper()
        hint = hints.get(dec, "")
        badge_color = "#5d4037"
        if dec == "SUBSTITUTE":
            badge_color = "#1b5e20"
        elif dec == "INCOMPARABLE":
            badge_color = "#4e342e"
        elif dec == "KEEP":
            badge_color = "#1565c0"

        if compact_row_legend:
            ax.text(0.02, 0.96, cat, fontsize=11.5, fontweight="bold", color="#111", transform=ax.transAxes, va="top")
            if dec:
                ax.text(
                    0.98,
                    0.96,
                    dec,
                    fontsize=9.0,
                    fontweight="bold",
                    color=badge_color,
                    transform=ax.transAxes,
                    va="top",
                    ha="right",
                )
            ax.text(0.02, 0.80, band, fontsize=9.6, color="#424242", transform=ax.transAxes, va="top")
            y_block_top = 0.72
            wrap_w = 46
            ctx_a_wrapped = "\n".join(textwrap.wrap(ctx_a, wrap_w)) if ctx_a else "—"
            ctx_b_wrapped = "\n".join(textwrap.wrap(ctx_b, wrap_w)) if ctx_b else "—"
            ax.plot(
                [0.5, 0.5],
                [0.04, y_block_top + 0.02],
                color="#cfd8dc",
                linewidth=0.9,
                transform=ax.transAxes,
                zorder=0,
            )
            ax.text(
                0.25,
                y_block_top,
                f"Human (A): {ea}",
                fontsize=10.8,
                color="#0d47a1",
                transform=ax.transAxes,
                va="top",
                ha="center",
                fontweight="bold",
                zorder=1,
            )
            ax.text(
                0.75,
                y_block_top,
                f"Model (B): {eb}",
                fontsize=10.8,
                color="#b71c1c",
                transform=ax.transAxes,
                va="top",
                ha="center",
                fontweight="bold",
                zorder=1,
            )
            ax.text(
                0.25,
                y_block_top - 0.14,
                ctx_a_wrapped,
                fontsize=8.35,
                color="#37474f",
                transform=ax.transAxes,
                va="top",
                ha="center",
                linespacing=1.22,
                zorder=1,
            )
            ax.text(
                0.75,
                y_block_top - 0.14,
                ctx_b_wrapped,
                fontsize=8.35,
                color="#37474f",
                transform=ax.transAxes,
                va="top",
                ha="center",
                linespacing=1.22,
                zorder=1,
            )
            continue

        if hint:
            wrapped = "\n".join(textwrap.wrap(hint, 108))
            ax.text(0.02, 0.94, wrapped, fontsize=7.35, color="#5d4037", transform=ax.transAxes, va="top")
            ax.text(0.02, 0.72, cat, fontsize=10.5, fontweight="bold", color="#111", transform=ax.transAxes, va="top")
            ax.text(0.02, 0.64, band, fontsize=9.0, color="#555", transform=ax.transAxes, va="top")
            y_block_top = 0.46
            # Legacy stacked layout for branch-pairs overview (no intro).
            ax.text(0.02, y_block_top, f"Human (A): {ea}", fontsize=9.8, color="#1565C0", transform=ax.transAxes, va="top")
            ax.text(
                0.98,
                y_block_top,
                f"Model (B): {eb}",
                fontsize=9.8,
                color="#c62828",
                transform=ax.transAxes,
                va="top",
                ha="right",
            )
            y_ctx = 0.28
            if ctx_a and ctx_b:
                ax.text(
                    0.5,
                    y_ctx,
                    _trunc(f"Context (A): {ctx_a}", 118),
                    fontsize=7.35,
                    color="#455a64",
                    transform=ax.transAxes,
                    va="top",
                    ha="center",
                )
                ax.text(
                    0.5,
                    max(0.06, y_ctx - 0.14),
                    _trunc(f"Context (B): {ctx_b}", 118),
                    fontsize=7.35,
                    color="#455a64",
                    transform=ax.transAxes,
                    va="top",
                    ha="center",
                )
            elif ctx_a:
                ax.text(
                    0.5,
                    y_ctx,
                    _trunc(f"Context (A): {ctx_a}", 124),
                    fontsize=7.35,
                    color="#455a64",
                    transform=ax.transAxes,
                    va="top",
                    ha="center",
                )
            elif ctx_b:
                ax.text(
                    0.5,
                    y_ctx,
                    _trunc(f"Context (B): {ctx_b}", 124),
                    fontsize=7.35,
                    color="#455a64",
                    transform=ax.transAxes,
                    va="top",
                    ha="center",
                )
            continue
        else:
            ax.text(0.02, 0.92, cat, fontsize=10.5, fontweight="bold", color="#111", transform=ax.transAxes, va="top")
            ax.text(0.02, 0.84, band, fontsize=9.0, color="#555", transform=ax.transAxes, va="top")
            y_ent, y_ctx = 0.62, 0.42
            ax.text(0.02, y_ent, f"Human (A): {ea}", fontsize=9.8, color="#1565C0", transform=ax.transAxes, va="top")
            ax.text(
                0.98,
                y_ent,
                f"Model (B): {eb}",
                fontsize=9.8,
                color="#c62828",
                transform=ax.transAxes,
                va="top",
                ha="right",
            )
            if ctx_a and ctx_b:
                ax.text(
                    0.5,
                    y_ctx,
                    _trunc(f"Context (A): {ctx_a}", 118),
                    fontsize=7.35,
                    color="#455a64",
                    transform=ax.transAxes,
                    va="top",
                    ha="center",
                )
                ax.text(
                    0.5,
                    max(0.06, y_ctx - 0.14),
                    _trunc(f"Context (B): {ctx_b}", 118),
                    fontsize=7.35,
                    color="#455a64",
                    transform=ax.transAxes,
                    va="top",
                    ha="center",
                )
            elif ctx_a:
                ax.text(
                    0.5,
                    y_ctx,
                    _trunc(f"Context (A): {ctx_a}", 124),
                    fontsize=7.35,
                    color="#455a64",
                    transform=ax.transAxes,
                    va="top",
                    ha="center",
                )
            elif ctx_b:
                ax.text(
                    0.5,
                    y_ctx,
                    _trunc(f"Context (B): {ctx_b}", 124),
                    fontsize=7.35,
                    color="#455a64",
                    transform=ax.transAxes,
                    va="top",
                    ha="center",
                )
            continue

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        out_path,
        format="jpeg",
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
        edgecolor="none",
        pil_kwargs=JPEG_KWARGS,
    )
    plt.close()
    print(f"Wrote {out_path}")


def main() -> None:
    ensure_results()
    p = argparse.ArgumentParser(
        description="Pick one divergent A×B pair per SNOMED category; save JPEG figures."
    )
    p.add_argument("--rf2-root", type=Path, default=DEFAULT_RF2_ROOT)
    p.add_argument("--branch-pairs", type=Path, default=DEFAULT_BRANCH_PAIRS)
    p.add_argument(
        "--jpeg-dir",
        type=Path,
        default=DEFAULT_JPEG_DIR,
        help="Folder for taxonomy_01_....jpg files.",
    )
    p.add_argument(
        "--no-category-jpeg",
        action="store_true",
        help="Skip per-category pictures (not usually what you want).",
    )
    p.add_argument("--overview-jpeg", type=Path, default=OUT_OVERVIEW_JPEG, help="Combined summary strip.")
    p.add_argument("--no-overview-jpeg", action="store_true", help="Skip combined summary strip.")
    p.add_argument(
        "--overview-max",
        type=int,
        default=None,
        metavar="N",
        help="If set, include at most N rows in the overview strip (per-category JPEGs unchanged).",
    )
    p.add_argument("--min-abs-depth-diff", type=int, default=1)
    p.add_argument("--max-rows", type=int, default=None)
    args = p.parse_args()

    if not args.branch_pairs.is_file():
        raise SystemExit(
            f"Missing branch-pairs file:\n  {args.branch_pairs}\n\n"
            "That CSV is built after NER. Run:\n"
            f"  {sys.executable} -m pipeline.branch_pairs --rf2-root <RF2> --dedupe-concept-pairs\n\n"
            "Or run the full chain (NER → branch_pairs → this step):\n"
            f"  {sys.executable} -m pipeline.run_snomed_ner_chain --rf2-root <RF2>\n"
        )

    rf2_root = args.rf2_root.resolve()
    if not rf2_root.exists():
        raise SystemExit(format_rf2_root_missing_error(rf2_root))

    desc_files, rel_files = discover_rf2_description_and_relationship_files(rf2_root)
    if not desc_files or not rel_files:
        raise SystemExit("Need RF2 Description and Relationship files.")

    print("Loading is-a map …", file=sys.stderr)
    parents = load_rf2_isa_parent_map(rel_files)

    print("Scanning branch pairs …", file=sys.stderr)
    pairs = pick_divergent_pairs_per_category(
        args.branch_pairs, parents, args.min_abs_depth_diff, args.max_rows
    )
    if not pairs:
        raise SystemExit(
            "No divergent concept pairs found. Try a lower --min-abs-depth-diff or regenerate branch_pairs."
        )

    want: set[str] = set()
    for r in pairs:
        for k in ("concept_a", "concept_b", "lca_concept"):
            want.add(r[k].strip())
        for ca_s, lca_s in ((r["concept_a"], r["lca_concept"]), (r["concept_b"], r["lca_concept"])):
            pa = shortest_path_up(ca_s.strip(), lca_s.strip(), parents)
            if pa:
                want.update(pa)

    print("Loading EN labels …", file=sys.stderr)
    labels = load_rf2_labels(desc_files, want)

    args.jpeg_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for i, r in enumerate(pairs, start=1):
        slug = category_slug(r.get("_category", "category"))
        vis = compute_pair_visual(r, labels, parents)
        if vis is None:
            continue
        if not args.no_category_jpeg:
            jpath = args.jpeg_dir / f"taxonomy_{i:02d}_{slug}.jpg"
            write_category_jpeg_figure(vis, labels, jpath)
            written.append(jpath)

    _print_text_summary(pairs, parents, labels)
    for path in written:
        print(path, file=sys.stderr)
    print(f"Wrote {len(written)} JPEG(s) under {args.jpeg_dir}", file=sys.stderr)

    if not args.no_overview_jpeg:
        overview_pairs = pairs
        if args.overview_max is not None and args.overview_max > 0:
            overview_pairs = pairs[: args.overview_max]
        write_overview_jpeg(overview_pairs, args.overview_jpeg)


if __name__ == "__main__":
    main()
