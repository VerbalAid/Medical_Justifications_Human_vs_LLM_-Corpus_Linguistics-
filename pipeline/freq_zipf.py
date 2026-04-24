#!/usr/bin/env python3
# Count words (no stopwords), save top 50 CSVs, draw a Zipf-style log-log plot. Optional SNOMED word filter.

from __future__ import annotations
import argparse
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords

from .paths import (
    CORPUS_A,
    CORPUS_B,
    FREQ_SNOMED_HUMAN,
    FREQ_SNOMED_MODEL,
    FREQ_TOP_HUMAN,
    FREQ_TOP_MODEL,
    NLTK_DATA,
    ZIPF_PLOT,
    ensure_results,
)
from .rf2 import (
    discover_rf2_description_and_relationship_files,
    format_rf2_root_missing_error,
    load_rf2_term_to_concept,
)

TOKEN_RE = re.compile(r"\b[a-z0-9]+(?:[-'][a-z0-9]+)*\b", re.IGNORECASE)


def ensure_nltk_stopwords() -> None:
    NLTK_DATA.mkdir(parents=True, exist_ok=True)
    nltk_dir = str(NLTK_DATA)
    if nltk_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_dir)
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", download_dir=nltk_dir, quiet=True)


def read_corpus_dir(folder: Path) -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(folder.glob("case_*.txt")))


def tokenize_filtered(text: str, stops: set[str]) -> list[str]:
    return [m.group(0) for m in TOKEN_RE.finditer(text.lower()) if m.group(0) not in stops]


def load_snomed_token_set(rf2_root: Path) -> set[str]:
    desc_files, _ = discover_rf2_description_and_relationship_files(rf2_root)
    if not desc_files:
        raise SystemExit(f"No sct2_Description_*.txt under {rf2_root}")
    return set(load_rf2_term_to_concept(desc_files).keys())


def ranked_frequencies(counter: Counter[str]) -> list[tuple[str, int]]:
    return sorted(counter.items(), key=lambda x: (-x[1], x[0]))


def write_top_csv(path: Path, ranked: list[tuple[str, int]], top_n: int) -> None:
    lines = ["rank,term,frequency"]
    lines += [f"{r},{term},{freq}" for r, (term, freq) in enumerate(ranked[:top_n], start=1)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_results()
    parser = argparse.ArgumentParser(description="Frequencies and Zipf plot for corpus_a vs corpus_b.")
    parser.add_argument(
        "--rf2-root",
        type=Path,
        default=None,
        help="If set, also write freq_*_medical.csv (tokens that match a SNOMED EN description).",
    )
    args = parser.parse_args()

    ensure_nltk_stopwords()
    stops = set(stopwords.words("english"))

    snomed_terms: set[str] | None = None
    if args.rf2_root is not None:
        rf2_root = args.rf2_root.resolve()
        if not rf2_root.exists():
            raise SystemExit(format_rf2_root_missing_error(rf2_root))
        if not rf2_root.is_dir():
            raise SystemExit(f"RF2 root must be a directory: {rf2_root}")
        print(f"Loading SNOMED EN description strings from {rf2_root} …")
        snomed_terms = load_snomed_token_set(rf2_root)
        print(f"  {len(snomed_terms):,} lowercased strings for filter.")

    text_a = read_corpus_dir(CORPUS_A)
    text_b = read_corpus_dir(CORPUS_B)
    if not text_a.strip() or not text_b.strip():
        raise SystemExit(f"Missing text under {CORPUS_A} and/or {CORPUS_B}. Run prep_corpora first.")

    toks_a = tokenize_filtered(text_a, stops)
    toks_b = tokenize_filtered(text_b, stops)
    rank_a = ranked_frequencies(Counter(toks_a))
    rank_b = ranked_frequencies(Counter(toks_b))

    write_top_csv(FREQ_TOP_HUMAN, rank_a, 50)
    write_top_csv(FREQ_TOP_MODEL, rank_b, 50)
    print(f"Wrote {FREQ_TOP_HUMAN} and {FREQ_TOP_MODEL}")

    if snomed_terms is not None:
        ma = ranked_frequencies(Counter(t for t in toks_a if t in snomed_terms))
        mb = ranked_frequencies(Counter(t for t in toks_b if t in snomed_terms))
        write_top_csv(FREQ_SNOMED_HUMAN, ma, 50)
        write_top_csv(FREQ_SNOMED_MODEL, mb, 50)
        print(f"wrote {FREQ_SNOMED_HUMAN.name} and {FREQ_SNOMED_MODEL.name} (snomed token filter)")

    fa = [c for _, c in rank_a]
    fb = [c for _, c in rank_b]
    ra, rb = list(range(1, len(fa) + 1)), list(range(1, len(fb) + 1))

    plt.figure(figsize=(8, 6))
    plt.loglog(ra, fa, label="Corpus A (human)", alpha=0.85)
    plt.loglog(rb, fb, label="Corpus B (LLM)", alpha=0.85)
    plt.xlabel("log(rank)")
    plt.ylabel("log(frequency)")
    plt.title("Zipf-style rank–frequency (stopwords removed)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ZIPF_PLOT, dpi=150)
    plt.close()
    print(f"saved {ZIPF_PLOT}")


if __name__ == "__main__":
    main()
