#!/usr/bin/env python3
# TF-IDF on A and on B separately (one document = one case file). CSV of shared types, mean score A minus B.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .paths import CORPUS_A, CORPUS_B, TFIDF_GAP, ensure_results


def load_documents(folder: Path):
    return [p.read_text(encoding="utf-8") for p in sorted(folder.glob("case_*.txt"))]


def column_means(X):
    return np.asarray(X.mean(axis=0)).ravel()


def main() -> None:
    ensure_results()
    docs_a = load_documents(CORPUS_A)
    docs_b = load_documents(CORPUS_B)
    if len(docs_a) != len(docs_b):
        print(f"Warning: |corpus_a|={len(docs_a)} vs |corpus_b|={len(docs_b)}")

    vec_a = TfidfVectorizer(lowercase=True)
    vec_b = TfidfVectorizer(lowercase=True)
    Xa = vec_a.fit_transform(docs_a)
    Xb = vec_b.fit_transform(docs_b)

    names_a = vec_a.get_feature_names_out()
    names_b = vec_b.get_feature_names_out()
    mean_a = column_means(Xa)
    mean_b = column_means(Xb)

    map_a = {t: float(mean_a[i]) for i, t in enumerate(names_a)}
    map_b = {t: float(mean_b[i]) for i, t in enumerate(names_b)}

    rows = []
    for term in sorted(set(map_a) & set(map_b)):
        va, vb = map_a[term], map_b[term]
        rows.append(
            {"term": term, "mean_tfidf_a": va, "mean_tfidf_b": vb, "difference_a_minus_b": va - vb}
        )
    rows.sort(key=lambda r: r["difference_a_minus_b"], reverse=True)

    pd.DataFrame(rows).to_csv(TFIDF_GAP, index=False)
    print(f"wrote {TFIDF_GAP} ({len(rows)} shared terms)")


if __name__ == "__main__":
    main()
