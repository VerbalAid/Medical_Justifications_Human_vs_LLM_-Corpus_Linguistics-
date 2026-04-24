# Qualitative follow-up: three pipeline visuals, then AntConc

**British English** in your write-up is fine. Use this page to **interpret** the main figures, then scroll to **Open AntConc** for keywords, collocations, and KWIC.

---

## 1. SNOMED subgraph (`results/snomed/plot_subgraph_high_depth_ner.png`)

High density is normal for a **full** ancestor closure. Prefer `python -m pipeline.graph_viz --compact` (or `--ancestor-hops 12 --top 5 --label-seeds-only`) for a readable slice; only seed nodes get text labels in that mode.

**Worth stating:** **Red** (Corpus B only) nodes are spread through the hierarchy, including **shallow** levels; **blue** (A only) tends to sit in **more specific** pockets. That pattern supports a “model more generic / human more anchored” story **alongside** other evidence — not on its own.

---

## 2. Depth difference histogram (`results/snomed/hist_depth_difference_b_minus_a.png`)

Produced by `pair_depths`: it pairs rows where the **same entity string** appears in A and B.

If the histogram is a **single spike at 0**, that is **expected**, not a bug: one string → one SNOMED concept → one depth. You are **not** comparing different-but-related concepts on the same branch.

**For pairwise analysis across the taxonomy**, use **`python -m pipeline.branch_pairs`** (LCA, hop limit, category breakdown). That is the right tool for “how do A and B entities relate on SNOMED branches?” — not the shared-string histogram.

---

## 3. Zipf plot (`results/corpus_frequency/zipf_rank_frequency_loglog.png`)

Corpus B often sits **above** A at most ranks because **B is longer** (more tokens): higher rank–frequency at each rank is largely **size**, not a special “shape” discovery.

**Similar slopes** ⇒ similar **overall** rank–frequency **shape** (Zipf-ish tail).

**What to foreground for over-generalisation:** the **heads** of the lists — open **`results/corpus_frequency/human_top50_token_freq.csv`** and **`model_top50_token_freq.csv`** and compare the **top 10** (or top 20) terms. Cross-check with **`results/tfidf/shared_terms_mean_tfidf_gap.csv`** and **`results/hedging/hedge_density_both_corpora.csv`**.

---

## Open AntConc

Prerequisite: `corpus_a/` and `corpus_b/` exist (`python -m pipeline.prep_corpora`).

From the project root:

```bash
cd "/home/drk/Desktop/Masters/Corpus Linguistics/Terminology"
```

**Optional — one file per corpus** (easier loading in AntConc):

```bash
mkdir -p antconc_input
cat corpus_a/case_*.txt > antconc_input/corpus_a_merged.txt
cat corpus_b/case_*.txt > antconc_input/corpus_b_merged.txt
```

Use either the **125 files per folder** or the two **merged** `.txt` files.

---

### Keyword list (log-likelihood)

**Aim:** Terms **over-represented in Corpus B** vs Corpus A (reference).

1. Load **Corpus A** as **reference**.
2. **Keyword list** tool → **target / comparison** = Corpus B.
3. Statistic: **log-likelihood** (LL).
4. Export the ranked list. Values favouring B = stronger association with the model than with humans.

Often includes generic framing (*condition*, *symptoms*, *treatment*, *typically*, …) — use next to your freq heads and TF–IDF gap.

---

### Collocations on focal terms

Pick **five** lemmas that appear in **both** corpora (e.g. *treatment*, *diagnosis*, *dysfunction*, *injury*, *symptoms*).

**Per term, per corpus:** Concordance → **Collocates** → span **5L, 5R** (or your proposal’s window) → one association measure (**MI**, **MI3**, **T-score** — keep it the same for A and B).

**Look for:** A — specific modifiers (*antibiotic treatment*, *acute injury*). B — hedging (*treatment may*, *symptoms often suggest*).

---

### KWIC concordance lines

For the strongest terms, export roughly **20 lines per corpus** per term. Contrast usage in prose (e.g. A ties *dysfunction* to an organ; B uses vaguer modifiers) — quotable for the dissertation.

---

## Quick copy-paste

```bash
cd "/home/drk/Desktop/Masters/Corpus Linguistics/Terminology"
mkdir -p antconc_input
cat corpus_a/case_*.txt > antconc_input/corpus_a_merged.txt
cat corpus_b/case_*.txt > antconc_input/corpus_b_merged.txt
```

Then AntConc: **keywords** (A ref, B target, LL) → **collocations** (five terms, ±5) → **KWIC** (~20 lines each).

---

## Related automated outputs

- `python -m pipeline.freq_zipf` — `results/corpus_frequency/`  
- `python -m pipeline.tfidf` — `results/tfidf/shared_terms_mean_tfidf_gap.csv`  
- `python -m pipeline.hedging` — `results/hedging/hedge_density_both_corpora.csv`  
- `python -m pipeline.branch_pairs` — branch-wise pairwise SNOMED analysis  

Full CLI list: `docs/command_prompts.md`.
