# Proposal vs repository: what is done, what you still do by hand

Your formal proposal (`corpus_proposal_v3-1-1.pdf`) lists **six** analysis steps. Below is the **exact** mapping and how to finish anything missing.

---

## Should you keep only files where A and B are both ≤ 200 words?

**Recommendation: leave the corpus as-is** for the main analysis.

- You already have **116** strict pairs; a hard word cap **drops arbitrary cases**, reduces power, and can **bias** the sample (e.g. you remove long model explanations, which are themselves a finding).
- **Length is already partially controlled** in `prep_corpora` (model target tied to human length), not a universal 200-word ceiling.
- If you want to **talk about length** in the thesis, report **word-count quantiles** (A vs B) in a table or appendix **without deleting files**.

**Optional sensitivity analysis (better than deleting):** define a subset “both ≤ 200 words” in a spreadsheet or small script, re-run **only** lightweight counts on that subset, and add one paragraph: “Results are robust / change when long-B cases are excluded.” Only do this if a supervisor explicitly wants bounded length.

---

## AntConc: copy-paste prompt script (proposal steps 3–5)

Use this as a literal checklist. **Menu names differ slightly** between AntConc 3.x and 4.x; the logic is the same.

### A — Build two files (terminal)

```bash
cd "/home/drk/Desktop/Masters/Corpus Linguistics/Terminology"
mkdir -p antconc_input
cat corpus_a/case_*.txt > antconc_input/corpus_a_merged.txt
cat corpus_b/case_*.txt > antconc_input/corpus_b_merged.txt
wc -w antconc_input/corpus_a_merged.txt antconc_input/corpus_b_merged.txt
```

### B — Keyword list (log-likelihood), B vs A reference

**Goal:** statistically overused lemmas in Corpus B compared to Corpus A.

**AntConc 4.x (typical Flatpak UI):**

1. Start AntConc: `flatpak run org.antconc.AntConc` (from a **desktop** terminal or app menu; if Qt fails on Wayland, use `QT_QPA_PLATFORM=wayland flatpak run org.antconc.AntConc`). See **Open AntConc on Linux** below for display errors.
2. Click the **Keyword List** tab (top row of tools).
3. **File → Open Files** (or toolbar “open”) and select **`corpus_a_merged.txt`** — this becomes the **reference** corpus (AntConc 4 often labels it *File 1* / reference).
4. In the Keyword List panel, use **File → Open** for the **comparison / target** slot (sometimes *File 2*, *comparison corpus*, or a second “open” button in that tab) and choose **`corpus_b_merged.txt`**.
5. In the keyword options, set the statistic to **Log-likelihood** (abbreviated **LL**). Run **Start** / **Calculate** / **Generate keywords** (wording varies by build).
6. When the table appears, **sort by LL** (click column header) so the strongest associations to Corpus B rise to the top.
7. **File → Save** or export the table as CSV; keep the top **40–60** rows; cite the top **10–15** in the dissertation and cross-check with `results/tfidf/shared_terms_mean_tfidf_gap.csv`.

If your window only shows one file slot: open **corpus_a_merged.txt** first from the **main File / Corpus** menu, then open the Keyword List tool—it usually picks up the active corpus as reference and asks for a comparison file in the same tab.

**One-sentence write-up prompt for yourself:** *“List the top ten keywords favouring Corpus B; group them into (i) clinical narrative, (ii) MCQ/meta language, (iii) vague shell nouns; note anything that contradicts the TF–IDF table.”*

### C — Collocations (five lemmas × two corpora)

**Goal:** same lemma, different company words left/right (proposal: specific modifiers vs hedging).

1. Pick **five** lemmas that appear often in **both** merged files (use your keyword list + TF–IDF + a quick search if unsure). Example pattern: *treatment, diagnosis, findings, injury, symptoms* — **replace with your actual high-frequency clinical lemmas**.
2. For **each lemma**, repeat **twice** (once per file):
   - Load **only** `corpus_a_merged.txt` → **Concordance** → search lemma (case-insensitive if available).
   - Open **Collocates** (or “compute collocates” from concordance).
   - Window: **5L, 5R** (five tokens each side).
   - Association measure: **MI** (or **MI3** if MI is sparse—**use the same measure for A and B**).
   - Save/export the **top 15–25** collocates.
3. Switch file to `corpus_b_merged.txt` and repeat the same lemma and same settings.

**One-sentence comparison prompt for yourself (per lemma):** *“For lemma X, list the top five collocates on A vs B; mark + if A shows organ/stage/specifier-rich strings and B shows may/might/suggest/typical.”*

### D — KWIC (concordance lines for evidence)

**Goal:** short quoted lines for the thesis.

1. Take **3–5** high-signal types from step B (e.g. *diagnosis*, *condition*, *treatment*, *dysfunction*, or items your keyword list flags).
2. For each type, **Concordance** in A → sort or filter → **copy 20 lines** to a document.
3. Repeat in B for the **same** type.
4. In the dissertation, **quote 2–4 lines total per type** (A vs B contrast), put the rest in an appendix.

**One-sentence reading prompt for yourself:** *“Underline the smallest span that shows specificity vs vagueness; avoid quoting lines where the case topic is unrelated noise.”*

---

## Already automated in this repo (re-run from project root, venv on)

| Proposal step | Proposal tool | **What you have** | Command(s) |
|----------------|---------------|-------------------|------------|
| 01 Word frequency + Zipf | Freq / Zipf | `freq_zipf` | `python -m pipeline.freq_zipf` |
| 02 TF–IDF | sklearn | `tfidf` | `python -m pipeline.tfidf` |
| 06 SNOMED depth (supporting) | QuickUMLS in proposal | **RF2 + string match + scispaCy NER** (no UMLS) | `python -m pipeline.rf2 --rf2-root "$RF2"` then `python -m pipeline.run_snomed_ner_chain --rf2-root "$RF2"` (or individual modules; see `README.md`) |

**Extra analyses not in the one-page table but implemented:** `hedging`, `hedging_area`, `graph_viz`, `pair_depths`, `branch_pairs`, `snomed_taxonomy_compare`, `llm_pair_align`, `llm_taxonomy_viz`.

---

## You still do this manually (proposal steps 03–05): **AntConc**

The proposal expects **AntConc** for keywords, collocations, and KWIC. The repository does **not** export AntConc project files; you drive AntConc on your machine.

### Open AntConc on Linux (Fedora / Flatpak)

If you installed AntConc as a **Flatpak** (common on Fedora), the **reliable** start is the **desktop launcher** (Activities / Super → type **AntConc** → Enter). That attaches the app to your session’s Wayland/XWayland correctly.

From a terminal, use a **host** terminal (GNOME **Console**, **Kitty**, etc.), **not** Cursor’s integrated terminal—IDE shells often have **no** `DISPLAY` and **no** `WAYLAND_DISPLAY`, so every Qt app fails the same way.

Quick check (run where you launch Flatpak):

```bash
echo "DISPLAY=$DISPLAY WAYLAND_DISPLAY=$WAYLAND_DISPLAY"
```

If both are empty, **do not** keep trying `flatpak` from that shell; open AntConc from the app menu or from **Console** on the laptop.

**GNOME Wayland + Qt defaulting to X11:** if the host terminal shows the warning *Ignoring XDG_SESSION_TYPE=wayland* and then `xcb` / *could not connect to display*, force Wayland and expose the socket:

```bash
QT_QPA_PLATFORM=wayland flatpak run --socket=wayland org.antconc.AntConc
```

If that fails, try `QT_QPA_PLATFORM=wayland-egl` instead of `wayland`.  
Fallback on X11 session or XWayland: `flatpak run org.antconc.AntConc` (often `DISPLAY=:0` is set automatically in a real desktop terminal).

If that errors, try once: `flatpak update org.antconc.AntConc`, then retry.  
Flatseal for `org.antconc.AntConc`: enable **Wayland** (and **X11** fallback if offered) if the app window never appears.

**Qt “could not connect to display” / `xcb` plugin (summary):** no compositor socket reached the process—almost always **wrong terminal** (Cursor/SSH) or **Qt on Wayland without** `QT_QPA_PLATFORM=wayland`.

AntConc needs a **GUI**; there is no useful headless mode for keyword lists.

**Grant file access:** after AntConc opens, use **File → Open** and browse to  
`/home/drk/Desktop/Masters/Corpus Linguistics/Terminology/antconc_input/`  
(or merge again with the commands in section A below). Flatpak may show only certain folders; if the drive is invisible, run AntConc with broader host access (Flatseal → Filesystem → add your `Terminology` folder) or copy `corpus_*_merged.txt` into `~/Documents` and open from there.

### Average length of each corpus (Python — no AntConc)

From the **Terminology** project root:

```bash
source .venv/bin/activate
python -m pipeline.corpus_length_stats
```

Per-case breakdown: `python -m pipeline.corpus_length_stats --per-file`

### 0. Prepare two plain-text corpora

From the **Terminology** project root (adjust path if yours differs):

```bash
cd "/home/drk/Desktop/Masters/Corpus Linguistics/Terminology"
mkdir -p antconc_input
cat corpus_a/case_*.txt > antconc_input/corpus_a_merged.txt
cat corpus_b/case_*.txt > antconc_input/corpus_b_merged.txt
```

You need `corpus_a/` and `corpus_b/` first (`python -m pipeline.prep_corpora`).

### 03 Keyword list (log-likelihood)

1. Launch: `flatpak run org.antconc.AntConc` (or your app menu → **AntConc**).
2. Open the **Keyword List** tab.
3. Load **reference** = `antconc_input/corpus_a_merged.txt` (File → Open in that tab, or toolbar—often *File 1* / reference).
4. Load **comparison / target** = `antconc_input/corpus_b_merged.txt` (second file control in the same tab).
5. Statistic: **Log-likelihood (LL)**. Run calculate / generate.
6. Sort the table by **LL**; **export** CSV (or copy). High LL toward **B** = overused in model text vs human reference—compare with `results/tfidf/shared_terms_mean_tfidf_gap.csv`.

### 04 Collocation profiles

1. Pick **five** lemmas that occur in **both** corpora (from step 03 + TF–IDF + quick search).
2. For **each lemma**, repeat for **A** then **B**:
   - **Concordance** tab → open only `corpus_a_merged.txt` (or use project switcher) → search lemma.
   - **Collocates** tool / button from concordance → span **5L, 5R** → association **MI** (same for B).
3. Export or screenshot top collocates; contrast specific modifiers (A) vs hedging company (B).

### 05 KWIC concordance

1. Choose **3–5** types from step 03 (and/or TF–IDF).
2. **Concordance** tab → **20 lines** per term, **each** corpus; copy to a notes file.
3. Quote **2–4** contrasting lines per term in the thesis; appendix for the rest.

**Longer narrative:** `docs/antconc_qualitative/README.md` (same steps, interpretation notes).

---

## Proposal data design you did **not** use (unless you re-run prep)

The proposal describes **Corpus B** as **without** the correct answer in the prompt.

This repository’s **default** `prep_corpora` is **keyed-correct** (the model is told which option is correct) so both streams stay aligned on the **true** diagnosis for register comparison.

To regenerate **proposal-style** Corpus B (stem + five options only, **no** revealed correct answer):

```bash
cd "/home/drk/Desktop/Masters/Corpus Linguistics/Terminology"
source .venv/bin/activate
python -m pipeline.prep_corpora --omit-correct-option --force
python -m pipeline.prune_corpora_pairs
# then re-run freq_zipf, hedging, tfidf, and the whole SNOMED chain if you want numbers on the new B
```

Use a **separate folder or git branch** if you want to keep both variants; otherwise `--force` overwrites `corpus_b/`.

---

## Proposal step 06 wording (QuickUMLS vs your stack)

The proposal mentions **QuickUMLS** on frequency-list terms. Your implementation uses **SNOMED CT RF2** + **description string matching** + **scispaCy** NER—same *role* (ontology depth as supporting evidence), different tooling. State that explicitly in the dissertation so examiners do not look for UMLS files you never used.

---

## Report PDF

After any figure or number change, from **`report/`**:

```bash
cd "/home/drk/Desktop/Masters/Corpus Linguistics/Terminology/report"
pdflatex -interaction=nonstopmode main && bibtex main && pdflatex main && pdflatex main
```
