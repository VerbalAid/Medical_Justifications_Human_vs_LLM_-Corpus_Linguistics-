# Medical Justifications: Human vs. LLM (Corpus Linguistics)

A corpus-linguistic audit comparing expert-written MCQ justifications with Mistral-generated rewrites on the CasiMedicos English test split. The analysis examines register divergence, epistemic hedging, patient framing, and lexical distribution across 116 parallel text pairs.

## Overview

Human justifications from the CasiMedicos dataset read as exam-facing justification — first-person, assertive, metalinguistic. Mistral rewrites of the same cases read as explanatory clinical prose — possessive patient framing, more hedging, longer. The study argues that this difference is better explained by register divergence than simplification, and that lexical overlap metrics such as BLEU and ROUGE are weak proxies for clinical quality unless the register of the gold standard is stated explicitly.

## Methods

- Log-likelihood keyword analysis (AntConc)
- TF–IDF on shared lemmas
- Mutual information collocations
- KWIC concordance
- Epistemic hedging lexicon (markers per 1,000 tokens)
- SNOMED CT RF2 ontology linkage via scispaCy
- Zipf rank-frequency profiling
- Unkeyed boundary check (n = 40)

## Data

- **Corpus A:** Expert justifications from the CasiMedicos-Arg English test split
- **Corpus B:** Mistral (7B parameter, open-weight instruction-tuned model) run locally via Ollama

Dataset: [CasiMedicos-Arg](https://huggingface.co/datasets/HiTZ/casimedicos-arg) — CC-BY licence. SNOMED CT RF2 is not redistributed here.

## Repository Structure

```
corpus_a/          # Human justifications (one file per case)
corpus_b/          # Mistral rewrites (one file per case)
results/           # TF-IDF, keyword, hedging, and SNOMED outputs
antconc/           # Merged files for AntConc analysis
report/            # LaTeX source and compiled PDF
```

## Requirements

Python 3, NLTK, scikit-learn, scispaCy (`en_core_sci_sm`), Ollama (for regeneration)

## Author

Darragh Kerins — University of the Basque Country (UPV/EHU)
