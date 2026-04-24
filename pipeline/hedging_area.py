#!/usr/bin/env python3
# Rough specialty label from keywords in the case stem, then hedging counts and bar charts by group.

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset

from .hedging import hedge_hits_and_tokens
from .paths import (
    CORPUS_A,
    CORPUS_B,
    HEDGE_CASE_TABLE,
    HEDGE_GROUP_TABLE,
    HEDGE_PLOT_BY_SPECIALTY,
    HEDGE_PLOT_OVERALL,
    HF_CACHE,
    ROOT,
    ensure_results,
)
from .prep_corpora import find_correct_answer_index, parse_tokenized_text, sentences_to_string

LABEL_GENERAL = "General medicine (no clear lexical match)"
LABEL_MIXED = "Multi-specialty stem (tied lexical scores)"

AREA_LEXICON: dict[str, frozenset[str]] = {
    "Cardiovascular medicine": frozenset(
        """
        heart cardiac myocardial hypertension angina coronary pericardial endocarditis murmur
        arrhythmia atrial fibrillation ventricular myocarditis cardiomyopathy ischaemic ischemic
        cardiogenic shock tamponade aortic mitral tricuspid endarterectomy stent bypass graft
        troponin pacemaker pacing percutaneous echocardiogram cardiomegaly endothelin
        """.split()
    ),
    "Respiratory medicine": frozenset(
        """
        lung pulmonary asthma copd wheeze cough pleural pneumothorax pneumonia hypoxaemia hypoxemia
        intubation ventilation bronchiectasis spirometry emphysema bronchiolitis respiration dyspnoea
        dyspnea obstructive restrictive pleurisy haemothorax hemothorax parenchyma sputum
        """.split()
    ),
    "Gastroenterology & hepatology": frozenset(
        """
        liver hepatitis cirrhosis jaundice abdominal gastric duodenal bowel colitis pancreatitis
        nausea vomiting diarrhoea diarrhea cholecystitis gallstone oesophageal esophageal portal
        ascites hepatic splenic peptic ulcerative crohn celiac coeliac mesenteric varices
        haematemesis hematemesis melaena melena cholangitis steatorrhoea steatorrhea
        """.split()
    ),
    "Endocrinology & metabolic medicine": frozenset(
        """
        diabetes insulin glucose hypoglycaemia hypoglycemia thyroid cortisol cushing addison pituitary
        parathyroid hyperthyroid hypothyroid thyrotoxicosis ketosis acidosis metabolic syndrome
        hyperparathyroidism acromegaly prolactin hba1c glycaemic glycemic lipids thyroxine
        """.split()
    ),
    "Nephrology & urology": frozenset(
        """
        renal kidney dialysis creatinine glomerulonephritis nephrotic proteinuria haematuria hematuria
        ureter bladder prostate cystitis incontinence nephrectomy uraemia uremia
        hydronephrosis glomerular tubular interstitial urethritis nephrolithiasis lithotripsy
        """.split()
    ),
    "Infectious diseases": frozenset(
        """
        fever sepsis antibiotic bacterial viral hiv tuberculosis malaria infection abscess cellulitis
        meningococcal streptococcal staphylococcal parasitic fungal influenza covid pyrexia
        bacteraemia bacteremia viraemia viremia quarantine antiviral antiretroviral
        """.split()
    ),
    "Haematology & coagulation": frozenset(
        """
        anemia anaemia hemophilia haemophilia platelet leukaemia leukemia transfusion thrombocytopenia
        coagulation bleeding haemostasis hemostasis sickle thalassaemia thalassemia haemolytic hemolytic
        lymphadenopathy marrow aplastic polycythaemia polycythemia anticoagulant fibrinogen
        haemoglobin hemoglobin neutropenia pancytopenia myeloma lymphoma hodgkin
        """.split()
    ),
    "Neurology": frozenset(
        """
        seizure epilepsy headache meningitis stroke encephalopathy neuropathy dementia parkinson
        sclerosis myelopathy cranial subdural intracerebral consciousness syncope vertigo tremor
        paraplegia quadriplegia delirium subarachnoid intracranial hemiparesis hemianopia
        """.split()
    ),
    "Rheumatology & musculoskeletal medicine": frozenset(
        """
        arthritis lupus sle rheumatoid gout scleroderma vasculitis joint erythema myositis dermatomyositis
        polymyalgia bursitis tendinitis ankylosing psoriatic fibromyalgia osteoarthritis
        sacroiliitis tenosynovitis polymyositis
        """.split()
    ),
    "Clinical oncology": frozenset(
        """
        cancer tumour tumor malignant metastasis chemotherapy radiotherapy radiation carcinoma sarcoma neoplasm
        biopsy staging immunotherapy oncologic malignancy cytotoxic palliative oncology
        """.split()
    ),
    "Psychiatry & psychological medicine": frozenset(
        """
        depression anxiety suicide psychosis bipolar schizophrenia overdose self harm self-harm
        antidepressant antipsychotic benzodiazepine hallucination paranoia cognitive psychotherapy
        """.split()
    ),
    "Obstetrics & gynaecology": frozenset(
        """
        pregnancy pregnant fetal caesarean cesarean uterine ovarian preeclampsia labour labor
        gestation trimester amniotic placenta ectopic menorrhagia endometriosis postpartum
        gynaecology gynecology miscarriage antenatal postnatal eclampsia
        """.split()
    ),
    "Dermatology": frozenset(
        """
        rash skin eczema psoriasis dermatitis melanoma urticaria pruritus vesicular bullous
        cutaneous excoriation keratosis pemphigus epidermolysis
        """.split()
    ),
    "Orthopaedics & trauma": frozenset(
        """
        fracture bone femur dislocation tendon ligament osteomyelitis orthopaedic orthopedic trauma
        sprain meniscus cruciate patella humerus tibia fibula pelvis arthroplasty
        """.split()
    ),
}


def stem_tokens(stem: str) -> set[str]:
    return set(re.findall(r"\b[a-z]{3,}\b", stem.lower()))


def classify_area(stem: str) -> str:
    words = stem_tokens(stem)
    if not words:
        return LABEL_GENERAL
    scores = {area: len(words & lex) for area, lex in AREA_LEXICON.items()}
    best = max(scores.values())
    if best < 2:
        return LABEL_GENERAL
    leaders = sorted(a for a, s in scores.items() if s == best)
    if len(leaders) > 1:
        return LABEL_MIXED
    return leaders[0]


def case_stem_from_row(raw_text: object) -> str | None:
    try:
        sents = parse_tokenized_text(raw_text)
        idx = find_correct_answer_index(sents)
        return sentences_to_string(sents[:idx])
    except (ValueError, SyntaxError):
        return None


def case_number_from_path(path: Path) -> int:
    m = re.match(r"case_(\d+)\.txt$", path.name)
    if not m:
        raise ValueError(f"unexpected filename: {path.name}")
    return int(m.group(1))


def collect_rows() -> pd.DataFrame:
    os.environ.setdefault("HF_HOME", str(HF_CACHE))
    ds = load_dataset("HiTZ/casimedicos-arg", "en", split="test")
    n = len(ds)
    records: list[dict[str, object]] = []
    for path_a in sorted(CORPUS_A.glob("case_*.txt")):
        path_b = CORPUS_B / path_a.name
        if not path_b.is_file():
            continue
        text_a = path_a.read_text(encoding="utf-8").strip()
        text_b = path_b.read_text(encoding="utf-8").strip()
        if not text_a or not text_b:
            continue
        case_num = case_number_from_path(path_a)
        idx = case_num - 1
        if idx < 0 or idx >= n:
            print(f"warn: case index {case_num} out of dataset range", file=sys.stderr)
            continue
        row = ds[idx]
        rid = row.get("id", case_num)
        stem = case_stem_from_row(row["text"])
        if stem is None:
            area = LABEL_GENERAL
        else:
            area = classify_area(stem)
        ha, ta = hedge_hits_and_tokens(text_a)
        hb, tb = hedge_hits_and_tokens(text_b)
        ra = (1000.0 * ha / ta) if ta else 0.0
        rb = (1000.0 * hb / tb) if tb else 0.0
        records.append(
            {
                "case_file": path_a.name,
                "case_num": case_num,
                "dataset_id": rid,
                "medical_area": area,
                "A_hedge_hits": ha,
                "A_tokens": ta,
                "A_per_1000": round(ra, 4),
                "B_hedge_hits": hb,
                "B_tokens": tb,
                "B_per_1000": round(rb, 4),
                "delta_B_minus_A_per_1000": round(rb - ra, 4),
            }
        )
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    df = df.sort_values("case_num").reset_index(drop=True)
    return df


def aggregate_by_area(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for area, g in df.groupby("medical_area", sort=False):
        rows.append(
            {
                "medical_area": area,
                "n_cases": len(g),
                "A_mean_per_1000": round(g["A_per_1000"].mean(), 4),
                "B_mean_per_1000": round(g["B_per_1000"].mean(), 4),
                "mean_delta_B_minus_A": round(g["delta_B_minus_A_per_1000"].mean(), 4),
                "A_tokens_sum": int(g["A_tokens"].sum()),
                "B_tokens_sum": int(g["B_tokens"].sum()),
            }
        )
    out = pd.DataFrame(rows).sort_values("n_cases", ascending=False).reset_index(drop=True)
    return out


def plot_overall(df: pd.DataFrame) -> None:
    a_mean = df["A_per_1000"].mean()
    b_mean = df["B_per_1000"].mean()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Human (A)", "Model (B)"], [a_mean, b_mean], color=["#2c5282", "#c05621"])
    ax.set_ylabel("Hedging markers per 1,000 tokens (mean over cases)")
    ax.set_title("Overall hedging density")
    fig.tight_layout()
    fig.savefig(HEDGE_PLOT_OVERALL, dpi=150)
    plt.close(fig)


def plot_by_area(area_df: pd.DataFrame, *, min_cases: int) -> None:
    sub = area_df[area_df["n_cases"] >= min_cases].copy()
    if sub.empty:
        print(f"No areas with ≥{min_cases} cases; skipping by-area plot.", file=sys.stderr)
        return
    sub = sub.sort_values("mean_delta_B_minus_A", ascending=False)
    labels = sub["medical_area"].tolist()
    x = range(len(labels))
    w = 0.38
    fig_w = max(10.0, len(labels) * 0.62)
    fig, ax = plt.subplots(figsize=(fig_w, 5.2))
    ax.bar([i - w / 2 for i in x], sub["A_mean_per_1000"], width=w, label="Human (A)", color="#2c5282")
    ax.bar([i + w / 2 for i in x], sub["B_mean_per_1000"], width=w, label="Model (B)", color="#c05621")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=42, ha="right", fontsize=8)
    ax.set_ylabel("Mean hedging per 1,000 tokens")
    ax.set_title(
        f"Hedging by medical specialty group (British labels; ≥{min_cases} cases per bar)\n"
        "Lexicon tagging on case stem text"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(HEDGE_PLOT_BY_SPECIALTY, dpi=150)
    plt.close(fig)


def main(*, min_area_cases: int = 3) -> None:
    ensure_results()
    df = collect_rows()
    if df.empty:
        raise SystemExit("No paired non-empty corpus files found.")
    df.to_csv(HEDGE_CASE_TABLE, index=False)
    area_df = aggregate_by_area(df)
    area_df.to_csv(HEDGE_GROUP_TABLE, index=False)
    plot_overall(df)
    plot_by_area(area_df, min_cases=min_area_cases)
    print(f"wrote {HEDGE_CASE_TABLE.relative_to(ROOT)} ({len(df)} cases)")
    print(f"wrote {HEDGE_GROUP_TABLE.relative_to(ROOT)} ({len(area_df)} groups)")
    print(f"wrote {HEDGE_PLOT_OVERALL.relative_to(ROOT)} and {HEDGE_PLOT_BY_SPECIALTY.relative_to(ROOT)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Medical-area labels + hedging by area + plots.")
    p.add_argument(
        "--min-area-cases",
        type=int,
        default=3,
        help="Medical areas with fewer cases are omitted from the by-area bar chart (still in CSV).",
    )
    main(min_area_cases=p.parse_args().min_area_cases)
