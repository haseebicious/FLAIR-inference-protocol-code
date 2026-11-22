# Lexical Overfitting in Zero-Shot Retinal Diagnosis: A Case Study of FLAIR on MESSIDOR-2

**Authors:** Haseebullah Jumakhan\*, Ghazi Al-Naymat, Mohammed Azmi Al-Betar  
**Affiliation:** Artificial Intelligence Research Center, Ajman University, P.O. Box 346, Ajman, United Arab Emirates  
\*Corresponding author (`hjk608@nyu.edu`)

---

## Overview

This repository provides the complete reproducibility codebase, 300-prompt taxonomies, targeted frozen-inference controls, image-clustered statistical reanalysis, and visualization notebooks for our manuscript:

> **Lexical Overfitting in Zero-Shot Retinal Diagnosis: A Case Study of FLAIR on MESSIDOR-2**

To separate pre-trained cross-modal representations from downstream dataset-specific fine-tuning, we evaluate the frozen **FLAIR** retinal foundation model (`jusiro2/FLAIR`, native inverse temperature $\tau = 17.4355$) on the **$N = 1,744$ clinically gradable MESSIDOR-2** fundus photographs (4 ungradable entries dropped from the 1,748 Google Brain third-party adjudicated grades: `20060411_58550_0200_PP.png`, `IM002385.jpg`, `IM003718.jpg`, and `IM004176.jpg`; grade distribution: 1,017 `noDR` [58.31%], 270 `mildDR`, 347 `modDR`, 75 `sevDR`, 35 `prolDR`).

Key methodological components implemented here:

1. **Four-Family Semantic Perturbation Taxonomy (200 Prompts):** Ten deterministic syntactic realizations across all five ICDR grades (`noDR`, `mildDR`, `modDR`, `sevDR`, `prolDR`) for $T_{\text{label}}$, $T_{\text{expert}}$, $T_{\text{layman}}$, and $T_{\text{exclusion}}$ ($348{,}800$ image–prompt similarity evaluations; $17{,}440$ five-way predictions per family).
2. **Two-Pass Automated NIH UMLS Sanitization & Audit Telemetry:** Asynchronous NIH UMLS REST API auditing targeting Semantic Types `T046` (*Pathologic Function*) and `T047` (*Disease or Syndrome*) across $5{,}532$ unique $\ge 4$-character n-grams (UMLS release `2026AB`, zero API errors), with full logging of stripped CUIs and residual `T033` (*Finding*), `T184` (*Sign or Symptom*), and `T190` (*Anatomical Abnormality*) concepts.
3. **Three Targeted Frozen-Inference Controls (100 Additional Prompts):**
   - **Control A:** 24-configuration official FLAIR baseline verification (`dk=False` and `dk=True` from published `dictionary.py`) decomposing caption wrapping, class-mean ensembling, base-label inclusion, and batched `[PAD]` averaging.
   - **Control B:** 50 matched minimal-pair negation probes (`CB01`–`CB50`) spanning five severity tiers, two registers (25 Clinical, 25 Lay), and five logical negation operators.
   - **Control C:** 50 information-preserving clinical paraphrases retaining complete ICDR lesion, severity, and quadrant criteria without UMLS stripping.
4. **Cluster-Robust Statistics & Post-Hoc Recalibration:** Image-level cluster-robust paired bootstrap ($B = 10{,}000$ unique `image_id` clusters), Exchangeable Generalized Estimating Equations (GEE) across all $69{,}760$ predictions ($1{,}744$ clusters $\times$ 40 prompts, $\hat{\rho} = 0.1160$, sandwich-robust SEs), and 5-fold stratified image-level cross-validated post-hoc temperature scaling ($\tau^*$).

---

## Repository Structure

```text
FLAIR-inference-protocol-code/
├── prompts/
│   ├── prompts.json                      # Pass-1 initial 200-prompt taxonomy
│   ├── replacement_prompts.json          # 41 rewritten non-expert prompts (Pass 2)
│   ├── final_prompts.json                # Final evaluated 200-prompt taxonomy
│   ├── control_b_pairs.json              # 50 matched minimal-pair negation probes (CB01-CB50)
│   └── control_c_prompts.json            # 50 information-preserving ICDR clinical paraphrases
├── scripts/
│   ├── 1_umls_sanitization.py            # Standalone async NIH UMLS T046/T047 sanitization pipeline
│   ├── 2_feature_extraction.py           # Frozen FLAIR visual & text embedding extraction
│   ├── 3_inference_and_audit.py          # Core cosine-similarity inference & ECE evaluation
│   ├── 4_generate_figures.py             # Reliability diagram plotting script
│   └── experiments/
│       ├── control_a_baseline.py         # Gate 0 GPU verification + 24-row Control A ablation
│       ├── control_b_negation.py         # Control B matched minimal-pair negation ablation
│       ├── control_c_paraphrase_umls.py  # Control C paraphrase inference + live UMLS telemetry
│       ├── run_m1.py                     # Unified runner & consolidator for Controls A-C
│       ├── stats_reanalysis.py           # Image-clustered bootstrap, GEE, RDR benchmarks, 5-fold CV tau*
│       └── regenerate_audited_figures.py # Publication-grade reliability & UMAP figure regeneration
├── notebooks/
│   ├── Thesis_Graphs.ipynb               # High-resolution graphical abstract & summary plots
│   └── Thesis_Results_Visualization.ipynb# Exploratory & publication visualization suite
├── data/
│   └── images/placeholder                # Place MESSIDOR-2 images under data/IMAGES/ and messidor_data.csv under data/
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Verified Headline Results

All values below are reproduced deterministically on the $N = 1{,}744$ gradable MESSIDOR-2 subset using frozen `jusiro2/FLAIR` weights ($\tau = 17.4355$). Brackets report 95% confidence intervals from $B = 10{,}000$ image-level cluster-robust bootstrap resamples.

### 1. Primary Four-Taxonomy Evaluation ($17{,}440$ Alignments per Family)

| Prompt Family | Acc. (%) | ACA (%) | Macro-F1 | QWK | RDR AUROC | RDR Sens / Spec / F1 (%) | ECE$_{10}$ (%) | Brier | Avg. Conf. (%) | 5-Fold CV $\tau^*$ / OOF ECE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$T_{\text{label}}$** | **56.18** `[54.24, 58.17]` | **51.10** `[48.09, 54.09]` | **0.4075** `[0.3823, 0.4307]` | **0.6659** `[0.6358, 0.6929]` | **0.8878** `[0.8690, 0.9051]` | 67.09 / 94.25 / 73.21 | **4.45** `[3.17, 6.25]` | **0.5817** | 60.63 | 14.90 ± 0.17 / 3.56 |
| **$T_{\text{expert}}$** | 45.23 `[44.13, 46.32]` | 31.69 `[29.98, 33.49]` | 0.3213 `[0.3035, 0.3383]` | 0.2573 `[0.2281, 0.2853]` | 0.6770 `[0.6620, 0.6916]` | 27.77 / 90.47 / 35.92 | 32.57 `[31.47, 33.69]` | 0.8287 | 77.80 | 5.93 ± 0.02 / 3.85 |
| **$T_{\text{layman}}$** | 29.62 `[28.90, 30.36]` | 22.29 `[21.37, 23.30]` | 0.1878 `[0.1806, 0.1946]` | -0.0134 `[-0.0253, -0.0015]` | 0.4802 `[0.4748, 0.4858]` | 34.51 / 65.30 / 29.72 | 38.70 `[37.95, 39.46]` | 1.0281 | 68.27 | 2.69 ± 0.01 / 3.25 |
| **$T_{\text{exclusion}}$** | 26.90 `[26.28, 27.55]` | 22.09 `[21.04, 23.22]` | 0.1938 `[0.1884, 0.1993]` | -0.0252 `[-0.0375, -0.0128]` | 0.5401 `[0.5265, 0.5538]` | 51.95 / 53.22 / 36.62 | 36.15 `[35.47, 36.82]` | 0.9808 | 63.06 | 2.81 ± 0.03 / 8.40 |

- **Clustered Significance (Exchangeable GEE, $69{,}760$ predictions, $\hat{\rho} = 0.1160$):** $T_{\text{label}}$ vs. $T_{\text{expert}}$ achieves $\text{OR} = 1.5526$ (Wald $\chi^2 = 209.00$, $p = 2.27 \times 10^{-47}$); all six pairwise taxonomy contrasts remain highly significant ($p \le 4.58 \times 10^{-10}$).
- **Exclusionary False-Positive Failure:** Under $T_{\text{exclusion}}$, healthy `noDR` eyes exhibit a **24.03%** class-conditional false-positive rate for Proliferative DR ($2{,}444 / 10{,}170$).
- **Recalibration Trade-Off:** 5-fold cross-validated temperature scaling reduces ECE on $T_{\text{layman}}$ and $T_{\text{exclusion}}$ only by flattening posteriors toward near-uniform uncertainty (29.33% and 27.94% out-of-fold mean confidence) without restoring diagnostic accuracy.
- **Modality Gap Geometry:** Frozen visual vs. text embeddings exhibit Silhouette score **0.6393**, mean visual–text cosine distance **0.8801**, and $T_{\text{label}}$–$T_{\text{layman}}$ cross-family cosine similarity **0.3434**.

### 2. Targeted Frozen-Inference Controls (Controls A–C)

#### Control A: Official FLAIR Baseline Verification & $T_{\text{expert}}$ Factor Decomposition

| Configuration | Acc. (%) | ACA (%) | Macro-F1 | QWK | RDR AUROC | ECE$_{10}$ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Official `dk=False` (`A01`) | 45.76 | 53.85 | 0.4870 | 0.6115 | 0.8510 | 15.58 |
| Official `dk=True` unrenorm (`A04`) | 56.94 | 52.38 | 0.5625 | 0.6803 | 0.9169 | 2.58 |
| Official `dk=True` L2-renorm (`A05`) | 56.59 | 51.63 | 0.5495 | 0.6638 | 0.9225 | 4.21 |
| Custom $T_{\text{expert}}$ raw pooled (`A19`) | 45.23 | 31.69 | 0.3213 | 0.2573 | 0.6770 | 32.57 |
| Custom $T_{\text{expert}}$ raw ensembled (`A20`) | 60.72 | 38.29 | 0.3650 | 0.6148 | 0.8752 | 10.97 |
| Custom $T_{\text{expert}}$ wrapped pooled (`A21`) | 45.82 | 38.12 | 0.3263 | 0.3441 | 0.7237 | 25.82 |
| Custom $T_{\text{expert}}$ wrapped + ens. + base label + pad (`A24`) | **63.88** | 48.04 | 0.4575 | 0.6594 | 0.8888 | 9.04 |

#### Control B: 50 Matched Minimal-Pair Negation Probes (`CB01`–`CB50`)

| Protocol | $r_{\text{text}}$ / $r_{\text{score}}$ | $\Delta S_{\text{noDR}}$ | $\Delta S_{\text{lesion}}$ | $\Delta\Delta S$ | 2-Way Polarity Error (`noDR` / Lesion) | Neutral-Anchor $\Delta$FPR | 5-Way Swap $\Delta$FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Raw single-string | 0.9415 / 0.9432 | +0.0288 | +0.0032 | +0.0256 | 35.08% / 37.55% | +6.36 pp | +6.23 pp |
| Caption-wrapped | 0.9207 / 0.9171 | +0.0153 | -0.0255 | +0.0408 | 39.86% / 35.58% | +4.47 pp | +4.55 pp |

#### Control C: Information-Preserving Clinical Paraphrases vs. Primary Taxonomies (Raw / Wrapped)

| Taxonomy | Acc. (%) | ACA (%) | Macro-F1 | QWK | RDR AUROC | ECE$_{10}$ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$T_{\text{control\_c}}$ (ICDR paraphrases)** | 35.42 / 39.58 | 29.55 / 30.89 | 0.2912 / 0.3183 | 0.2551 / 0.2991 | **0.7762 / 0.7910** | 44.64 / 31.10 |
| $T_{\text{label}}$ (bare labels) | 56.18 / 59.58 | 51.10 / 43.07 | 0.4075 / 0.4579 | 0.6659 / 0.6326 | 0.8878 / 0.8798 | 4.45 / 7.47 |
| $T_{\text{expert}}$ (custom clinical) | 45.23 / 45.82 | 31.69 / 38.12 | 0.3213 / 0.3263 | 0.2573 / 0.3441 | 0.6770 / 0.7237 | 32.57 / 25.82 |
| $T_{\text{layman}}$ (UMLS-sanitized) | 29.62 / 37.67 | 22.29 / 25.85 | 0.1878 / 0.2389 | -0.0134 / 0.1140 | 0.4802 / 0.6291 | 38.70 / 18.46 |
| $T_{\text{exclusion}}$ (negated) | 26.90 / 30.51 | 22.09 / 24.48 | 0.1938 / 0.2097 | -0.0252 / 0.0655 | 0.5401 / 0.5475 | 36.15 / 32.38 |

#### UMLS Audit Telemetry (`2026AB`, $5{,}532$ Unique $\ge 4$-Char N-Grams)

- **Pass 1 (`prompts.json`):** Original `2025AB` sanitization rejected 41/100 non-expert prompts (18 $T_{\text{layman}}$, 23 $T_{\text{exclusion}}$); live re-audit under `2026AB` flagged 40/100 non-expert prompts across 44 prompt–term hits, 9 surface terms, and 8 CUIs (39 `T046` led by `C0019080` *Hemorrhage*; 5 `T047`).
- **Pass 2 (`replacement_prompts.json`):** 41/41 rewritten prompts passed (0% rejection across $1{,}125$ unique $\ge 4$-char n-grams).
- **Final Evaluated Set (`final_prompts.json`):** Live `2026AB` audit detected 1 residual `T046` hit (`leakage` $\rightarrow$ `C0015376` *Extravasation*) in $T_{\text{layman}}$ and 0 in $T_{\text{exclusion}}$, alongside residual non-forbidden concepts in 22/50 $T_{\text{layman}}$ prompts (14 `T033`, 10 `T184`, 0 `T190`) and 36/50 $T_{\text{exclusion}}$ prompts (30 `T033`, 12 `T184`, 0 `T190`).

---

## Reproducibility Quickstart

### 1. Environment Setup

```bash
git clone https://github.com/haseebicious/FLAIR-inference-protocol-code.git
cd FLAIR-inference-protocol-code

pip install -r requirements.txt
pip install git+https://github.com/jusiro/FLAIR.git
```

### 2. Configure Your NIH UMLS API Key (Required for Live UMLS Sanitization/Telemetry)

Register for a free UMLS Terminology Services (UTS) account under standard National Institutes of Health licensing at [https://uts.nlm.nih.gov/uts/](https://uts.nlm.nih.gov/uts/) and export your API key as an environment variable (never commit API keys to version control):

**Linux / macOS:**
```bash
export UMLS_API_KEY="YOUR_UMLS_API_KEY"
```

**Windows PowerShell:**
```powershell
$env:UMLS_API_KEY = "YOUR_UMLS_API_KEY"
```

*(Note: `scripts/1_umls_sanitization.py` also accepts `NIH_API_KEY`.)*

### 3. Dataset Preparation (MESSIDOR-2 + Google Brain Adjudicated Grades)

1. Download the publicly available **MESSIDOR-2** fundus photographs ([Decencière et al., 2014](https://www.adcis.net/en/third-party/messidor2/)) and extract them into `data/IMAGES/`.
2. Download the publicly released Google Brain third-party adjudicated five-point ICDR severity grading file (`messidor_data.csv`, 1,748 rows) and place it at `data/messidor_data.csv`.
3. Copy `prompts/final_prompts.json` to `data/final_prompts.json` if running the legacy `scripts/2_feature_extraction.py` directly (the `scripts/experiments/` runners read directly from `./prompts/`).

### 4. Step 1 — Frozen Visual & Text Feature Extraction

Run frozen `jusiro2/FLAIR` feature extraction to cache the $1{,}744$ L2-normalized visual embeddings, 200 primary text embeddings, and native logit scale ($\tau = 17.4355$) into `results/`:

```bash
cd scripts
python 2_feature_extraction.py
python 3_inference_and_audit.py
cd ..
```

### 5. Step 2 — Run Controls A, B, and C (+ Live NIH UMLS Audit Telemetry)

```bash
python scripts/experiments/run_m1.py \
  --data-dir ./data \
  --cache-dir ./results \
  --prompts-dir ./prompts \
  --artifacts-dir ./artifacts
```

You can also invoke each control independently:
```bash
python scripts/experiments/control_a_baseline.py --data-dir ./data --cache-dir ./results --prompts-dir ./prompts --artifacts-dir ./artifacts
python scripts/experiments/control_b_negation.py --data-dir ./data --cache-dir ./results --prompts-dir ./prompts --artifacts-dir ./artifacts
python scripts/experiments/control_c_paraphrase_umls.py --data-dir ./data --cache-dir ./results --prompts-dir ./prompts --artifacts-dir ./artifacts
```

### 6. Step 3 — Image-Clustered Statistical Reanalysis & 5-Fold CV Temperature Scaling

Produces `artifacts/m2_verified_metrics.json` and the consolidated `artifacts/verified_metrics.json` ($B = 10{,}000$ cluster-robust bootstrap, Exchangeable GEE, binary Referable DR benchmarks, binning-sensitivity ECE, 5-fold CV $\tau^*$, and latent-space geometry):

```bash
python scripts/experiments/stats_reanalysis.py \
  --data-dir ./data \
  --cache-dir ./results \
  --artifacts-dir ./artifacts
```

### 7. Step 4 — Regenerate Publication Figures & Graphical Abstract

```bash
python scripts/experiments/regenerate_audited_figures.py \
  --cache-dir ./results \
  --output-dir ./figures
```

For full exploratory figures and the 300-DPI graphical abstract, run the notebooks in `notebooks/Thesis_Graphs.ipynb` and `notebooks/Thesis_Results_Visualization.ipynb`. Deterministic 2D UMAP projections use `n_neighbors=15`, `min_dist=0.1`, `metric="cosine"`, and `random_state=42`.

---

## Citation

A preprint version of this work is available on SSRN, and the revised manuscript has been submitted to *Engineering Applications of Artificial Intelligence* (Elsevier). If you use this codebase, prompt taxonomies, or evaluation protocol in your research, please cite:

```bibtex
@article{jumakhan2026lexical,
  title   = {Lexical Overfitting in Zero-Shot Retinal Diagnosis: A Case Study of FLAIR on MESSIDOR-2},
  author  = {Jumakhan, Haseebullah and Al-Naymat, Ghazi and Al-Betar, Mohammed Azmi},
  journal = {Engineering Applications of Artificial Intelligence},
  year    = {2026},
  note    = {Manuscript submitted for publication; preprint available on SSRN},
  url     = {https://github.com/haseebicious/FLAIR-inference-protocol-code}
}
```

---

## License

This project is released under the **MIT License** — see the [LICENSE](LICENSE) file for details.
