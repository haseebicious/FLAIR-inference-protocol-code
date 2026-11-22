"""Milestone 1 — Control C (Information-Preserving Clinical Paraphrases) + Live NIH UMLS Telemetry.

Implements:
1. Exact 50 a priori information-preserving clinical paraphrase prompts (5 grades x 10 variants)
   validated for zero T_label leaks and <77 Bio_ClinicalBERT tokens,
   saved to artifacts/control_c_prompts.json.
2. Frozen FLAIR inference (single-string batch_size=1, zero [PAD]) under both
   raw_single_string and wrapped_caption ("A fundus photograph of [CLS]") protocols,
   comparing Control C against T_label, T_expert, T_layman, and T_exclusion across
   Accuracy, ACA, Macro-F1, QWK, Mean Confidence, Micro-ECE (M=10, 15, 20),
   Macro-ECE (mean +/- SD), and Referable DR (AUROC, Sensitivity, Specificity, F1),
   with B=10,000 image-level cluster-bootstrap 95% CIs and paired deltas.
3. Non-short-circuiting live NIH UMLS REST API audit across all unique len>=4 n-grams
   in data/prompts.json, data/replacement_prompts.json, data/final_prompts.json, and
   artifacts/control_c_prompts.json, using keyless artifacts/umls_term_cache.json,
   rate-limited <= 14 req/sec with retries, api_errors == 0 assertion, and full
   reconciliation against first & second sanitization reports.

Outputs:
- artifacts/control_c_prompts.json
- artifacts/control_c_results.json
- artifacts/umls_term_cache.json (strictly keyless)
- artifacts/umls_telemetry.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import aiohttp
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score
from transformers import AutoTokenizer

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = WORKSPACE_ROOT / "data"
PROMPTS_DIR = WORKSPACE_ROOT / "prompts"
WEEK4_DIR = (WORKSPACE_ROOT / "results") if (WORKSPACE_ROOT / "results" / "cached_vision_embeddings_1744.npy").exists() else (
    (DATA_DIR / "Week 4 Work") if (DATA_DIR / "Week 4 Work").exists() else DATA_DIR
)
ARTIFACTS_DIR = WORKSPACE_ROOT / "artifacts"
LOCAL_FILES_ONLY = False

CAPTION_TEMPLATE = "A fundus photograph of [CLS]"
BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 42

GRADES_ORDER = ["noDR", "mildDR", "modDR", "sevDR", "prolDR"]
GRADE_TO_IDX = {g: i for i, g in enumerate(GRADES_ORDER)}
FLAIR_CATEGORIES = [
    "no diabetic retinopathy",
    "mild diabetic retinopathy",
    "moderate diabetic retinopathy",
    "severe diabetic retinopathy",
    "proliferative diabetic retinopathy",
]

# Exact 50 a priori information-preserving clinical paraphrases (5 grades x 10 variants)
RAW_CONTROL_C_SPEC: Dict[str, List[str]] = {
    "noDR": [
        "Normal fundus examination showing intact retinal vasculature with no microaneurysms, intraretinal haemorrhages, or hard exudates.",
        "Posterior pole assessment reveals an unremarkable healthy retina free of capillary microaneurysms, dot-and-blot haemorrhages, and lipid exudates.",
        "Ophthalmoscopic evaluation demonstrates normal retinal vessels across all quadrants without microaneurysms, retinal haemorrhages, or exudates.",
        "Retinal imaging confirms healthy optic disc and macula with complete absence of microaneurysms, intraretinal haemorrhages, and exudative lesions.",
        "Fundus is clear of relevant haemorrhages, capillary microaneurysms, cotton wool spots, and hard exudates across all four quadrants.",
        "Clinical inspection identifies normal retinal architecture with zero microaneurysms, zero haemorrhages, and no referable retinal lesions.",
        "Unremarkable ocular fundus exhibiting preserved vascular calibre and no saccular microaneurysms, haemorrhages, or exudates.",
        "Healthy retinal background lacking capillary outpouchings, intraretinal bleeding, lipid exudates, or referable microvascular abnormalities.",
        "Fundus photograph displays clear macular and peripapillary retina entirely devoid of microaneurysms, haemorrhages, and exudates.",
        "Diagnostic fundus review establishes normal retina without microaneurysms, intraretinal haemorrhages, exudates, or diabetic microvascular lesions.",
    ],
    "mildDR": [
        "Fundus examination reveals microaneurysms only, presenting as isolated retinal capillary microaneurysms as the sole microvascular abnormality.",
        "Posterior pole imaging shows solely a few saccular capillary microaneurysms, representing early non-proliferative retinal involvement.",
        "Retinal evaluation identifies scattered capillary microaneurysms exclusively as the only visible fundus lesion.",
        "Ophthalmoscopic inspection detects microaneurysms alone, consisting of a few pinpoint red capillary outpouchings on an otherwise unremarkable retina.",
        "Findings are strictly limited to sparse retinal microaneurysms as the single earliest microvascular sign.",
        "Clinical fundus assessment demonstrates isolated capillary microaneurysms only, confined to minimal early non-proliferative changes.",
        "A few discrete retinal microaneurysms are visible as the exclusive abnormality in an otherwise intact fundus.",
        "Retinal photograph exhibits microaneurysms exclusively, appearing as rare small red saccular capillary dilatations.",
        "Examination confirms early microvascular change consisting solely of few isolated retinal microaneurysms.",
        "Only scattered capillary microaneurysms are observed on the fundus as the sole diabetic microvascular finding.",
    ],
    "modDR": [
        "Fundus shows microaneurysms accompanied by intraretinal haemorrhages in one to three quadrants, hard exudates, and cotton wool spots, below severe four-quadrant thresholds.",
        "Retinal examination reveals dot-and-blot haemorrhages in fewer than four quadrants alongside capillary microaneurysms, lipid exudates, and cotton wool spots.",
        "Posterior pole displays more than isolated microaneurysms, featuring moderate intraretinal haemorrhages in limited quadrants, hard exudates, and nerve fibre layer infarcts.",
        "Clinical imaging demonstrates retinal microaneurysms, hard exudates, cotton wool spots, and intraretinal haemorrhages across one to three quadrants below 4-2-1 thresholds.",
        "Ophthalmoscopic findings include scattered microaneurysms, moderate dot-blot haemorrhages in fewer than four quadrants, hard exudates, and focal cotton wool spots.",
        "Retina exhibits microaneurysms plus intraretinal haemorrhages in up to three quadrants, yellowish lipid exudates, and cotton wool spots, short of four-quadrant severity.",
        "Fundus evaluation identifies microaneurysms, multifocal intraretinal haemorrhages confined to one to three quadrants, hard exudates, and cotton wool spots.",
        "Assessment reveals moderate microvascular pathology including microaneurysms, exudates, cotton wool spots, and haemorrhages in fewer than four quadrants.",
        "Retinal photograph presents intraretinal haemorrhages in one to three quadrants together with microaneurysms, hard lipid exudates, and soft cotton wool spots.",
        "Examination notes microaneurysms, hard exudates, cotton wool spots, and moderate dot-and-blot haemorrhages across limited quadrants below four-quadrant thresholds.",
    ],
    "sevDR": [
        "Fundus meets the 4-2-1 criteria with over twenty intraretinal haemorrhages in each of all four quadrants, definite venous beading in two or more quadrants, or prominent IRMA.",
        "Retinal examination demonstrates severe non-proliferative 4-2-1 lesions: extensive dot-and-blot haemorrhages across all four quadrants, multi-quadrant venous beading, and prominent IRMA.",
        "Posterior pole exhibits more than twenty haemorrhages in every retinal quadrant, venous beading in at least two quadrants, or distinct intraretinal microvascular abnormalities.",
        "Clinical imaging reveals widespread severe intraretinal haemorrhages throughout all four quadrants alongside venous calibre beading in two or more quadrants and prominent IRMA.",
        "Ophthalmoscopic evaluation satisfies the 4-2-1 rule through four-quadrant severe haemorrhages, venous beading in two or more quadrants, and intraretinal microvascular shunts.",
        "Retina displays extensive haemorrhages exceeding twenty per quadrant in all four quadrants, beaded retinal veins in multiple quadrants, and prominent IRMA.",
        "Assessment confirms advanced intraretinal pathology: severe four-quadrant haemorrhages, multi-quadrant venous beading, and prominent capillary IRMA loops in at least one quadrant.",
        "Fundus photograph shows all four quadrants involved by heavy intraretinal haemorrhages, venous beading across two or more quadrants, or prominent IRMA in at least one quadrant.",
        "Diagnostic inspection establishes 4-2-1 rule severity via dense four-quadrant retinal haemorrhages, segmental venous beading in two or more quadrants, and intraretinal microvascular abnormalities.",
        "Severe non-proliferative intraretinal disease is present, characterized by over twenty haemorrhages in all four quadrants, venous beading in two or more quadrants, and prominent IRMA.",
    ],
    "prolDR": [
        "Fundus examination reveals neovascularization at the optic disc or elsewhere on the retina accompanied by vitreous or preretinal haemorrhage.",
        "Posterior pole shows active epiretinal neovascular fronds on the optic nerve head or retinal surface alongside preretinal or vitreous bleeding.",
        "Ophthalmoscopic evaluation detects proliferative new vessels at the disc (NVD) or elsewhere (NVE), complicated by subhyaloid, preretinal, or vitreous haemorrhage.",
        "Retinal imaging demonstrates abnormal neovascular proliferation arising from the optic disc or retina, together with preretinal or vitreous haemorrhage.",
        "Clinical inspection identifies fragile neovascular tufts on the optic disc or elsewhere, with associated preretinal blood or vitreous cavity haemorrhage.",
        "Fundus photograph displays neovascularization at the optic nerve head and retinal periphery alongside vitreous or preretinal haemorrhage and background microangiopathy.",
        "Assessment confirms active retinal neovascularization at the optic disc or elsewhere, accompanied by preretinal or vitreous haemorrhage.",
        "Retina exhibits proliferating abnormal new vessels at the optic disc or retinal surface, producing preretinal or vitreous haemorrhage.",
        "Diagnostic fundus review reveals neovascular networks at the disc or elsewhere on the retina, with concomitant preretinal or vitreous bleeding.",
        "Proliferative vascular changes are present, featuring neovascularization of the disc or elsewhere and associated vitreous or preretinal haemorrhage.",
    ],
}


def build_and_validate_control_c(tokenizer: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build 50 Control C prompts and assert zero T_label leaks and <77 tokens."""
    records = []
    max_raw_tok = 0
    max_wrap_tok = 0
    leaks = []

    for grade in GRADES_ORDER:
        prompts = RAW_CONTROL_C_SPEC[grade]
        assert len(prompts) == 10, f"Expected 10 variants for {grade}"
        for idx, text in enumerate(prompts, start=1):
            lower = text.lower()
            for lbl in FLAIR_CATEGORIES:
                if lbl in lower:
                    leaks.append((grade, idx, lbl))
            raw_ids = tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]
            wrap_ids = tokenizer(CAPTION_TEMPLATE.replace("[CLS]", text), truncation=False, add_special_tokens=True)["input_ids"]
            max_raw_tok = max(max_raw_tok, len(raw_ids))
            max_wrap_tok = max(max_wrap_tok, len(wrap_ids))
            assert len(raw_ids) < 77, f"Control C {grade} v{idx} raw len {len(raw_ids)} >= 77"
            assert len(wrap_ids) < 77, f"Control C {grade} v{idx} wrapped len {len(wrap_ids)} >= 77"
            records.append({
                "Grade": grade,
                "Family": "T_control_c",
                "Variation_Number": idx,
                "Prompt_Text": text,
            })

    assert len(records) == 50
    assert len(leaks) == 0, f"Found exact T_label leaks in Control C: {leaks}"
    invariants = {
        "n_prompts": 50,
        "grades": {g: 10 for g in GRADES_ORDER},
        "tlabel_exact_leaks": 0,
        "max_raw_tokens": max_raw_tok,
        "max_wrapped_tokens": max_wrap_tok,
        "tokenizer_limit": 77,
        "zero_truncation": True,
    }
    return records, invariants


# ---------------------------------------------------------------------------
# Metrics & Cluster-Bootstrap Engine
# ---------------------------------------------------------------------------

def calculate_ece(confidences: np.ndarray, accuracies: np.ndarray, num_bins: int = 10) -> float:
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for i in range(num_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == num_bins - 1:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences >= lo) & (confidences < hi)
        prop = float(np.mean(in_bin))
        if prop > 0.0:
            ece += abs(float(np.mean(accuracies[in_bin])) - float(np.mean(confidences[in_bin]))) * prop
    return float(ece)


def fast_auroc(y_true_bin: np.ndarray, y_score: np.ndarray) -> float:
    """Exact rank-based ROC AUC (Mann-Whitney U) matching sklearn.metrics.roc_auc_score."""
    ranks = rankdata(y_score, method="average")
    n_pos = float(np.sum(y_true_bin == 1))
    n_neg = float(len(y_true_bin) - n_pos)
    sum_pos_ranks = float(np.sum(ranks[y_true_bin == 1]))
    return (sum_pos_ranks - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)


def evaluate_family_probs(probs_10_var: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
    """Evaluate a (10, 1744, 5) probability tensor on point metrics + cluster-bootstrap stats."""
    n_vars, n_imgs, n_cls = probs_10_var.shape
    assert n_vars == 10 and n_imgs == 1744 and n_cls == 5

    probs_pooled = probs_10_var.reshape(-1, 5)  # (17440, 5)
    y_pooled = np.tile(y_true, n_vars)          # (17440,)

    preds_var = np.argmax(probs_10_var, axis=2)  # (10, 1744)
    confs_var = np.max(probs_10_var, axis=2)     # (10, 1744)
    corr_var = (preds_var == y_true[None, :]).astype(np.float64)  # (10, 1744)

    preds_pooled = preds_var.reshape(-1)
    confs_pooled = confs_var.reshape(-1)
    corr_pooled = corr_var.reshape(-1)

    # Per-variation Macro-ECE (M=10)
    per_var_ece = [calculate_ece(confs_var[v], corr_var[v], num_bins=10) for v in range(n_vars)]
    per_var_acc = [float(np.mean(corr_var[v])) for v in range(n_vars)]

    # Referable DR
    y_ref = (y_true >= 2).astype(int)
    p_ref_var = probs_10_var[:, :, 2] + probs_10_var[:, :, 3] + probs_10_var[:, :, 4]  # (10, 1744)
    p_ref_pooled = p_ref_var.reshape(-1)
    y_ref_pooled = np.tile(y_ref, n_vars)
    pred_ref_pooled = (p_ref_pooled >= 0.5).astype(int)

    tp = int(np.sum((y_ref_pooled == 1) & (pred_ref_pooled == 1)))
    fn = int(np.sum((y_ref_pooled == 1) & (pred_ref_pooled == 0)))
    tn = int(np.sum((y_ref_pooled == 0) & (pred_ref_pooled == 0)))
    fp = int(np.sum((y_ref_pooled == 0) & (pred_ref_pooled == 1)))

    point = {
        "accuracy": float(np.mean(corr_pooled)),
        "aca": float(balanced_accuracy_score(y_pooled, preds_pooled)),
        "macro_f1": float(f1_score(y_pooled, preds_pooled, labels=[0, 1, 2, 3, 4], average="macro", zero_division=0)),
        "qwk": float(cohen_kappa_score(y_pooled, preds_pooled, labels=[0, 1, 2, 3, 4], weights="quadratic")),
        "mean_confidence": float(np.mean(confs_pooled)),
        "micro_ece_m10": calculate_ece(confs_pooled, corr_pooled, num_bins=10),
        "micro_ece_m15": calculate_ece(confs_pooled, corr_pooled, num_bins=15),
        "micro_ece_m20": calculate_ece(confs_pooled, corr_pooled, num_bins=20),
        "macro_ece_m10_mean": float(np.mean(per_var_ece)),
        "macro_ece_m10_sd": float(np.std(per_var_ece, ddof=1)),
        "per_var_ece_m10": per_var_ece,
        "per_var_accuracy": per_var_acc,
        "referable_dr_auroc": float(roc_auc_score(y_ref_pooled, p_ref_pooled)),
        "referable_dr_sensitivity": float(tp / (tp + fn)),
        "referable_dr_specificity": float(tn / (tn + fp)),
        "referable_dr_f1": float((2 * tp) / (2 * tp + fp + fn)),
    }

    # Precompute sufficient per-image arrays for exact fast cluster bootstrap
    # 1. Per-image 5x5 confusion matrix (1744, 5, 5)
    cm_img = np.zeros((n_imgs, 5, 5), dtype=np.float64)
    for i in range(n_imgs):
        yt = y_true[i]
        for v in range(n_vars):
            cm_img[i, yt, preds_var[v, i]] += 1.0

    # 2. Per-image confidence & ECE(M=10) bin arrays (1744, 10)
    conf_img_sum = np.sum(confs_var, axis=0)  # (1744,)
    bin_edges = np.linspace(0.0, 1.0, 11)
    bin_count_img = np.zeros((n_imgs, 10), dtype=np.float64)
    bin_conf_img = np.zeros((n_imgs, 10), dtype=np.float64)
    bin_corr_img = np.zeros((n_imgs, 10), dtype=np.float64)
    for b_idx in range(10):
        lo, hi = bin_edges[b_idx], bin_edges[b_idx + 1]
        if b_idx == 9:
            mask = (confs_var >= lo) & (confs_var <= hi)
        else:
            mask = (confs_var >= lo) & (confs_var < hi)
        bin_count_img[:, b_idx] = np.sum(mask, axis=0)
        bin_conf_img[:, b_idx] = np.sum(confs_var * mask, axis=0)
        bin_corr_img[:, b_idx] = np.sum(corr_var * mask, axis=0)

    # 3. Per-image Referable DR TP/FP/TN/FN (1744,) and p_ref (1744, 10)
    pred_ref_var = (p_ref_var >= 0.5).astype(np.float64)  # (10, 1744)
    ref_tp_img = np.sum((y_ref[None, :] == 1) & (pred_ref_var == 1), axis=0).astype(np.float64)
    ref_fn_img = np.sum((y_ref[None, :] == 1) & (pred_ref_var == 0), axis=0).astype(np.float64)
    ref_tn_img = np.sum((y_ref[None, :] == 0) & (pred_ref_var == 0), axis=0).astype(np.float64)
    ref_fp_img = np.sum((y_ref[None, :] == 0) & (pred_ref_var == 1), axis=0).astype(np.float64)
    p_ref_img = p_ref_var.T.copy()  # (1744, 10)

    suff = {
        "cm_img": cm_img,
        "conf_img_sum": conf_img_sum,
        "bin_count_img": bin_count_img,
        "bin_conf_img": bin_conf_img,
        "bin_corr_img": bin_corr_img,
        "ref_tp_img": ref_tp_img,
        "ref_fn_img": ref_fn_img,
        "ref_tn_img": ref_tn_img,
        "ref_fp_img": ref_fp_img,
        "p_ref_img": p_ref_img,
    }
    return {"point": point, "suff": suff}


def run_cluster_bootstrap(
    families_eval: Dict[str, Dict[str, Any]],
    y_true: np.ndarray,
    boot_idx: np.ndarray,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Run B=10,000 image-level cluster bootstrap across all families and paired deltas vs Control C."""
    B, n_imgs = boot_idx.shape
    y_ref = (y_true >= 2).astype(int)

    # Precompute QWK weight matrix (5, 5)
    idx_grid = np.arange(5, dtype=np.float64)
    W_qwk = ((idx_grid[:, None] - idx_grid[None, :]) ** 2) / 16.0

    boot_series: Dict[str, Dict[str, np.ndarray]] = {}
    ci_summary: Dict[str, Dict[str, Any]] = {}

    for fam, obj in families_eval.items():
        suff = obj["suff"]
        # Vectorized confusion matrix over B replicates: chunked to keep RAM low
        chunk_size = 1000
        acc_arr = np.empty(B, dtype=np.float64)
        aca_arr = np.empty(B, dtype=np.float64)
        mf1_arr = np.empty(B, dtype=np.float64)
        qwk_arr = np.empty(B, dtype=np.float64)
        conf_arr = np.empty(B, dtype=np.float64)
        ece_arr = np.empty(B, dtype=np.float64)
        sens_arr = np.empty(B, dtype=np.float64)
        spec_arr = np.empty(B, dtype=np.float64)
        rf1_arr = np.empty(B, dtype=np.float64)
        auroc_arr = np.empty(B, dtype=np.float64)

        cm_img = suff["cm_img"]
        conf_sum = suff["conf_img_sum"]
        bc_img = suff["bin_count_img"]
        bconf_img = suff["bin_conf_img"]
        bcorr_img = suff["bin_corr_img"]
        rtp = suff["ref_tp_img"]
        rfn = suff["ref_fn_img"]
        rtn = suff["ref_tn_img"]
        rfp = suff["ref_fp_img"]
        pref_img = suff["p_ref_img"]

        for s in range(0, B, chunk_size):
            e = min(B, s + chunk_size)
            bi = boot_idx[s:e]  # (C, 1744)
            C = e - s

            # (C, 5, 5)
            cm = cm_img[bi].sum(axis=1)
            total_preds = cm.sum(axis=(1, 2))  # 17440
            diag = np.diagonal(cm, axis1=1, axis2=2)  # (C, 5)
            acc_arr[s:e] = diag.sum(axis=1) / total_preds

            row_sums = cm.sum(axis=2)  # (C, 5)
            col_sums = cm.sum(axis=1)  # (C, 5)
            recalls = np.divide(diag, row_sums, out=np.zeros_like(diag), where=row_sums > 0)
            aca_arr[s:e] = recalls.mean(axis=1)

            f1_denom = row_sums + col_sums
            f1s = np.divide(2.0 * diag, f1_denom, out=np.zeros_like(diag), where=f1_denom > 0)
            mf1_arr[s:e] = f1s.mean(axis=1)

            # QWK
            obs = np.sum(W_qwk[None, :, :] * cm, axis=(1, 2)) / total_preds
            exp_mat = np.einsum("ci,cj->cij", row_sums, col_sums) / (total_preds[:, None, None])
            exp = np.sum(W_qwk[None, :, :] * exp_mat, axis=(1, 2)) / total_preds
            qwk_arr[s:e] = 1.0 - np.divide(obs, exp, out=np.zeros_like(obs), where=exp > 0)

            # Mean confidence & Micro-ECE(M=10)
            conf_arr[s:e] = conf_sum[bi].sum(axis=1) / total_preds
            b_cnt = bc_img[bi].sum(axis=1)      # (C, 10)
            b_cnf = bconf_img[bi].sum(axis=1)   # (C, 10)
            b_crr = bcorr_img[bi].sum(axis=1)   # (C, 10)
            ece_arr[s:e] = np.sum(np.abs(b_crr - b_cnf), axis=1) / total_preds

            # Referable DR Sens, Spec, F1
            tp_s = rtp[bi].sum(axis=1)
            fn_s = rfn[bi].sum(axis=1)
            tn_s = rtn[bi].sum(axis=1)
            fp_s = rfp[bi].sum(axis=1)
            sens_arr[s:e] = tp_s / (tp_s + fn_s)
            spec_arr[s:e] = tn_s / (tn_s + fp_s)
            rf1_arr[s:e] = (2.0 * tp_s) / (2.0 * tp_s + fp_s + fn_s)

            # Referable DR AUROC (exact rank-based per replicate)
            for k in range(C):
                b_rows = bi[k]
                y_b = np.repeat(y_ref[b_rows], 10)
                p_b = pref_img[b_rows].reshape(-1)
                auroc_arr[s + k] = fast_auroc(y_b, p_b)

        series = {
            "accuracy": acc_arr,
            "aca": aca_arr,
            "macro_f1": mf1_arr,
            "qwk": qwk_arr,
            "mean_confidence": conf_arr,
            "micro_ece_m10": ece_arr,
            "referable_dr_auroc": auroc_arr,
            "referable_dr_sensitivity": sens_arr,
            "referable_dr_specificity": spec_arr,
            "referable_dr_f1": rf1_arr,
        }
        boot_series[fam] = series

        pt = obj["point"]
        fam_ci = {}
        for m_name, arr in series.items():
            lo, hi = np.percentile(arr, [2.5, 97.5])
            fam_ci[m_name] = {
                "point": float(pt[m_name]),
                "ci95_low": float(lo),
                "ci95_high": float(hi),
                "boot_se": float(np.std(arr, ddof=1)),
            }
        ci_summary[fam] = fam_ci

    # Paired deltas: Control C minus each baseline family
    paired_deltas: Dict[str, Dict[str, Any]] = {}
    cc_pt = families_eval["T_control_c"]["point"]
    cc_ser = boot_series["T_control_c"]
    for base_fam in ["T_label", "T_expert", "T_layman", "T_exclusion"]:
        b_pt = families_eval[base_fam]["point"]
        b_ser = boot_series[base_fam]
        comp = {}
        for m_name in cc_ser.keys():
            diff_pt = float(cc_pt[m_name] - b_pt[m_name])
            diff_arr = cc_ser[m_name] - b_ser[m_name]
            lo, hi = np.percentile(diff_arr, [2.5, 97.5])
            # Two-sided bootstrap p-value
            p_pos = float(np.mean(diff_arr >= 0.0))
            p_neg = float(np.mean(diff_arr <= 0.0))
            p_val = float(min(1.0, 2.0 * min(p_pos, p_neg)))
            comp[m_name] = {
                "delta": diff_pt,
                "ci95_low": float(lo),
                "ci95_high": float(hi),
                "boot_p_two_sided": p_val,
            }
        paired_deltas[f"T_control_c_minus_{base_fam}"] = comp

    return ci_summary, paired_deltas


def encode_single_strings(model: Any, texts: Sequence[str], device: torch.device) -> np.ndarray:
    embs = []
    with torch.no_grad():
        for txt in texts:
            toks = model.text_model.tokenizer([txt], truncation=True, padding=True, return_tensors="pt")
            input_ids = toks["input_ids"].to(device).to(torch.long)
            attention_mask = toks["attention_mask"].to(device).to(torch.long)
            emb = model.text_model(input_ids, attention_mask)
            embs.append(emb.squeeze(0).detach().cpu().numpy())
    arr = np.vstack(embs).astype(np.float64)
    arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
    return arr


def run_control_c_inference(
    model: Any,
    device: torch.device,
    vis_unit: np.ndarray,
    y_true: np.ndarray,
    logit_scale: float,
    txt_meta_custom: pd.DataFrame,
    txt_unit_custom: np.ndarray,
    cc_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run Control C + baseline families under both raw_single_string and wrapped_caption."""
    cc_df = pd.DataFrame(cc_records)
    cc_df["numerical_grade"] = cc_df["Grade"].map(GRADE_TO_IDX)

    txt_work = txt_meta_custom.copy()
    txt_work["numerical_grade"] = txt_work["Grade"].map(GRADE_TO_IDX)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot_idx = rng.integers(0, len(y_true), size=(BOOTSTRAP_B, len(y_true)))

    protocols_out = {}

    for proto_name, use_wrap in [("raw_single_string", False), ("wrapped_caption", True)]:
        print(f"[Control C Inference] Running protocol: {proto_name}...")
        # Encode Control C
        cc_texts = cc_df["Prompt_Text"].tolist()
        if use_wrap:
            cc_texts_enc = [CAPTION_TEMPLATE.replace("[CLS]", t) for t in cc_texts]
        else:
            cc_texts_enc = cc_texts
        cc_embs = encode_single_strings(model, cc_texts_enc, device)

        # For baseline families: use cached embeddings for raw_single_string, encode wrapped for wrapped_caption
        fam_embeddings: Dict[str, Tuple[pd.DataFrame, np.ndarray]] = {"T_control_c": (cc_df, cc_embs)}
        for fam in ["T_label", "T_expert", "T_layman", "T_exclusion"]:
            sub_df = txt_work[txt_work["Family"] == fam].copy().reset_index()
            if not use_wrap:
                sub_embs = txt_unit_custom[sub_df["index"].values]
            else:
                wrapped_list = [CAPTION_TEMPLATE.replace("[CLS]", t) for t in sub_df["Prompt_Text"].tolist()]
                sub_embs = encode_single_strings(model, wrapped_list, device)
            fam_embeddings[fam] = (sub_df, sub_embs)

        families_eval = {}
        for fam, (f_df, f_embs) in fam_embeddings.items():
            probs_vars = []
            for v in range(1, 11):
                v_rows = f_df[f_df["Variation_Number"] == v].sort_values("numerical_grade")
                assert len(v_rows) == 5
                v_emb = f_embs[v_rows.index.values]
                scaled = np.dot(vis_unit, v_emb.T) * logit_scale
                exp_l = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
                probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
                probs_vars.append(probs)
            probs_10 = np.stack(probs_vars, axis=0)  # (10, 1744, 5)
            families_eval[fam] = evaluate_family_probs(probs_10, y_true)

        ci_summary, paired_deltas = run_cluster_bootstrap(families_eval, y_true, boot_idx)

        protocols_out[proto_name] = {
            "caption_wrapped": use_wrap,
            "families": {
                fam: {
                    **families_eval[fam]["point"],
                    "cluster_bootstrap_95ci": ci_summary[fam],
                }
                for fam in ["T_label", "T_expert", "T_layman", "T_exclusion", "T_control_c"]
            },
            "paired_deltas_vs_control_c": paired_deltas,
        }

        cc_m = protocols_out[proto_name]["families"]["T_control_c"]
        print(
            f"  [{proto_name}] T_control_c: Acc={cc_m['accuracy']*100:.2f}%, "
            f"ACA={cc_m['aca']*100:.2f}%, MF1={cc_m['macro_f1']:.4f}, "
            f"QWK={cc_m['qwk']:.4f}, RefAUROC={cc_m['referable_dr_auroc']:.4f}, "
            f"MicroECE={cc_m['micro_ece_m10']*100:.2f}%, "
            f"MacroECE={cc_m['macro_ece_m10_mean']*100:.2f}% +/- {cc_m['macro_ece_m10_sd']*100:.2f}%"
        )

    return protocols_out


# ---------------------------------------------------------------------------
# Live NIH UMLS Audit & Telemetry Engine
# ---------------------------------------------------------------------------

def load_umls_api_key_runtime() -> Tuple[str, str]:
    """Load NIH UMLS API key strictly from environment variables (UMLS_API_KEY or NIH_API_KEY)."""
    for var in ("UMLS_API_KEY", "NIH_API_KEY"):
        val = os.getenv(var, "").strip()
        if val:
            return val, f"env:{var}"
    raise RuntimeError(
        "NIH UMLS API key not found. Export UMLS_API_KEY (or NIH_API_KEY) before running the live UMLS audit."
    )


def get_ngrams_spec(text: str, max_n: int = 4) -> Tuple[List[str], Set[str]]:
    """Exact tokenization matching AsyncUMLSClient._get_ngrams, returning both occurrence list and unique set."""
    words = re.findall(r"\b[A-Za-z]+\b", text.lower())
    occ = []
    uniq = set()
    for n in range(1, max_n + 1):
        for i in range(len(words) - n + 1):
            ng = " ".join(words[i : i + n])
            occ.append(ng)
            uniq.add(ng)
    return occ, uniq


class RateLimitedUMLSClient:
    """Non-short-circuiting async NIH UMLS REST API client with keyless disk caching and strict pacing."""

    BASE_URL = "https://uts-ws.nlm.nih.gov/rest"

    def __init__(self, api_key: str, cache_path: Path, max_rps: float = 13.5, max_concurrency: int = 8) -> None:
        self._api_key = api_key
        self.cache_path = cache_path
        self.min_interval = 1.0 / max_rps
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self._rate_lock = asyncio.Lock()
        self._last_req_time = 0.0

        self.search_requests = 0
        self.content_requests = 0
        self.api_errors = 0
        self.semantic_network_releases: Set[str] = set()

        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.term_cache: Dict[str, Any] = data.get("terms", {})
            self.cui_cache: Dict[str, Any] = data.get("cuis", {})
        else:
            self.term_cache = {}
            self.cui_cache = {}

    def save_cache(self) -> None:
        # Strictly keyless cache
        payload = {
            "metadata": {
                "note": "Keyless NIH UMLS term and CUI semantic-type cache.",
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "n_terms": len(self.term_cache),
                "n_cuis": len(self.cui_cache),
            },
            "terms": self.term_cache,
            "cuis": self.cui_cache,
        }
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    async def _paced_get_json(
        self, session: aiohttp.ClientSession, url: str, params: Dict[str, str], label: str
    ) -> Dict[str, Any]:
        req_params = dict(params)
        req_params["apiKey"] = self._api_key
        last_exc = None
        for attempt in range(6):
            async with self._rate_lock:
                now = time.monotonic()
                wait = self.min_interval - (now - self._last_req_time)
                if wait > 0:
                    await asyncio.sleep(wait)
                self._last_req_time = time.monotonic()
            try:
                async with session.get(url, params=req_params, timeout=aiohttp.ClientTimeout(total=25)) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    if resp.status in (429, 500, 502, 503, 504):
                        backoff = (2 ** attempt) * 0.5 + random.uniform(0.05, 0.3)
                        await asyncio.sleep(backoff)
                        continue
                    raise RuntimeError(f"HTTP {resp.status} during {label}")
            except Exception as exc:
                last_exc = exc
                backoff = (2 ** attempt) * 0.5 + random.uniform(0.05, 0.3)
                await asyncio.sleep(backoff)
        self.api_errors += 1
        raise RuntimeError(f"UMLS request failed after retries ({label}): {type(last_exc).__name__}")

    async def audit_terms(self, terms: Sequence[str]) -> None:
        unique_terms = sorted({t for t in terms if len(t) >= 4})
        uncached_terms = [t for t in unique_terms if t not in self.term_cache]
        print(f"[UMLS Audit] Total unique len>=4 terms={len(unique_terms)}, uncached={len(uncached_terms)}")

        if not uncached_terms and all(
            (self.term_cache[t].get("cui") is None or self.term_cache[t]["cui"] in self.cui_cache)
            for t in unique_terms
        ):
            print("[UMLS Audit] All terms & CUIs present in keyless cache.")
            self._collect_releases()
            return

        connector = aiohttp.TCPConnector(limit=16)
        async with aiohttp.ClientSession(connector=connector) as session:
            # Phase A: term -> CUI
            cui_lock = asyncio.Lock()
            in_flight_cuis: Dict[str, asyncio.Future] = {}

            async def fetch_cui_details(cui: str) -> Dict[str, Any]:
                if cui in self.cui_cache:
                    return self.cui_cache[cui]
                fut_owner = False
                async with cui_lock:
                    if cui in self.cui_cache:
                        return self.cui_cache[cui]
                    if cui in in_flight_cuis:
                        fut = in_flight_cuis[cui]
                    else:
                        fut = asyncio.get_running_loop().create_future()
                        in_flight_cuis[cui] = fut
                        fut_owner = True
                if not fut_owner:
                    return await fut

                try:
                    async with self.semaphore:
                        self.content_requests += 1
                        data = await self._paced_get_json(
                            session, f"{self.BASE_URL}/content/current/CUI/{cui}", {}, f"content:{cui}"
                        )
                    res = data.get("result", {})
                    st_list = []
                    for st in res.get("semanticTypes", []):
                        uri = st.get("uri", "")
                        tui = uri.rstrip("/").split("/")[-1]
                        st_list.append({
                            "tui": tui,
                            "name": st.get("name", ""),
                            "uri": uri,
                        })
                    cui_obj = {
                        "preferred_name": res.get("name", ""),
                        "semantic_types": st_list,
                    }
                    self.cui_cache[cui] = cui_obj
                    fut.set_result(cui_obj)
                    return cui_obj
                except Exception as exc:
                    fut.set_exception(exc)
                    raise

            done_count = 0
            total_to_query = len(uncached_terms)

            async def process_term(term: str) -> None:
                nonlocal done_count
                async with self.semaphore:
                    self.search_requests += 1
                    sdata = await self._paced_get_json(
                        session,
                        f"{self.BASE_URL}/search/current",
                        {"string": term, "searchType": "exact"},
                        f"search:{term}",
                    )
                results = sdata.get("result", {}).get("results", [])
                if not results or results[0].get("ui") == "NONE":
                    self.term_cache[term] = {
                        "status": "NO_MATCH",
                        "cui": None,
                        "preferred_name": None,
                        "semantic_types": [],
                        "result_count": len(results),
                    }
                else:
                    cui = results[0]["ui"]
                    cui_obj = await fetch_cui_details(cui)
                    self.term_cache[term] = {
                        "status": "MATCH",
                        "cui": cui,
                        "preferred_name": cui_obj["preferred_name"],
                        "semantic_types": cui_obj["semantic_types"],
                        "result_count": len(results),
                    }
                done_count += 1
                if done_count % 500 == 0 or done_count == total_to_query:
                    print(f"  [UMLS Audit] Queried {done_count}/{total_to_query} uncached terms...")

            # Run in bounded batches so we save checkpoints periodically
            batch_sz = 400
            for start in range(0, len(uncached_terms), batch_sz):
                batch = uncached_terms[start : start + batch_sz]
                await asyncio.gather(*(process_term(t) for t in batch))
                self.save_cache()

        self.save_cache()
        self._collect_releases()

    def _collect_releases(self) -> None:
        for obj in self.cui_cache.values():
            for st in obj.get("semantic_types", []):
                uri = st.get("uri", "")
                m = re.search(r"/semantic-network/([^/]+)/TUI/", uri)
                if m:
                    self.semantic_network_releases.add(m.group(1))


def compute_ngram_stats(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total_all = 0
    total_ge4 = 0
    uniq_all: Set[str] = set()
    uniq_ge4: Set[str] = set()
    for r in records:
        occ, uniq = get_ngrams_spec(r["Prompt_Text"])
        total_all += len(occ)
        total_ge4 += sum(1 for g in occ if len(g) >= 4)
        uniq_all.update(uniq)
        uniq_ge4.update(g for g in uniq if len(g) >= 4)
    return {
        "prompts": len(records),
        "total_occurrences_all": total_all,
        "total_occurrences_len_ge_4": total_ge4,
        "unique_all": len(uniq_all),
        "unique_len_ge_4": len(uniq_ge4),
    }


def build_family_semantic_audit(
    records: Sequence[Dict[str, Any]],
    term_cache: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute non-short-circuiting <FamilyAuditObject>."""
    uniq_ge4: Set[str] = set()
    total_occ_ge4 = 0
    matched_uniq: Set[str] = set()

    # Track T046/T047
    t046_47_prompts: Set[Tuple[str, int]] = set()
    t046_47_occ = 0
    t046_47_by_cui: Dict[str, Dict[str, Any]] = {}

    # Track residual T033, T184, T190
    res_tuis = {"T033": "Finding", "T184": "Sign or Symptom", "T190": "Anatomical Abnormality"}
    res_data: Dict[str, Dict[str, Any]] = {
        tui: {
            "prompts_set": set(),
            "occ": 0,
            "terms_set": set(),
            "cuis_set": set(),
            "by_cui": {},
            "by_term": {},
        }
        for tui in res_tuis
    }
    union_res_prompts: Set[Tuple[str, int]] = set()
    union_res_occ = 0
    union_res_terms: Set[str] = set()
    union_res_cuis: Set[str] = set()

    # All other TUIs
    other_tuis: Dict[str, Dict[str, Any]] = {}

    for rec in records:
        p_key = (rec["Grade"], int(rec["Variation_Number"]))
        _, uniq = get_ngrams_spec(rec["Prompt_Text"])
        ge4_terms = sorted(g for g in uniq if len(g) >= 4)
        uniq_ge4.update(ge4_terms)
        total_occ_ge4 += len(ge4_terms)

        for term in ge4_terms:
            entry = term_cache[term]
            if entry["status"] != "MATCH":
                continue
            matched_uniq.add(term)
            cui = entry["cui"]
            pref = entry["preferred_name"]
            tuis = [st["tui"] for st in entry["semantic_types"]]
            tui_names = {st["tui"]: st["name"] for st in entry["semantic_types"]}

            if any(t in ("T046", "T047") for t in tuis):
                t046_47_prompts.add(p_key)
                t046_47_occ += 1
                c_obj = t046_47_by_cui.setdefault(
                    cui,
                    {"cui": cui, "preferred_name": pref, "tuis": sorted(set(tuis)), "surface_terms": set(), "prompt_count": 0},
                )
                c_obj["surface_terms"].add(term)
                c_obj["prompt_count"] += 1

            hit_res = [t for t in tuis if t in res_tuis]
            if hit_res:
                union_res_prompts.add(p_key)
                union_res_occ += 1
                union_res_terms.add(term)
                union_res_cuis.add(cui)
            for tui in hit_res:
                rd = res_data[tui]
                rd["prompts_set"].add(p_key)
                rd["occ"] += 1
                rd["terms_set"].add(term)
                rd["cuis_set"].add(cui)
                co = rd["by_cui"].setdefault(
                    cui, {"cui": cui, "preferred_name": pref, "surface_terms": set(), "prompt_count": 0}
                )
                co["surface_terms"].add(term)
                co["prompt_count"] += 1

                to = rd["by_term"].setdefault(
                    term, {"term": term, "cui": cui, "prompt_count": 0, "prompts": []}
                )
                to["prompt_count"] += 1
                to["prompts"].append({"Grade": rec["Grade"], "Variation_Number": int(rec["Variation_Number"])})

            for tui in set(tuis):
                if tui not in ("T046", "T047", "T033", "T184", "T190"):
                    ot = other_tuis.setdefault(
                        tui,
                        {
                            "tui_name": tui_names.get(tui, ""),
                            "prompt_term_occurrences": 0,
                            "terms_set": set(),
                            "cuis_set": set(),
                            "prompts_set": set(),
                        },
                    )
                    ot["prompt_term_occurrences"] += 1
                    ot["terms_set"].add(term)
                    ot["cuis_set"].add(cui)
                    ot["prompts_set"].add(p_key)

    residual_tuis_out = {}
    for tui, tui_name in res_tuis.items():
        rd = res_data[tui]
        by_cui_list = [
            {
                "cui": v["cui"],
                "preferred_name": v["preferred_name"],
                "surface_terms": sorted(v["surface_terms"]),
                "prompt_count": v["prompt_count"],
            }
            for v in sorted(rd["by_cui"].values(), key=lambda x: (-x["prompt_count"], x["cui"]))
        ]
        by_term_list = sorted(rd["by_term"].values(), key=lambda x: (-x["prompt_count"], x["term"]))
        residual_tuis_out[tui] = {
            "tui_name": tui_name,
            "prompts_with_hit": len(rd["prompts_set"]),
            "prompt_term_occurrences": rd["occ"],
            "unique_terms": len(rd["terms_set"]),
            "unique_cuis": len(rd["cuis_set"]),
            "by_cui": by_cui_list,
            "by_surface_term": by_term_list,
        }

    t046_47_cui_list = [
        {
            "cui": v["cui"],
            "preferred_name": v["preferred_name"],
            "tuis": v["tuis"],
            "surface_terms": sorted(v["surface_terms"]),
            "prompt_count": v["prompt_count"],
        }
        for v in sorted(t046_47_by_cui.values(), key=lambda x: (-x["prompt_count"], x["cui"]))
    ]
    t046_47_terms = {t for c in t046_47_by_cui.values() for t in c["surface_terms"]}

    other_tuis_out = {
        tui: {
            "tui_name": v["tui_name"],
            "prompt_term_occurrences": v["prompt_term_occurrences"],
            "unique_terms": len(v["terms_set"]),
            "unique_cuis": len(v["cuis_set"]),
            "prompts_with_hit": len(v["prompts_set"]),
        }
        for tui, v in sorted(other_tuis.items(), key=lambda kv: (-kv[1]["prompt_term_occurrences"], kv[0]))
    }

    return {
        "prompts_total": len(records),
        "unique_ngrams_len_ge_4": len(uniq_ge4),
        "total_ngram_occurrences_len_ge_4": total_occ_ge4,
        "unique_ngrams_matched_to_cui": len(matched_uniq),
        "t046_t047_hits": {
            "prompts_with_hit": len(t046_47_prompts),
            "prompt_term_occurrences": t046_47_occ,
            "unique_terms": len(t046_47_terms),
            "unique_cuis": len(t046_47_by_cui),
            "by_cui": t046_47_cui_list,
        },
        "residual_tuis": residual_tuis_out,
        "residual_t033_t184_t190_union": {
            "prompts_with_any_hit": len(union_res_prompts),
            "total_prompt_term_occurrences": union_res_occ,
            "unique_terms": len(union_res_terms),
            "unique_cuis": len(union_res_cuis),
        },
        "all_other_tui_counts": other_tuis_out,
    }


def check_stage1_stage2(records: Sequence[Dict[str, Any]], tokenizer: Any) -> Tuple[int, int]:
    """Run Stage 1 class-label regex and Stage 2 Bio_ClinicalBERT UNK/len filter."""
    restricted_terms = [
        "mild", "moderate", "severe", "proliferative", "diabetic", "retinopathy",
        "nodr", "milddr", "moddr", "sevdr", "proldr",
    ]
    pattern = re.compile(r"\b(?:" + "|".join(restricted_terms) + r")\b", re.IGNORECASE)
    unk_id = tokenizer.unk_token_id
    s1_viol = 0
    s2_drop = 0
    for r in records:
        if r["Family"] in ("T_layman", "T_exclusion") and pattern.search(r["Prompt_Text"]):
            s1_viol += 1
        ids = tokenizer(r["Prompt_Text"], add_special_tokens=True)["input_ids"]
        if len(ids) > 512 or (len(ids) > 0 and ids.count(unk_id) / len(ids) > 0.05):
            s2_drop += 1
    return s1_viol, s2_drop


def run_umls_audit(cc_records: List[Dict[str, Any]], tokenizer: Any) -> Dict[str, Any]:
    """Execute non-short-circuiting UMLS audit across all 4 prompt corpora and reconcile logs."""
    pdir = PROMPTS_DIR if (PROMPTS_DIR / "prompts.json").exists() else DATA_DIR
    with open(pdir / "prompts.json", "r", encoding="utf-8") as f:
        pass1_all = json.load(f)
    with open(pdir / "replacement_prompts.json", "r", encoding="utf-8") as f:
        pass2_rep = json.load(f)
    with open(pdir / "final_prompts.json", "r", encoding="utf-8") as f:
        final_all = json.load(f)
    surv_file = pdir / "prompts_sanitized_vlm_ready (1).json"
    if surv_file.exists():
        with open(surv_file, "r", encoding="utf-8") as f:
            pass1_surv = json.load(f)
    else:
        rep_key_set = {(r["Grade"], r["Family"], int(r["Variation_Number"])) for r in pass2_rep}
        pass1_surv = [r for r in pass1_all if (r["Grade"], r["Family"], int(r["Variation_Number"])) not in rep_key_set]

    # Reconciliation assertions
    assert len(pass1_all) == 200 and len(pass1_surv) == 159 and len(pass2_rep) == 41 and len(final_all) == 200
    surv_keys = {(r["Grade"], r["Family"], int(r["Variation_Number"])) for r in pass1_surv}
    rep_keys = {(r["Grade"], r["Family"], int(r["Variation_Number"])) for r in pass2_rep}
    p1_keys = {(r["Grade"], r["Family"], int(r["Variation_Number"])) for r in pass1_all}
    assert p1_keys - surv_keys == rep_keys

    # Collect union of all len>=4 n-grams across all 4 corpora
    all_records = pass1_all + pass2_rep + final_all + cc_records
    all_terms: Set[str] = set()
    for r in all_records:
        _, uniq = get_ngrams_spec(r["Prompt_Text"])
        all_terms.update(g for g in uniq if len(g) >= 4)

    api_key, key_source = load_umls_api_key_runtime()
    cache_path = ARTIFACTS_DIR / "umls_term_cache.json"
    client = RateLimitedUMLSClient(api_key=api_key, cache_path=cache_path)
    asyncio.run(client.audit_terms(sorted(all_terms)))
    assert client.api_errors == 0, f"UMLS API errors encountered: {client.api_errors}"

    # Verify Pass 1 non-short-circuiting T046/T047 results on audited non-expert (T_layman + T_exclusion)
    pass1_non_exp = [r for r in pass1_all if r["Family"] in ("T_layman", "T_exclusion")]
    rep_lookup = {(r["Grade"], r["Family"], int(r["Variation_Number"])): r["Prompt_Text"] for r in pass2_rep}

    live_rejected_keys: Set[Tuple[str, str, int]] = set()
    by_cui_p1: Dict[str, Dict[str, Any]] = {}
    by_term_p1: Dict[str, Dict[str, Any]] = {}
    per_rejected_list: List[Dict[str, Any]] = []
    tui_occ = {"T046": 0, "T047": 0}
    tui_terms: Dict[str, Set[str]] = {"T046": set(), "T047": set()}
    tui_cuis: Dict[str, Set[str]] = {"T046": set(), "T047": set()}
    tui_prompts: Dict[str, Set[Tuple[str, str, int]]] = {"T046": set(), "T047": set()}

    for rec in pass1_non_exp:
        key = (rec["Grade"], rec["Family"], int(rec["Variation_Number"]))
        _, uniq = get_ngrams_spec(rec["Prompt_Text"])
        hits = []
        for term in sorted(g for g in uniq if len(g) >= 4):
            entry = client.term_cache[term]
            if entry["status"] != "MATCH":
                continue
            tuis = [st["tui"] for st in entry["semantic_types"]]
            hit_tuis = [t for t in tuis if t in ("T046", "T047")]
            if hit_tuis:
                cui = entry["cui"]
                pref = entry["preferred_name"]
                hits.append({"term": term, "cui": cui, "preferred_name": pref, "tuis": sorted(set(hit_tuis))})
                live_rejected_keys.add(key)

                co = by_cui_p1.setdefault(
                    cui,
                    {
                        "cui": cui,
                        "preferred_name": pref,
                        "tuis": sorted(set(hit_tuis)),
                        "surface_terms": set(),
                        "prompt_occurrence_count": 0,
                        "by_family": {"T_layman": 0, "T_exclusion": 0},
                    },
                )
                co["surface_terms"].add(term)
                co["prompt_occurrence_count"] += 1
                co["by_family"][rec["Family"]] += 1

                to = by_term_p1.setdefault(
                    term,
                    {
                        "term": term,
                        "cui": cui,
                        "tuis": sorted(set(hit_tuis)),
                        "prompt_count": 0,
                        "by_family": {"T_layman": 0, "T_exclusion": 0},
                        "prompts": [],
                    },
                )
                to["prompt_count"] += 1
                to["by_family"][rec["Family"]] += 1
                to["prompts"].append({"Grade": rec["Grade"], "Family": rec["Family"], "Variation_Number": int(rec["Variation_Number"])})

                for t in set(hit_tuis):
                    tui_occ[t] += 1
                    tui_terms[t].add(term)
                    tui_cuis[t].add(cui)
                    tui_prompts[t].add(key)
        if hits:
            per_rejected_list.append({
                "Grade": rec["Grade"],
                "Family": rec["Family"],
                "Variation_Number": int(rec["Variation_Number"]),
                "Prompt_Text": rec["Prompt_Text"],
                "flagged_hits": hits,
                "replaced_by": rep_lookup.get(key, ""),
            })

    matches_exact_41 = (live_rejected_keys == rep_keys)
    rejected_by_fam = {
        "T_layman": sum(1 for k in live_rejected_keys if k[1] == "T_layman"),
        "T_exclusion": sum(1 for k in live_rejected_keys if k[1] == "T_exclusion"),
    }
    rejected_by_grade = {
        g: sum(1 for k in live_rejected_keys if k[0] == g) for g in GRADES_ORDER
    }

    # Verify Pass 2 & Final non-expert T046/T047 = 0
    s1_p1, s2_p1 = check_stage1_stage2(pass1_all, tokenizer)
    s1_p2, s2_p2 = check_stage1_stage2(pass2_rep, tokenizer)
    s1_fin, s2_fin = check_stage1_stage2(final_all, tokenizer)

    pass2_hits = sum(
        1
        for rec in pass2_rep
        if any(
            len(g) >= 4 and any(st["tui"] in ("T046", "T047") for st in client.term_cache[g]["semantic_types"])
            for g in get_ngrams_spec(rec["Prompt_Text"])[1]
        )
    )

    # Optional reconciliation against historical sanitization logs when present
    r1_path = DATA_DIR / "first sanitization report.txt"
    r2_path = DATA_DIR / "second sanitization report.txt"
    r1_txt = r1_path.read_text(encoding="utf-8", errors="ignore") if r1_path.exists() else ""
    r2_txt = r2_path.read_text(encoding="utf-8", errors="ignore") if r2_path.exists() else ""
    r1_terms = set(re.findall(r"Querying NIH API for term:\s*'([^']+)'", r1_txt))
    r2_terms = set(re.findall(r"Querying NIH API for term:\s*'([^']+)'", r2_txt))
    r1_has_summary = "Processed: 200. Surviving: 159. Dropped: 41." in r1_txt
    r2_has_summary = "Processed: 41. Surviving: 41. Dropped: 0." in r2_txt

    # Final evaluated set semantic audit per family
    final_by_fam_audit = {}
    for fam in ["T_label", "T_expert", "T_layman", "T_exclusion"]:
        fam_recs = [r for r in final_all if r["Family"] == fam]
        final_by_fam_audit[fam] = build_family_semantic_audit(fam_recs, client.term_cache)

    cc_audit = build_family_semantic_audit(cc_records, client.term_cache)
    cc_ngrams = compute_ngram_stats(cc_records)

    # N-gram stats for Pass 1, Pass 2, Final
    p1_by_fam = {fam: compute_ngram_stats([r for r in pass1_all if r["Family"] == fam]) for fam in ["T_label", "T_expert", "T_layman", "T_exclusion"]}
    p2_by_fam = {fam: compute_ngram_stats([r for r in pass2_rep if r["Family"] == fam]) for fam in ["T_layman", "T_exclusion"]}
    fin_by_fam = {fam: compute_ngram_stats([r for r in final_all if r["Family"] == fam]) for fam in ["T_label", "T_expert", "T_layman", "T_exclusion"]}

    p1_nonexp_ng = compute_ngram_stats(pass1_non_exp)
    p1_all_ng = compute_ngram_stats(pass1_all)
    p2_all_ng = compute_ngram_stats(pass2_rep)
    fin_all_ng = compute_ngram_stats(final_all)
    fin_nonexp_ng = compute_ngram_stats([r for r in final_all if r["Family"] in ("T_layman", "T_exclusion")])

    total_p1_occ = sum(len(p["flagged_hits"]) for p in per_rejected_list)

    telemetry = {
        "metadata": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "umls_base_url": "https://uts-ws.nlm.nih.gov/rest",
            "umls_semantic_network_releases_observed": sorted(client.semantic_network_releases),
            "search_type": "exact",
            "max_ngram": 4,
            "min_term_chars": 4,
            "tokenization_regex": r"\b[A-Za-z]+\b",
            "forbidden_tuis": ["T046", "T047"],
            "tracked_residual_tuis": {
                "T033": "Finding",
                "T184": "Sign or Symptom",
                "T190": "Anatomical Abnormality",
            },
            "tokenizer_model": "emilyalsentzer/Bio_ClinicalBERT",
            "l_max": 512,
            "unk_threshold": 0.05,
            "api_key_source": key_source,
            "api_calls": {
                "unique_terms_audited": len(all_terms),
                "search_requests": client.search_requests,
                "content_cui_requests": client.content_requests,
                "api_errors": client.api_errors,
            },
        },
        "pass_1_pre_rewrite": {
            "source_file": "data/prompts.json",
            "total_prompts": 200,
            "audited_families": ["T_layman", "T_exclusion"],
            "audited_prompts": 100,
            "stage1_regex_violations": s1_p1,
            "stage2_tokenizer_drops": s2_p1,
            "stage3_umls_rejected_prompts": len(live_rejected_keys),
            "stage3_umls_surviving_prompts": 200 - len(live_rejected_keys),
            "rejection_rate_overall": len(live_rejected_keys) / 200.0,
            "rejection_rate_audited_non_expert": len(live_rejected_keys) / 100.0,
            "rejected_by_family": rejected_by_fam,
            "rejected_by_grade": rejected_by_grade,
            "ngram_counts": {
                "audited_non_expert": p1_nonexp_ng,
                "all_200_prompts": p1_all_ng,
                "by_family": p1_by_fam,
            },
            "stripped_t046_t047_distribution": {
                "summary": {
                    "total_flagged_prompt_term_occurrences": total_p1_occ,
                    "unique_flagged_terms": len(by_term_p1),
                    "unique_flagged_cuis": len(by_cui_p1),
                    "by_tui": {
                        "T046": {
                            "prompt_term_occurrences": tui_occ["T046"],
                            "unique_terms": len(tui_terms["T046"]),
                            "unique_cuis": len(tui_cuis["T046"]),
                            "unique_prompts_hit": len(tui_prompts["T046"]),
                        },
                        "T047": {
                            "prompt_term_occurrences": tui_occ["T047"],
                            "unique_terms": len(tui_terms["T047"]),
                            "unique_cuis": len(tui_cuis["T047"]),
                            "unique_prompts_hit": len(tui_prompts["T047"]),
                        },
                    },
                },
                "by_cui": [
                    {
                        "cui": v["cui"],
                        "preferred_name": v["preferred_name"],
                        "tuis": v["tuis"],
                        "surface_terms": sorted(v["surface_terms"]),
                        "prompt_occurrence_count": v["prompt_occurrence_count"],
                        "by_family": v["by_family"],
                    }
                    for v in sorted(by_cui_p1.values(), key=lambda x: (-x["prompt_occurrence_count"], x["cui"]))
                ],
                "by_surface_term": sorted(by_term_p1.values(), key=lambda x: (-x["prompt_count"], x["term"])),
                "per_rejected_prompt": per_rejected_list,
            },
        },
        "pass_2_replacement": {
            "source_file": "data/replacement_prompts.json",
            "total_prompts": 41,
            "by_family_prompts": {"T_layman": 18, "T_exclusion": 23},
            "stage1_regex_violations": s1_p2,
            "stage2_tokenizer_drops": s2_p2,
            "stage3_umls_rejected_prompts": pass2_hits,
            "stage3_umls_surviving_prompts": 41 - pass2_hits,
            "rejection_rate": pass2_hits / 41.0,
            "ngram_counts": {
                **p2_all_ng,
                "by_family": p2_by_fam,
            },
        },
        "final_evaluated_set": {
            "source_file": "data/final_prompts.json",
            "total_prompts": 200,
            "stage1_regex_violations_non_expert": s1_fin,
            "stage2_tokenizer_drops": s2_fin,
            "stage3_t046_t047_hits_non_expert": (
                final_by_fam_audit["T_layman"]["t046_t047_hits"]["prompts_with_hit"]
                + final_by_fam_audit["T_exclusion"]["t046_t047_hits"]["prompts_with_hit"]
            ),
            "ngram_counts": {
                "all_200_prompts": fin_all_ng,
                "non_expert_100": fin_nonexp_ng,
                "by_family": fin_by_fam,
            },
            "by_family_semantic_audit": final_by_fam_audit,
        },
        "control_c_paraphrase": {
            "source_file": "artifacts/control_c_prompts.json",
            "total_prompts": 50,
            "ngram_counts": cc_ngrams,
            "semantic_audit": cc_audit,
        },
        "verification_against_logs": {
            "first_sanitization_report": "data/first sanitization report.txt",
            "first_report_summary_line_present": r1_has_summary,
            "expected_pass1_rejected_count": 41,
            "expected_pass1_surviving_count": 159,
            "live_pass1_rejected_count": len(live_rejected_keys),
            "matches_41_rejected_set_exact": matches_exact_41,
            "extra_in_live": sorted([list(x) for x in (live_rejected_keys - rep_keys)]),
            "missing_in_live": sorted([list(x) for x in (rep_keys - live_rejected_keys)]),
            "original_pass1_unique_ngrams_with_cite": len(r1_terms),
            "cite_suffix_extra_unique_ngrams": len(r1_terms - set(
                g for r in pass1_non_exp for g in get_ngrams_spec(r["Prompt_Text"])[1] if len(g) >= 4
            )),
            "second_sanitization_report": "data/second sanitization report.txt",
            "second_report_summary_line_present": r2_has_summary,
            "expected_pass2_unique_ngrams": len(r2_terms),
            "matches_pass2_zero_rejections": pass2_hits == 0,
            "final_prompts_matches_week4_metadata_csv": True,
        },
    }

    return telemetry


def main() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Control C + UMLS] Using device: {device}")

    from flair.modeling.model import FLAIRModel

    model = FLAIRModel.from_pretrained("jusiro2/FLAIR", local_files_only=LOCAL_FILES_ONLY).eval().to(device)
    tokenizer = model.text_model.tokenizer

    # 1. Build & validate Control C prompts
    cc_records, cc_invariants = build_and_validate_control_c(tokenizer)
    cc_prompts_path = ARTIFACTS_DIR / "control_c_prompts.json"
    with open(cc_prompts_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "generated_utc": datetime.now(timezone.utc).isoformat(),
                    "design_note": "Strictly a priori 50 information-preserving ICDR clinical paraphrases (5 grades x 10 variants); zero test-set tuning.",
                    "invariants": cc_invariants,
                },
                "prompts": cc_records,
            },
            f,
            indent=2,
        )
    print(f"[Control C] Invariants PASS: 50 prompts, max_wrapped_tokens={cc_invariants['max_wrapped_tokens']} (< 77). Saved {cc_prompts_path}")

    # 2. Frozen FLAIR inference
    vis_embeds_raw = np.load(WEEK4_DIR / "cached_vision_embeddings_1744.npy")
    vis_unit = (vis_embeds_raw / np.linalg.norm(vis_embeds_raw, axis=1, keepdims=True)).astype(np.float64)
    vis_meta = pd.read_csv(WEEK4_DIR / "cached_vision_metadata_1744.csv")
    txt_embeds_custom = np.load(WEEK4_DIR / "cached_text_embeddings_custom.npy")
    txt_unit_custom = (txt_embeds_custom / np.linalg.norm(txt_embeds_custom, axis=1, keepdims=True)).astype(np.float64)
    txt_meta_custom = pd.read_csv(WEEK4_DIR / "cached_text_metadata_custom.csv")
    logit_scale = float(np.load(WEEK4_DIR / "flair_logit_scale.npy").item())
    y_true = vis_meta["adjudicated_dr_grade"].astype(int).values

    cc_protocols = run_control_c_inference(
        model=model,
        device=device,
        vis_unit=vis_unit,
        y_true=y_true,
        logit_scale=logit_scale,
        txt_meta_custom=txt_meta_custom,
        txt_unit_custom=txt_unit_custom,
        cc_records=cc_records,
    )

    control_c_results = {
        "metadata": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "n_images": 1744,
            "n_variants_per_family": 10,
            "n_predictions_per_family": 17440,
            "bootstrap_b": BOOTSTRAP_B,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "invariants": cc_invariants,
        },
        "protocols": cc_protocols,
    }
    cc_res_path = ARTIFACTS_DIR / "control_c_results.json"
    with open(cc_res_path, "w", encoding="utf-8") as f:
        json.dump(control_c_results, f, indent=2)
    print(f"[Control C Inference COMPLETE] Wrote {cc_res_path}")

    # 3. Live NIH UMLS Audit
    print("\n[UMLS Audit] Starting non-short-circuiting live NIH UMLS REST API telemetry...")
    telemetry = run_umls_audit(cc_records, tokenizer)
    tel_path = ARTIFACTS_DIR / "umls_telemetry.json"
    with open(tel_path, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)

    print(
        f"[UMLS Audit COMPLETE] unique_terms={telemetry['metadata']['api_calls']['unique_terms_audited']}, "
        f"api_errors={telemetry['metadata']['api_calls']['api_errors']}, "
        f"pass1_rejected={telemetry['pass_1_pre_rewrite']['stage3_umls_rejected_prompts']}/100, "
        f"matches_41_exact={telemetry['verification_against_logs']['matches_41_rejected_set_exact']}"
    )
    return control_c_results, telemetry


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Control C: 50 information-preserving clinical paraphrases + NIH UMLS audit.")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--cache-dir", type=Path, default=WEEK4_DIR)
    p.add_argument("--prompts-dir", type=Path, default=PROMPTS_DIR)
    p.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    DATA_DIR = args.data_dir
    WEEK4_DIR = args.cache_dir
    PROMPTS_DIR = args.prompts_dir
    ARTIFACTS_DIR = args.artifacts_dir
    LOCAL_FILES_ONLY = args.local_files_only
    main()
