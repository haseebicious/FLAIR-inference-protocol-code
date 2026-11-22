"""Milestone 2 — Statistical Re-Analysis, Clinical Benchmarks, & Calibration Robustness.

Computes all Milestone 2 analyses directly from cached Week 4 embeddings and metadata,
MESSIDOR-2 grading CSV, thesis_audit_results_1744.csv, and Milestone 1 outputs:

1. Deterministic Reconstruction & Gate Assertions (`core_table2`)
2. Cluster-Robust Repeated-Measures Significance (`cluster_significance`):
   - B=10,000 image-level cluster-robust paired bootstrap (seed=42, shared boot_idx & W_all)
   - Joint 4-level and Pairwise 2-level Image-Clustered GEE Logistic Regression (Exchangeable)
   - Legacy pooled McNemar reference values (explicitly flagged invalid due to 10x clustering)
3. Standard MESSIDOR-2 Clinical Benchmarks (`clinical_benchmarks`)
4. Calibration Robustness & 5-Fold Stratified Image-Level CV Temperature Scaling (`calibration_robustness`)
5. Complete 40-Prompt Variance Breakdown & T_layman Diagnosis (`per_prompt_variance`)
6. Latent Geometry Check (`latent_geometry_check`)
7. Consolidated Authoritative Ledger (`artifacts/m2_verified_metrics.json`, `artifacts/verified_metrics.json`)
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import minimize_scalar
from scipy.sparse import csr_matrix
from scipy.special import logsumexp
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.contingency_tables import mcnemar


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = WORKSPACE_ROOT / "data"
WEEK4_DIR = (WORKSPACE_ROOT / "results") if (WORKSPACE_ROOT / "results" / "cached_vision_embeddings_1744.npy").exists() else (
    (DATA_DIR / "Week 4 Work") if (DATA_DIR / "Week 4 Work").exists() else DATA_DIR
)
MESSIDOR_CSV = (DATA_DIR / "messidor_data.csv") if (DATA_DIR / "messidor_data.csv").exists() else (DATA_DIR / "Messidor-2" / "Third Party Grading" / "messidor_data.csv")
ARTIFACTS_DIR = WORKSPACE_ROOT / "artifacts"

FAMILIES: List[str] = ["T_label", "T_expert", "T_layman", "T_exclusion"]
IDX_TO_GRADE: List[str] = ["noDR", "mildDR", "modDR", "sevDR", "prolDR"]
GRADE_TO_IDX: Dict[str, int] = {g: i for i, g in enumerate(IDX_TO_GRADE)}

TAXONOMY_PAIRS: List[Tuple[str, str]] = [
    ("T_label", "T_expert"),
    ("T_label", "T_layman"),
    ("T_label", "T_exclusion"),
    ("T_expert", "T_layman"),
    ("T_expert", "T_exclusion"),
    ("T_layman", "T_exclusion"),
]

BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 42
CV_N_SPLITS = 5
CV_SEED = 42

EXPECTED_TOP_LEVEL_KEYS: List[str] = [
    "metadata",
    "control_a",
    "control_b",
    "control_c",
    "umls_telemetry",
    "core_table2",
    "cluster_significance",
    "clinical_benchmarks",
    "calibration_robustness",
    "per_prompt_variance",
    "latent_geometry_check",
]

# Expected main.tex Table 2 & Table 3 values (2-decimal formatted strings) for exact gate assertion
MAIN_TEX_TABLE2_EXPECTED = {
    "T_label": {"acc": "56.18", "aca": "51.10", "ece": "4.45", "conf": "60.63"},
    "T_expert": {"acc": "45.23", "aca": "31.69", "ece": "32.57", "conf": "77.80"},
    "T_layman": {"acc": "29.62", "aca": "22.29", "ece": "38.70", "conf": "68.27"},
    "T_exclusion": {"acc": "26.90", "aca": "22.09", "ece": "36.15", "conf": "63.06"},
}
MAIN_TEX_TABLE3_EXPECTED = {
    "T_label": {"acc_mean": "56.18", "acc_sd": "6.67", "ece_mean": "7.61", "ece_sd": "7.35"},
    "T_expert": {"acc_mean": "45.23", "acc_sd": "14.20", "ece_mean": "32.59", "ece_sd": "14.58"},
    "T_layman": {"acc_mean": "29.62", "acc_sd": "18.39", "ece_mean": "39.22", "ece_sd": "23.51"},
    "T_exclusion": {"acc_mean": "26.90", "acc_sd": "15.57", "ece_mean": "36.16", "ece_sd": "10.85"},
}


def calculate_ece_equal_width(
    confidences: np.ndarray, accuracies: np.ndarray, num_bins: int
) -> float:
    """Uniform-bin ECE matching 3_inference_and_audit.py and control_a_baseline.py."""
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for i in range(num_bins):
        lo = bin_boundaries[i]
        hi = bin_boundaries[i + 1]
        if i == num_bins - 1:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences >= lo) & (confidences < hi)
        prop_in_bin = float(np.mean(in_bin))
        if prop_in_bin > 0.0:
            acc_in_bin = float(np.mean(accuracies[in_bin]))
            conf_in_bin = float(np.mean(confidences[in_bin]))
            ece += abs(acc_in_bin - conf_in_bin) * prop_in_bin
    return float(ece)


def calculate_ece_adaptive(
    confidences: np.ndarray, accuracies: np.ndarray, num_bins: int
) -> float:
    """Equal-mass adaptive ECE via deterministic mergesort + array_split."""
    n = len(confidences)
    order = np.argsort(confidences, kind="mergesort")
    bins = np.array_split(order, num_bins)
    ece = 0.0
    for b_idx in bins:
        if len(b_idx) == 0:
            continue
        prop = len(b_idx) / n
        acc_b = float(np.mean(accuracies[b_idx]))
        conf_b = float(np.mean(confidences[b_idx]))
        ece += abs(acc_b - conf_b) * prop
    return float(ece)


def calculate_multiclass_brier(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Multi-class Brier Score in [0, 2]: mean_i sum_c (p_{i,c} - 1[y_i == c])^2."""
    n = probs.shape[0]
    y_oh = np.zeros_like(probs, dtype=np.float64)
    y_oh[np.arange(n), y_true] = 1.0
    return float(np.mean(np.sum((probs - y_oh) ** 2, axis=1)))


def calculate_nll_from_logits(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Numerically stable mean negative log-likelihood (cross-entropy) from logits."""
    lse = logsumexp(logits, axis=-1)
    y_logits = logits[np.arange(len(y_true)), y_true]
    return float(np.mean(lse - y_logits))


def compute_pooled_point_metrics(
    probs_10: np.ndarray, cos_10: np.ndarray, y_true: np.ndarray, tau_native: float
) -> Dict[str, Any]:
    """Compute exact point estimates for all 23+ metrics on pooled (10, 1744, 5) arrays."""
    probs = probs_10.reshape(-1, 5)
    cos_flat = cos_10.reshape(-1, 5)
    y_pooled = np.tile(y_true, probs_10.shape[0])
    n_tot = len(y_pooled)

    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    correct = (preds == y_pooled).astype(np.float64)

    acc = float(np.mean(correct))
    aca = float(balanced_accuracy_score(y_pooled, preds))
    per_grade_f1_arr = f1_score(
        y_pooled, preds, labels=[0, 1, 2, 3, 4], average=None, zero_division=0
    )
    macro_f1 = float(np.mean(per_grade_f1_arr))
    qwk = float(
        cohen_kappa_score(y_pooled, preds, labels=[0, 1, 2, 3, 4], weights="quadratic")
    )
    mean_conf = float(np.mean(confs))

    # Referable DR: grade >= 2 vs 0-1
    y_ref = (y_pooled >= 2).astype(int)
    p_ref = probs[:, 2] + probs[:, 3] + probs[:, 4]
    auroc = float(roc_auc_score(y_ref, p_ref))

    # Primary rule: P(RDR) >= 0.5
    pred_ref_p05 = (p_ref >= 0.5).astype(int)
    tp05 = int(np.sum((y_ref == 1) & (pred_ref_p05 == 1)))
    fn05 = int(np.sum((y_ref == 1) & (pred_ref_p05 == 0)))
    tn05 = int(np.sum((y_ref == 0) & (pred_ref_p05 == 0)))
    fp05 = int(np.sum((y_ref == 0) & (pred_ref_p05 == 1)))
    p05_sens = float(tp05 / (tp05 + fn05))
    p05_spec = float(tn05 / (tn05 + fp05))
    p05_f1 = float((2 * tp05) / (2 * tp05 + fp05 + fn05))

    # Secondary rule: argmax(probs) >= 2
    pred_ref_arg = (preds >= 2).astype(int)
    tpa = int(np.sum((y_ref == 1) & (pred_ref_arg == 1)))
    fna = int(np.sum((y_ref == 1) & (pred_ref_arg == 0)))
    tna = int(np.sum((y_ref == 0) & (pred_ref_arg == 0)))
    fpa = int(np.sum((y_ref == 0) & (pred_ref_arg == 1)))
    arg_sens = float(tpa / (tpa + fna))
    arg_spec = float(tna / (tna + fpa))
    arg_f1 = float((2 * tpa) / (2 * tpa + fpa + fna))

    ece_m10 = calculate_ece_equal_width(confs, correct, 10)
    ece_m15 = calculate_ece_equal_width(confs, correct, 15)
    ece_m20 = calculate_ece_equal_width(confs, correct, 20)
    aece_m10 = calculate_ece_adaptive(confs, correct, 10)
    aece_m15 = calculate_ece_adaptive(confs, correct, 15)
    brier = calculate_multiclass_brier(probs, y_pooled)
    nll = calculate_nll_from_logits(cos_flat * tau_native, y_pooled)

    pred_counts = {str(c): int(np.sum(preds == c)) for c in range(5)}

    return {
        "n_predictions": int(n_tot),
        "n_unique_confidences": int(len(np.unique(confs))),
        "accuracy": acc,
        "aca": aca,
        "macro_f1": macro_f1,
        "per_grade_f1_0": float(per_grade_f1_arr[0]),
        "per_grade_f1_1": float(per_grade_f1_arr[1]),
        "per_grade_f1_2": float(per_grade_f1_arr[2]),
        "per_grade_f1_3": float(per_grade_f1_arr[3]),
        "per_grade_f1_4": float(per_grade_f1_arr[4]),
        "qwk": qwk,
        "referable_dr_auroc": auroc,
        "referable_dr_p05_sensitivity": p05_sens,
        "referable_dr_p05_specificity": p05_spec,
        "referable_dr_p05_f1": p05_f1,
        "referable_dr_argmax_sensitivity": arg_sens,
        "referable_dr_argmax_specificity": arg_spec,
        "referable_dr_argmax_f1": arg_f1,
        "micro_ece_m10": ece_m10,
        "micro_ece_m15": ece_m15,
        "micro_ece_m20": ece_m20,
        "adaptive_ece_m10": aece_m10,
        "adaptive_ece_m15": aece_m15,
        "brier_score": brier,
        "nll": nll,
        "mean_confidence": mean_conf,
        "predicted_class_counts": pred_counts,
    }


def compute_variant_metrics(
    probs: np.ndarray, cos_mat: np.ndarray, y_true: np.ndarray, tau_native: float
) -> Dict[str, Any]:
    """Compute per-variant metrics for N=1,744 images."""
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    correct = (preds == y_true).astype(np.float64)

    acc = float(np.mean(correct))
    aca = float(balanced_accuracy_score(y_true, preds))
    macro_f1 = float(
        f1_score(y_true, preds, labels=[0, 1, 2, 3, 4], average="macro", zero_division=0)
    )
    qwk = float(
        cohen_kappa_score(y_true, preds, labels=[0, 1, 2, 3, 4], weights="quadratic")
    )
    y_ref = (y_true >= 2).astype(int)
    p_ref = probs[:, 2] + probs[:, 3] + probs[:, 4]
    auroc = float(roc_auc_score(y_ref, p_ref))

    ece_m10 = calculate_ece_equal_width(confs, correct, 10)
    ece_m15 = calculate_ece_equal_width(confs, correct, 15)
    aece_m10 = calculate_ece_adaptive(confs, correct, 10)
    brier = calculate_multiclass_brier(probs, y_true)
    nll = calculate_nll_from_logits(cos_mat * tau_native, y_true)
    mean_conf = float(np.mean(confs))

    counts = {str(c): int(np.sum(preds == c)) for c in range(5)}
    dom_idx = int(max(range(5), key=lambda c: counts[str(c)]))
    dom_share = float(counts[str(dom_idx)] / len(y_true))

    return {
        "accuracy": acc,
        "aca": aca,
        "macro_f1": macro_f1,
        "qwk": qwk,
        "referable_dr_auroc": auroc,
        "ece_m10": ece_m10,
        "ece_m15": ece_m15,
        "adaptive_ece_m10": aece_m10,
        "brier_score": brier,
        "nll": nll,
        "mean_confidence": mean_conf,
        "dominant_predicted_class_idx": dom_idx,
        "dominant_predicted_class_name": IDX_TO_GRADE[dom_idx],
        "dominant_predicted_share": dom_share,
        "predicted_class_counts": counts,
    }


def summarize_macro_across_variants(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    q25, q50, q75 = np.percentile(arr, [25.0, 50.0, 75.0])
    return {
        "mean": float(np.mean(arr)),
        "sd_ddof0": float(np.std(arr, ddof=0)),
        "sd_ddof1": float(np.std(arr, ddof=1)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(q50),
        "iqr": float(q75 - q25),
    }


def verify_and_build_core_table2(
    probs_by_fam: Dict[str, np.ndarray],
    cos_by_fam: Dict[str, np.ndarray],
    y_true: np.ndarray,
    tau_native: float,
    vis_meta: pd.DataFrame,
    messidor_df: pd.DataFrame,
    thesis_csv: pd.DataFrame,
    m1_data: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """Reconstruct Tables 2-3 deterministically and assert exact parity with M1 & CSVs."""
    m1_gate0 = m1_data["control_a"]["gate0_verification"]
    m1_base = m1_gate0["cached_baseline_reproduction_matches_csv"]

    point_by_fam: Dict[str, Dict[str, Any]] = {}
    variants_by_fam: Dict[str, List[Dict[str, Any]]] = {}
    core_by_fam: Dict[str, Any] = {}

    csv_lookup = thesis_csv.set_index("Prompt Family")

    for fam in FAMILIES:
        probs_10 = probs_by_fam[fam]
        cos_10 = cos_by_fam[fam]
        pt = compute_pooled_point_metrics(probs_10, cos_10, y_true, tau_native)
        point_by_fam[fam] = pt

        # Assert < 1e-12 vs M1 Gate 0 baseline metrics
        m1_fam = m1_base[fam]
        assert abs(pt["accuracy"] - m1_fam["accuracy"]) < 1e-12
        assert abs(pt["aca"] - m1_fam["aca"]) < 1e-12
        assert abs(pt["macro_f1"] - m1_fam["macro_f1"]) < 1e-12
        assert abs(pt["qwk"] - m1_fam["qwk"]) < 1e-12
        assert abs(pt["referable_dr_auroc"] - m1_fam["referable_dr_auroc"]) < 1e-12
        assert abs(pt["referable_dr_p05_sensitivity"] - m1_fam["referable_dr_sensitivity"]) < 1e-12
        assert abs(pt["referable_dr_p05_specificity"] - m1_fam["referable_dr_specificity"]) < 1e-12
        assert abs(pt["referable_dr_p05_f1"] - m1_fam["referable_dr_f1"]) < 1e-12
        assert abs(pt["micro_ece_m10"] - m1_fam["ece_m10"]) < 1e-12
        assert abs(pt["mean_confidence"] - m1_fam["mean_confidence"]) < 1e-12
        assert pt["predicted_class_counts"] == m1_fam["predicted_class_counts"]

        # Assert exact thesis_audit_results_1744.csv & main.tex Table 2 reproduction
        csv_row = csv_lookup.loc[fam]
        assert f"{pt['accuracy']*100:.2f}%" == str(csv_row["Accuracy"]).strip()
        assert f"{pt['micro_ece_m10']*100:.2f}%" == str(csv_row["ECE"]).strip()
        assert f"{pt['mean_confidence']*100:.2f}%" == str(csv_row["Avg Confidence"]).strip()

        t2_exp = MAIN_TEX_TABLE2_EXPECTED[fam]
        assert f"{pt['accuracy']*100:.2f}" == t2_exp["acc"]
        assert f"{pt['aca']*100:.2f}" == t2_exp["aca"]
        assert f"{pt['micro_ece_m10']*100:.2f}" == t2_exp["ece"]
        assert f"{pt['mean_confidence']*100:.2f}" == t2_exp["conf"]

        # Compute per-variant metrics (v=1..10)
        var_rows: List[Dict[str, Any]] = []
        for v_idx in range(10):
            vm = compute_variant_metrics(probs_10[v_idx], cos_10[v_idx], y_true, tau_native)
            vm["variation_number"] = v_idx + 1
            var_rows.append(vm)
        variants_by_fam[fam] = var_rows

        macro_acc = summarize_macro_across_variants([r["accuracy"] for r in var_rows])
        macro_ece = summarize_macro_across_variants([r["ece_m10"] for r in var_rows])
        macro_aca = summarize_macro_across_variants([r["aca"] for r in var_rows])
        macro_mf1 = summarize_macro_across_variants([r["macro_f1"] for r in var_rows])
        macro_qwk = summarize_macro_across_variants([r["qwk"] for r in var_rows])
        macro_conf = summarize_macro_across_variants([r["mean_confidence"] for r in var_rows])

        # Assert exact main.tex Table 3 (ddof=0) reproduction
        t3_exp = MAIN_TEX_TABLE3_EXPECTED[fam]
        assert f"{macro_acc['mean']*100:.2f}" == t3_exp["acc_mean"]
        assert f"{macro_acc['sd_ddof0']*100:.2f}" == t3_exp["acc_sd"]
        assert f"{macro_ece['mean']*100:.2f}" == t3_exp["ece_mean"]
        assert f"{macro_ece['sd_ddof0']*100:.2f}" == t3_exp["ece_sd"]

        core_by_fam[fam] = {
            "n_predictions": pt["n_predictions"],
            "accuracy": pt["accuracy"],
            "aca": pt["aca"],
            "macro_f1": pt["macro_f1"],
            "qwk": pt["qwk"],
            "micro_ece_m10": pt["micro_ece_m10"],
            "mean_confidence": pt["mean_confidence"],
            "per_grade_f1": {
                IDX_TO_GRADE[c]: pt[f"per_grade_f1_{c}"] for c in range(5)
            },
            "predicted_class_counts": pt["predicted_class_counts"],
            "macro_summary": {
                "sd_convention_note": "ddof=0 matches main.tex Table 3; ddof=1 matches M1 control_c sample SD.",
                "accuracy": macro_acc,
                "ece_m10": macro_ece,
                "aca": macro_aca,
                "macro_f1": macro_mf1,
                "qwk": macro_qwk,
                "mean_confidence": macro_conf,
            },
        }

    # Verify T_exclusion noDR -> prolDR conditional FPR (2444 / 10170)
    texcl_preds = np.argmax(probs_by_fam["T_exclusion"].reshape(-1, 5), axis=1)
    y_pooled = np.tile(y_true, 10)
    nodr_mask = y_pooled == 0
    nodr_denom = int(np.sum(nodr_mask))
    nodr_to_proldr_num = int(np.sum(nodr_mask & (texcl_preds == 4)))
    assert nodr_denom == 10170
    assert nodr_to_proldr_num == 2444

    # Verify T_exclusion minority F1s reported at main.tex L251 (0.055, 0.036)
    assert f"{point_by_fam['T_exclusion']['per_grade_f1_3']:.3f}" == "0.055"
    assert f"{point_by_fam['T_exclusion']['per_grade_f1_4']:.3f}" == "0.036"

    # MESSIDOR-2 distribution audit (1748 total -> 4 ungradable excluded -> 1744 gradable)
    total_rows = int(len(messidor_df))
    ungradable_df = messidor_df[messidor_df["adjudicated_dr_grade"].isna()]
    gradable_df = messidor_df[messidor_df["adjudicated_dr_grade"].notna()].copy()
    assert total_rows == 1748
    assert len(ungradable_df) == 4
    assert len(gradable_df) == 1744
    assert list(vis_meta["image_id"].values) == list(gradable_df["image_id"].values)
    assert np.array_equal(y_true, gradable_df["adjudicated_dr_grade"].astype(int).values)

    grade_counts = {
        IDX_TO_GRADE[c]: int(np.sum(y_true == c)) for c in range(5)
    }
    assert grade_counts == {"noDR": 1017, "mildDR": 270, "modDR": 347, "sevDR": 75, "prolDR": 35}

    nodr_count = grade_counts["noDR"]
    core_table2 = {
        "by_family": core_by_fam,
        "t_exclusion_nodr_to_proldr_fpr": {
            "numerator": nodr_to_proldr_num,
            "denominator": nodr_denom,
            "rate": float(nodr_to_proldr_num / nodr_denom),
        },
        "no_dr_prevalence": {
            "numerator": nodr_count,
            "denominator": int(len(y_true)),
            "rate": float(nodr_count / len(y_true)),
        },
        "messidor_distribution_audit": {
            "messidor_csv_total_rows": total_rows,
            "excluded_ungradable_count": int(len(ungradable_df)),
            "excluded_image_ids": [str(x) for x in ungradable_df["image_id"].tolist()],
            "gradable_total": int(len(gradable_df)),
            "gradable_class_counts": grade_counts,
            "thesis_graphs_ipynb_discrepancy": (
                "code/Thesis_Graphs.ipynb Cell 0 hardcoded counts=[1017, 270, 347, 75, 39] "
                "with title N=1,748, erroneously merging the 4 ungradable (NaN) rows into prolDR (35 + 4 = 39)."
            ),
        },
        "reproduction_assertions": {
            "m1_gate0_max_abs_diff_below_1e12": True,
            "thesis_audit_results_1744_csv_exact_match": True,
            "main_tex_table2_exact_match": True,
            "main_tex_table3_ddof0_exact_match": True,
            "t_exclusion_minority_f1_exact_match": True,
        },
    }
    return core_table2, point_by_fam, variants_by_fam


def precompute_bootstrap_ingredients(
    probs_10: np.ndarray, cos_10: np.ndarray, y_true: np.ndarray, tau_native: float
) -> Dict[str, Any]:
    """Precompute per-image sufficient statistics, sparse AUROC matrices, and sorted ECE vectors."""
    n_vars, n_imgs, _ = probs_10.shape
    preds_10 = np.argmax(probs_10, axis=2)  # (10, 1744)
    confs_10 = np.max(probs_10, axis=2)   # (10, 1744)
    corr_10 = (preds_10 == y_true[None, :]).astype(np.float64)

    # 1. Per-image 5x5 confusion matrices flattened to (1744, 25)
    cm_img = np.zeros((n_imgs, 5, 5), dtype=np.float64)
    img_arange = np.arange(n_imgs)
    for v in range(n_vars):
        cm_img[img_arange, y_true, preds_10[v]] += 1.0
    cm_img_flat = cm_img.reshape(n_imgs, 25)

    # 2. Additive per-image vectors
    conf_img = confs_10.sum(axis=0)  # (1744,)
    y_oh = np.zeros((n_imgs, 5), dtype=np.float64)
    y_oh[img_arange, y_true] = 1.0
    brier_img = np.sum((probs_10 - y_oh[None, :, :]) ** 2, axis=(0, 2))  # (1744,)

    logits_10 = cos_10 * tau_native
    lse_10 = logsumexp(logits_10, axis=2)  # (10, 1744)
    y_log_10 = logits_10[np.arange(n_vars)[:, None], img_arange[None, :], y_true[None, :]]
    nll_img = np.sum(lse_10 - y_log_10, axis=0)  # (1744,)

    # Referable DR (grade >= 2 vs 0-1) P(RDR) >= 0.5 per-image counts
    y_ref = (y_true >= 2).astype(int)
    p_ref_10 = probs_10[:, :, 2] + probs_10[:, :, 3] + probs_10[:, :, 4]  # (10, 1744)
    pred_ref_10 = p_ref_10 >= 0.5
    rtp_img = np.sum((y_ref[None, :] == 1) & pred_ref_10, axis=0).astype(np.float64)
    rfn_img = np.sum((y_ref[None, :] == 1) & (~pred_ref_10), axis=0).astype(np.float64)
    rtn_img = np.sum((y_ref[None, :] == 0) & (~pred_ref_10), axis=0).astype(np.float64)
    rfp_img = np.sum((y_ref[None, :] == 0) & pred_ref_10, axis=0).astype(np.float64)

    # Equal-width ECE bin sums per image for M in (10, 15, 20)
    bin_stats: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for m_bins in (10, 15, 20):
        bounds = np.linspace(0.0, 1.0, m_bins + 1)
        b_cnf = np.zeros((n_imgs, m_bins), dtype=np.float64)
        b_crr = np.zeros((n_imgs, m_bins), dtype=np.float64)
        for v in range(n_vars):
            cv = confs_10[v]
            av = corr_10[v]
            for m in range(m_bins):
                if m == m_bins - 1:
                    mask = (cv >= bounds[m]) & (cv <= bounds[m + 1])
                else:
                    mask = (cv >= bounds[m]) & (cv < bounds[m + 1])
                if np.any(mask):
                    b_cnf[mask, m] += cv[mask]
                    b_crr[mask, m] += av[mask]
        bin_stats[m_bins] = (b_cnf, b_crr)

    # 3. Sparse CSR matrices for exact tie-aware AUROC
    p_ref_flat = p_ref_10.reshape(-1)
    img_flat = np.tile(img_arange, n_vars)
    y_ref_flat = np.tile(y_ref, n_vars)
    u_scores, inv = np.unique(p_ref_flat, return_inverse=True)
    n_groups = len(u_scores)
    pos_mask = y_ref_flat == 1
    neg_mask = y_ref_flat == 0
    M_pos = csr_matrix(
        (np.ones(int(np.sum(pos_mask)), dtype=np.float64), (img_flat[pos_mask], inv[pos_mask])),
        shape=(n_imgs, n_groups),
    )
    M_neg = csr_matrix(
        (np.ones(int(np.sum(neg_mask)), dtype=np.float64), (img_flat[neg_mask], inv[neg_mask])),
        shape=(n_imgs, n_groups),
    )

    # 4. Sorted arrays for prefix-sum Adaptive ECE (M=10, 15)
    confs_flat = confs_10.reshape(-1)
    corr_flat = corr_10.reshape(-1)
    order = np.argsort(confs_flat, kind="mergesort")
    c_s = confs_flat[order]
    a_s = corr_flat[order]
    img_s = img_flat[order]

    return {
        "cm_img_flat": cm_img_flat,
        "conf_img": conf_img,
        "brier_img": brier_img,
        "nll_img": nll_img,
        "rtp_img": rtp_img,
        "rfn_img": rfn_img,
        "rtn_img": rtn_img,
        "rfp_img": rfp_img,
        "bin_stats": bin_stats,
        "M_pos": M_pos,
        "M_neg": M_neg,
        "c_s": c_s,
        "a_s": a_s,
        "img_s": img_s,
        "probs_10": probs_10,
        "y_true": y_true,
    }


def run_vectorized_bootstrap_for_family(
    ing: Dict[str, Any], W_all: np.ndarray, boot_idx: np.ndarray
) -> Dict[str, np.ndarray]:
    """Run B=10,000 exact vectorized bootstrap across all metrics for one taxonomy."""
    B, n_imgs = W_all.shape
    N_tot = float(n_imgs * 10)  # 17440.0

    # 1. Confusion-matrix metrics in one BLAS call
    cm = (W_all @ ing["cm_img_flat"]).reshape(B, 5, 5)
    diag = np.diagonal(cm, axis1=1, axis2=2)  # (B, 5)
    acc_arr = diag.sum(axis=1) / N_tot

    row_sums = cm.sum(axis=2)
    col_sums = cm.sum(axis=1)
    recalls = np.divide(diag, row_sums, out=np.zeros_like(diag), where=row_sums > 0)
    aca_arr = recalls.mean(axis=1)

    f1_denom = row_sums + col_sums
    f1s = np.divide(2.0 * diag, f1_denom, out=np.zeros_like(diag), where=f1_denom > 0)
    mf1_arr = f1s.mean(axis=1)

    idx_grid = np.arange(5, dtype=np.float64)
    W_qwk = ((idx_grid[:, None] - idx_grid[None, :]) ** 2) / 16.0
    obs = np.sum(W_qwk[None, :, :] * cm, axis=(1, 2)) / N_tot
    exp_mat = (row_sums[:, :, None] * col_sums[:, None, :]) / N_tot
    exp = np.sum(W_qwk[None, :, :] * exp_mat, axis=(1, 2)) / N_tot
    qwk_arr = 1.0 - np.divide(obs, exp, out=np.zeros_like(obs), where=exp > 0)

    # argmax >= 2 Referable DR
    tp_a = cm[:, 2:, 2:].sum(axis=(1, 2))
    fn_a = cm[:, 2:, :2].sum(axis=(1, 2))
    tn_a = cm[:, :2, :2].sum(axis=(1, 2))
    fp_a = cm[:, :2, 2:].sum(axis=(1, 2))
    arg_sens_arr = tp_a / (tp_a + fn_a)
    arg_spec_arr = tn_a / (tn_a + fp_a)
    arg_f1_arr = (2.0 * tp_a) / (2.0 * tp_a + fp_a + fn_a)

    # 2. Additive per-image metrics
    conf_arr = (W_all @ ing["conf_img"]) / N_tot
    brier_arr = (W_all @ ing["brier_img"]) / N_tot
    nll_arr = (W_all @ ing["nll_img"]) / N_tot

    tp_p = W_all @ ing["rtp_img"]
    fn_p = W_all @ ing["rfn_img"]
    tn_p = W_all @ ing["rtn_img"]
    fp_p = W_all @ ing["rfp_img"]
    p05_sens_arr = tp_p / (tp_p + fn_p)
    p05_spec_arr = tn_p / (tn_p + fp_p)
    p05_f1_arr = (2.0 * tp_p) / (2.0 * tp_p + fp_p + fn_p)

    ece_arrays: Dict[int, np.ndarray] = {}
    for m_bins, (b_cnf, b_crr) in ing["bin_stats"].items():
        tot_cnf = W_all @ b_cnf
        tot_crr = W_all @ b_crr
        ece_arrays[m_bins] = np.sum(np.abs(tot_crr - tot_cnf), axis=1) / N_tot

    # 3. Sparse tie-aware AUROC & Prefix-sum Adaptive ECE in chunks of 200
    auroc_arr = np.empty(B, dtype=np.float64)
    aece10_arr = np.empty(B, dtype=np.float64)
    aece15_arr = np.empty(B, dtype=np.float64)

    M_pos = ing["M_pos"]
    M_neg = ing["M_neg"]
    c_s = ing["c_s"]
    a_s = ing["a_s"]
    img_s = ing["img_s"]

    t10 = np.cumsum([len(x) for x in np.array_split(np.arange(int(N_tot)), 10)]).astype(np.float64)
    t15 = np.cumsum([len(x) for x in np.array_split(np.arange(int(N_tot)), 15)]).astype(np.float64)
    t_both = np.concatenate([t10, t15])  # len 25

    chunk_size = 200
    for s in range(0, B, chunk_size):
        e = min(B, s + chunk_size)
        Wc = W_all[s:e]
        C = e - s

        # Sparse tie-aware AUROC
        pg = Wc @ M_pos
        ng = Wc @ M_neg
        cnb = np.cumsum(ng, axis=1) - ng
        u_stat = np.sum(pg * (cnb + 0.5 * ng), axis=1)
        auroc_arr[s:e] = u_stat / (pg.sum(axis=1) * ng.sum(axis=1))

        # Prefix-sum Adaptive ECE (M=10, 15)
        Ws = Wc[:, img_s]  # (C, 17440)
        Pw = np.cumsum(Ws, axis=1)
        Pwc = np.cumsum(Ws * c_s[None, :], axis=1)
        Pwa = np.cumsum(Ws * a_s[None, :], axis=1)

        for k in range(C):
            Pwk = Pw[k]
            idx = np.searchsorted(Pwk, t_both, side="left")
            prev_idx = idx - 1
            has_prev = prev_idx >= 0
            prev_pw = np.where(has_prev, Pwk[np.maximum(prev_idx, 0)], 0.0)
            prev_c = np.where(has_prev, Pwc[k, np.maximum(prev_idx, 0)], 0.0)
            prev_a = np.where(has_prev, Pwa[k, np.maximum(prev_idx, 0)], 0.0)
            rem = t_both - prev_pw
            pref_c = prev_c + rem * c_s[idx]
            pref_a = prev_a + rem * a_s[idx]

            # M=10
            pc10 = pref_c[:10]
            pa10 = pref_a[:10]
            dc10 = np.diff(np.concatenate([[0.0], pc10]))
            da10 = np.diff(np.concatenate([[0.0], pa10]))
            aece10_arr[s + k] = np.sum(np.abs(da10 - dc10)) / N_tot

            # M=15
            pc15 = pref_c[10:]
            pa15 = pref_a[10:]
            dc15 = np.diff(np.concatenate([[0.0], pc15]))
            da15 = np.diff(np.concatenate([[0.0], pa15]))
            aece15_arr[s + k] = np.sum(np.abs(da15 - dc15)) / N_tot

    return {
        "accuracy": acc_arr,
        "aca": aca_arr,
        "macro_f1": mf1_arr,
        "per_grade_f1_0": f1s[:, 0],
        "per_grade_f1_1": f1s[:, 1],
        "per_grade_f1_2": f1s[:, 2],
        "per_grade_f1_3": f1s[:, 3],
        "per_grade_f1_4": f1s[:, 4],
        "qwk": qwk_arr,
        "referable_dr_auroc": auroc_arr,
        "referable_dr_p05_sensitivity": p05_sens_arr,
        "referable_dr_p05_specificity": p05_spec_arr,
        "referable_dr_p05_f1": p05_f1_arr,
        "referable_dr_argmax_sensitivity": arg_sens_arr,
        "referable_dr_argmax_specificity": arg_spec_arr,
        "referable_dr_argmax_f1": arg_f1_arr,
        "micro_ece_m10": ece_arrays[10],
        "micro_ece_m15": ece_arrays[15],
        "micro_ece_m20": ece_arrays[20],
        "adaptive_ece_m10": aece10_arr,
        "adaptive_ece_m15": aece15_arr,
        "brier_score": brier_arr,
        "nll": nll_arr,
        "mean_confidence": conf_arr,
    }


def spot_check_vectorization(
    ing: Dict[str, Any], boot_series: Dict[str, np.ndarray], boot_idx: np.ndarray
) -> float:
    """Spot-check 5 bootstrap draws against sklearn / np.array_split to assert < 1e-12 error."""
    probs_10 = ing["probs_10"]
    y_true = ing["y_true"]
    y_ref = (y_true >= 2).astype(int)
    max_err = 0.0
    for b in (0, 7, 123, 4567, 9999):
        bi = boot_idx[b]
        p_b = probs_10[:, bi, :].reshape(-1, 5)
        y_b = np.tile(y_true[bi], 10)
        y_ref_b = np.tile(y_ref[bi], 10)
        preds_b = np.argmax(p_b, axis=1)
        confs_b = np.max(p_b, axis=1)
        corr_b = (preds_b == y_b).astype(np.float64)
        p_ref_b = p_b[:, 2] + p_b[:, 3] + p_b[:, 4]

        oracle_acc = float(np.mean(corr_b))
        oracle_qwk = float(cohen_kappa_score(y_b, preds_b, labels=[0, 1, 2, 3, 4], weights="quadratic"))
        oracle_mf1 = float(f1_score(y_b, preds_b, labels=[0, 1, 2, 3, 4], average="macro", zero_division=0))
        oracle_auc = float(roc_auc_score(y_ref_b, p_ref_b))
        oracle_aece10 = calculate_ece_adaptive(confs_b, corr_b, 10)
        oracle_aece15 = calculate_ece_adaptive(confs_b, corr_b, 15)
        oracle_ece15 = calculate_ece_equal_width(confs_b, corr_b, 15)
        oracle_brier = calculate_multiclass_brier(p_b, y_b)

        diffs = [
            abs(oracle_acc - boot_series["accuracy"][b]),
            abs(oracle_qwk - boot_series["qwk"][b]),
            abs(oracle_mf1 - boot_series["macro_f1"][b]),
            abs(oracle_auc - boot_series["referable_dr_auroc"][b]),
            abs(oracle_aece10 - boot_series["adaptive_ece_m10"][b]),
            abs(oracle_aece15 - boot_series["adaptive_ece_m15"][b]),
            abs(oracle_ece15 - boot_series["micro_ece_m15"][b]),
            abs(oracle_brier - boot_series["brier_score"][b]),
        ]
        max_err = max(max_err, max(diffs))
    assert max_err < 1e-12, f"Vectorization spot-check failed: max_err={max_err}"
    return float(max_err)


def run_gee_and_mcnemar(
    probs_by_fam: Dict[str, np.ndarray], y_true: np.ndarray
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Run joint 4-level GEE, 6 pairwise 2-level GEEs, and legacy pooled McNemar reference."""
    n_imgs = len(y_true)
    correct_by_fam: Dict[str, np.ndarray] = {}
    rows = []
    for fam in FAMILIES:
        preds = np.argmax(probs_by_fam[fam], axis=2)  # (10, 1744)
        corr_flat = (preds == y_true[None, :]).astype(int).reshape(-1)
        correct_by_fam[fam] = corr_flat
        img_flat = np.tile(np.arange(n_imgs, dtype=int), 10)
        rows.append(
            pd.DataFrame(
                {
                    "img_idx": img_flat,
                    "taxonomy": fam,
                    "correct": corr_flat,
                }
            )
        )

    df_all = pd.concat(rows, ignore_index=True)
    df_all["taxonomy"] = pd.Categorical(df_all["taxonomy"], categories=FAMILIES, ordered=False)
    # Sort by cluster for clean statsmodels group alignment
    df_all = df_all.sort_values(["img_idx", "taxonomy"], kind="mergesort").reset_index(drop=True)

    t0 = time.perf_counter()
    gee_model = smf.gee(
        "correct ~ C(taxonomy, Treatment(reference='T_label'))",
        groups="img_idx",
        data=df_all,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    )
    gee_res = gee_model.fit()
    gee_elapsed = time.perf_counter() - t0
    assert bool(gee_res.converged) is True
    assert int(gee_res.nobs) == 69760
    assert int(gee_model.num_group) == 1744

    # Extract parameters
    param_names = list(gee_res.params.index)
    params_out = {}
    for pname in param_names:
        coef = float(gee_res.params[pname])
        se = float(gee_res.bse[pname])
        z_val = float(gee_res.tvalues[pname])
        p_val = float(gee_res.pvalues[pname])
        ci_lo = coef - 1.959963984540054 * se
        ci_hi = coef + 1.959963984540054 * se
        params_out[pname] = {
            "coef": coef,
            "robust_se": se,
            "z_stat": z_val,
            "p_value": p_val,
            "odds_ratio": float(math.exp(coef)),
            "or_ci95_low": float(math.exp(ci_lo)),
            "or_ci95_high": float(math.exp(ci_hi)),
        }

    # All 6 pairwise Wald contrasts (f1 - f2 log-odds ratio, i.e., OR > 1 means f1 has higher accuracy odds)
    # Map family -> index in parameter vector (T_label is reference -> 0 for all treatment dummies)
    fam_to_pidx: Dict[str, int | None] = {"T_label": None}
    for i, pname in enumerate(param_names):
        for fam in FAMILIES[1:]:
            if f"[T.{fam}]" in pname:
                fam_to_pidx[fam] = i

    pairwise_wald: Dict[str, Any] = {}
    for f1, f2 in TAXONOMY_PAIRS:
        L = np.zeros(len(param_names), dtype=np.float64)
        if fam_to_pidx[f1] is not None:
            L[fam_to_pidx[f1]] += 1.0
        if fam_to_pidx[f2] is not None:
            L[fam_to_pidx[f2]] -= 1.0
        wt = gee_res.wald_test(L, scalar=True)
        delta_lor = float(L @ gee_res.params.values)
        se_delta = float(np.sqrt(L @ gee_res.cov_params().values @ L))
        z_delta = float(delta_lor / se_delta)
        chi2_stat = float(wt.statistic)
        p_val = float(wt.pvalue)
        assert abs(chi2_stat - z_delta ** 2) < 1e-8
        pairwise_wald[f"{f1}_vs_{f2}"] = {
            "contrast": f"{f1} - {f2}",
            "delta_log_or": delta_lor,
            "robust_se": se_delta,
            "z_stat": z_delta,
            "wald_chi2": chi2_stat,
            "p_value": p_val,
            "odds_ratio_f1_over_f2": float(math.exp(delta_lor)),
            "or_ci95_low": float(math.exp(delta_lor - 1.959963984540054 * se_delta)),
            "or_ci95_high": float(math.exp(delta_lor + 1.959963984540054 * se_delta)),
        }

    gee_joint_out = {
        "formula": "correct ~ C(taxonomy, Treatment(reference='T_label'))",
        "groups": "img_idx",
        "family": "Binomial",
        "cov_struct": "Exchangeable",
        "cov_type": str(gee_res.cov_type),
        "converged": bool(gee_res.converged),
        "n_obs": int(gee_res.nobs),
        "n_clusters": int(gee_model.num_group),
        "cluster_size": 40,
        "exchangeable_corr_rho": float(gee_res.cov_struct.dep_params),
        "elapsed_seconds": float(gee_elapsed),
        "parameters": params_out,
        "pairwise_wald_contrasts": pairwise_wald,
    }

    # Pairwise 2-level exchangeable GEE sensitivity confirmation of the joint 4-level model
    gee_pairwise_out: Dict[str, Any] = {}
    for f1, f2 in TAXONOMY_PAIRS:
        df_pair = df_all[df_all["taxonomy"].isin([f1, f2])].copy()
        df_pair["taxonomy"] = pd.Categorical(df_pair["taxonomy"], categories=[f2, f1], ordered=False)
        m2 = smf.gee(
            f"correct ~ C(taxonomy, Treatment(reference='{f2}'))",
            groups="img_idx",
            data=df_pair,
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Exchangeable(),
        )
        r2 = m2.fit()
        assert bool(r2.converged) is True
        pname = [p for p in r2.params.index if f"[T.{f1}]" in p][0]
        coef = float(r2.params[pname])
        se = float(r2.bse[pname])
        z_val = float(r2.tvalues[pname])
        p_val = float(r2.pvalues[pname])
        gee_pairwise_out[f"{f1}_vs_{f2}"] = {
            "reference": f2,
            "target": f1,
            "converged": bool(r2.converged),
            "n_obs": int(r2.nobs),
            "n_clusters": int(m2.num_group),
            "exchangeable_corr_rho": float(r2.cov_struct.dep_params),
            "delta_log_or": coef,
            "robust_se": se,
            "z_stat": z_val,
            "wald_chi2": float(z_val ** 2),
            "p_value": p_val,
            "odds_ratio_f1_over_f2": float(math.exp(coef)),
            "or_ci95_low": float(math.exp(coef - 1.959963984540054 * se)),
            "or_ci95_high": float(math.exp(coef + 1.959963984540054 * se)),
        }

    # Legacy pooled McNemar (explicitly flagged invalid due to 10x clustering)
    mcnemar_out: Dict[str, Any] = {
        "valid": False,
        "invalidity_reason": (
            "Pooled McNemar treats N=17,440 prompt evaluations as independent across 10 prompts per image "
            "(10x repeated-measures clustering), severely inflating chi2 statistics "
            "(e.g., T_label vs T_expert chi2=527.85 vs cluster-robust GEE Wald chi2=209.00)."
        ),
        "pairs": {},
    }
    for f1, f2 in TAXONOMY_PAIRS:
        c1 = correct_by_fam[f1]
        c2 = correct_by_fam[f2]
        n11 = int(np.sum((c1 == 1) & (c2 == 1)))
        n10 = int(np.sum((c1 == 1) & (c2 == 0)))
        n01 = int(np.sum((c1 == 0) & (c2 == 1)))
        n00 = int(np.sum((c1 == 0) & (c2 == 0)))
        res_mc = mcnemar([[n11, n10], [n01, n00]], exact=False, correction=True)
        mcnemar_out["pairs"][f"{f1}_vs_{f2}"] = {
            "n11_both_correct": n11,
            "n10_f1_only": n10,
            "n01_f2_only": n01,
            "n00_both_wrong": n00,
            "chi2_corrected": float(res_mc.statistic),
            "p_value": float(res_mc.pvalue),
        }

    return gee_joint_out, gee_pairwise_out, mcnemar_out


def run_calibration_robustness(
    probs_by_fam: Dict[str, np.ndarray],
    cos_by_fam: Dict[str, np.ndarray],
    point_by_fam: Dict[str, Dict[str, Any]],
    ci_by_fam: Dict[str, Dict[str, Dict[str, float]]],
    y_true: np.ndarray,
    tau_native: float,
) -> Dict[str, Any]:
    """Build calibration_robustness including native CIs and 5-fold image-level CV temperature scaling."""
    skf = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=CV_SEED)
    splits = list(skf.split(np.arange(len(y_true)), y_true))

    by_family: Dict[str, Any] = {}
    for fam in FAMILIES:
        pt = point_by_fam[fam]
        ci_f = ci_by_fam[fam]
        cos_10 = cos_by_fam[fam]  # (10, 1744, 5)
        n_vars, n_imgs, _ = cos_10.shape

        oof_probs_10 = np.empty_like(cos_10, dtype=np.float64)
        covered_imgs = set()
        folds_out: List[Dict[str, Any]] = []
        tau_stars: List[float] = []
        t_ratios: List[float] = []

        for fold_idx, (tr_idx, te_idx) in enumerate(splits):
            assert len(set(tr_idx).intersection(set(te_idx))) == 0
            covered_imgs.update(te_idx.tolist())

            tr_cos = cos_10[:, tr_idx, :].reshape(-1, 5)
            tr_y = np.tile(y_true[tr_idx], n_vars)

            def nll_obj(tau_val: float) -> float:
                return calculate_nll_from_logits(tr_cos * tau_val, tr_y)

            def nll_grad_hess(tau_val: float) -> Tuple[float, float]:
                z = tr_cos * tau_val
                z_max = np.max(z, axis=1, keepdims=True)
                exp_z = np.exp(z - z_max)
                p = exp_z / np.sum(exp_z, axis=1, keepdims=True)
                es = np.sum(p * tr_cos, axis=1)
                es2 = np.sum(p * (tr_cos ** 2), axis=1)
                y_s = tr_cos[np.arange(len(tr_y)), tr_y]
                grad = float(np.mean(es - y_s))
                hess = float(np.mean(es2 - es ** 2))
                return grad, hess

            opt = minimize_scalar(
                nll_obj, bounds=(0.0, 200.0), method="bounded", options={"xatol": 1e-10}
            )
            assert opt.success
            tau_star = float(opt.x)
            t_ratio = float(tau_native / tau_star)

            grad_zero, _ = nll_grad_hess(0.0)
            grad_opt, hess_opt = nll_grad_hess(tau_star)
            assert abs(grad_opt) < 1e-6, f"Gradient check failed for {fam} fold {fold_idx}: {grad_opt}"
            assert hess_opt > 0.0, f"Hessian check failed for {fam} fold {fold_idx}: {hess_opt}"

            train_nll_native = nll_obj(tau_native)
            train_nll_opt = float(opt.fun)

            # Evaluate on held-out test fold
            te_logits = cos_10[:, te_idx, :] * tau_star
            te_max = np.max(te_logits, axis=2, keepdims=True)
            te_exp = np.exp(te_logits - te_max)
            oof_probs_10[:, te_idx, :] = te_exp / np.sum(te_exp, axis=2, keepdims=True)

            tau_stars.append(tau_star)
            t_ratios.append(t_ratio)
            folds_out.append(
                {
                    "fold": int(fold_idx),
                    "n_train_images": int(len(tr_idx)),
                    "n_test_images": int(len(te_idx)),
                    "tau_star": tau_star,
                    "T_ratio": t_ratio,
                    "train_nll_native": float(train_nll_native),
                    "train_nll_opt": float(train_nll_opt),
                    "grad_at_zero": float(grad_zero),
                    "grad_at_opt": float(grad_opt),
                    "hess_at_opt": float(hess_opt),
                }
            )

        assert len(covered_imgs) == n_imgs

        oof_probs = oof_probs_10.reshape(-1, 5)
        y_pooled = np.tile(y_true, n_vars)
        oof_preds = np.argmax(oof_probs, axis=1)
        oof_confs = np.max(oof_probs, axis=1)
        oof_corr = (oof_preds == y_pooled).astype(np.float64)

        oof_acc = float(np.mean(oof_corr))
        oof_conf = float(np.mean(oof_confs))
        oof_brier = calculate_multiclass_brier(oof_probs, y_pooled)
        oof_ece10 = calculate_ece_equal_width(oof_confs, oof_corr, 10)
        oof_ece15 = calculate_ece_equal_width(oof_confs, oof_corr, 15)
        oof_ece20 = calculate_ece_equal_width(oof_confs, oof_corr, 20)
        oof_aece10 = calculate_ece_adaptive(oof_confs, oof_corr, 10)
        oof_aece15 = calculate_ece_adaptive(oof_confs, oof_corr, 15)
        # Exact OOF NLL using log-probs (avoid log(0))
        oof_nll = float(-np.mean(np.log(np.clip(oof_probs[np.arange(len(y_pooled)), y_pooled], 1e-300, 1.0))))

        # Temperature scaling is strictly monotonic on logits so argmax accuracy is invariant
        assert abs(oof_acc - pt["accuracy"]) < 1e-12

        by_family[fam] = {
            "native_tau": {
                "tau": float(tau_native),
                "n_predictions": pt["n_predictions"],
                "n_unique_confidences": pt["n_unique_confidences"],
                "accuracy": pt["accuracy"],
                "mean_confidence": pt["mean_confidence"],
                "nll": pt["nll"],
                "brier_score": pt["brier_score"],
                "ece_equal_width": {
                    "M10": pt["micro_ece_m10"],
                    "M15": pt["micro_ece_m15"],
                    "M20": pt["micro_ece_m20"],
                },
                "ece_adaptive": {
                    "M10": pt["adaptive_ece_m10"],
                    "M15": pt["adaptive_ece_m15"],
                },
                "ci95": {
                    "ece_m10": ci_f["micro_ece_m10"],
                    "ece_m15": ci_f["micro_ece_m15"],
                    "ece_m20": ci_f["micro_ece_m20"],
                    "adaptive_ece_m10": ci_f["adaptive_ece_m10"],
                    "adaptive_ece_m15": ci_f["adaptive_ece_m15"],
                    "brier_score": ci_f["brier_score"],
                    "nll": ci_f["nll"],
                    "mean_confidence": ci_f["mean_confidence"],
                },
            },
            "cv_temperature_scaling": {
                "tau_star_mean": float(np.mean(tau_stars)),
                "tau_star_sd_ddof1": float(np.std(tau_stars, ddof=1)),
                "T_ratio_mean": float(np.mean(t_ratios)),
                "T_ratio_sd_ddof1": float(np.std(t_ratios, ddof=1)),
                "folds": folds_out,
                "oof_pooled": {
                    "n_predictions": int(len(y_pooled)),
                    "accuracy": oof_acc,
                    "mean_confidence": oof_conf,
                    "nll": oof_nll,
                    "brier_score": oof_brier,
                    "ece_equal_width": {
                        "M10": oof_ece10,
                        "M15": oof_ece15,
                        "M20": oof_ece20,
                    },
                    "ece_adaptive": {
                        "M10": oof_aece10,
                        "M15": oof_aece15,
                    },
                },
                "deltas_oof_minus_native": {
                    "accuracy_pp": float((oof_acc - pt["accuracy"]) * 100.0),
                    "mean_confidence_pp": float((oof_conf - pt["mean_confidence"]) * 100.0),
                    "nll": float(oof_nll - pt["nll"]),
                    "brier_score": float(oof_brier - pt["brier_score"]),
                    "ece_m10_pp": float((oof_ece10 - pt["micro_ece_m10"]) * 100.0),
                    "ece_adaptive_m10_pp": float((oof_aece10 - pt["adaptive_ece_m10"]) * 100.0),
                },
            },
        }

    return {
        "ece_binning_convention": {
            "equal_width": "np.linspace(0, 1, M+1); [lo, hi) for i < M-1, [lo, hi] for i == M-1",
            "adaptive": "np.array_split(np.argsort(confidences, kind='mergesort'), M)",
            "brier_range": "[0, 2] (multi-class sum of squared probability errors over 5 classes)",
        },
        "cv_protocol": {
            "splitter": f"StratifiedKFold(n_splits={CV_N_SPLITS}, shuffle=True, random_state={CV_SEED})",
            "stratify_on": "adjudicated_dr_grade (1744 unique images)",
            "prompt_grouping": "all 10 prompt variants per image kept in same fold (zero image leakage)",
            "objective": "mean NLL (cross-entropy) over logits = tau * cos_sim, bounded [0.0, 200.0]",
            "native_tau": float(tau_native),
        },
        "by_family": by_family,
    }


def build_per_prompt_variance(
    variants_by_fam: Dict[str, List[Dict[str, Any]]],
    txt_meta: pd.DataFrame,
) -> Dict[str, Any]:
    """Assemble 40-prompt variance breakdown and T_layman outlier collapse diagnosis."""
    by_family: Dict[str, Any] = {}
    summary_metrics = [
        "accuracy",
        "aca",
        "macro_f1",
        "qwk",
        "ece_m10",
        "ece_m15",
        "adaptive_ece_m10",
        "brier_score",
        "mean_confidence",
        "dominant_predicted_share",
    ]

    for fam in FAMILIES:
        var_rows = variants_by_fam[fam]
        enriched_rows: List[Dict[str, Any]] = []
        for row in var_rows:
            v_num = row["variation_number"]
            sub = txt_meta[(txt_meta["Family"] == fam) & (txt_meta["Variation_Number"] == v_num)]
            prompts_by_grade = {
                str(r["Grade"]): str(r["Prompt_Text"])
                for _, r in sub.sort_values("numerical_grade").iterrows()
            }
            enriched_rows.append(
                {
                    "variation_number": v_num,
                    "prompts_by_grade": prompts_by_grade,
                    "accuracy": row["accuracy"],
                    "aca": row["aca"],
                    "macro_f1": row["macro_f1"],
                    "qwk": row["qwk"],
                    "referable_dr_auroc": row["referable_dr_auroc"],
                    "ece_m10": row["ece_m10"],
                    "ece_m15": row["ece_m15"],
                    "adaptive_ece_m10": row["adaptive_ece_m10"],
                    "brier_score": row["brier_score"],
                    "nll": row["nll"],
                    "mean_confidence": row["mean_confidence"],
                    "dominant_predicted_class_idx": row["dominant_predicted_class_idx"],
                    "dominant_predicted_class_name": row["dominant_predicted_class_name"],
                    "dominant_predicted_share": row["dominant_predicted_share"],
                    "predicted_class_counts": row["predicted_class_counts"],
                }
            )

        macro_summary = {
            m: summarize_macro_across_variants([r[m] for r in enriched_rows])
            for m in summary_metrics
        }
        by_family[fam] = {
            "macro_summary": macro_summary,
            "variants": enriched_rows,
        }

    # Concise T_layman variance diagnosis computed directly from enriched_rows
    lay_vars = {r["variation_number"]: r for r in by_family["T_layman"]["variants"]}
    ece_vals = [r["ece_m10"] for r in by_family["T_layman"]["variants"]]
    collapse_variants = {}
    for v_num in (3, 4, 8, 10, 5):
        r = lay_vars[v_num]
        collapse_variants[f"v{v_num}"] = {
            "variation_number": v_num,
            "accuracy": r["accuracy"],
            "ece_m10": r["ece_m10"],
            "mean_confidence": r["mean_confidence"],
            "dominant_class": r["dominant_predicted_class_name"],
            "dominant_share": r["dominant_predicted_share"],
            "dominant_count": r["predicted_class_counts"][str(r["dominant_predicted_class_idx"])],
        }
    moderate_variants = {}
    for v_num in (1, 9, 2, 6, 7):
        r = lay_vars[v_num]
        moderate_variants[f"v{v_num}"] = {
            "variation_number": v_num,
            "accuracy": r["accuracy"],
            "ece_m10": r["ece_m10"],
            "mean_confidence": r["mean_confidence"],
            "dominant_class": r["dominant_predicted_class_name"],
            "dominant_share": r["dominant_predicted_share"],
        }

    return {
        "n_variants_total": 40,
        "n_images_per_variant": 1744,
        "sd_note": "sd_ddof0 matches main.tex Table 3 (population SD across K=10 variants); sd_ddof1 is sample SD.",
        "by_family": by_family,
        "t_layman_variance_diagnosis": {
            "pattern": "non_uniform_outlier_driven_single_class_collapse",
            "ece_m10_min": float(min(ece_vals)),
            "ece_m10_max": float(max(ece_vals)),
            "ece_m10_range": float(max(ece_vals) - min(ece_vals)),
            "extreme_collapse_variants": collapse_variants,
            "moderate_and_lower_ece_variants": moderate_variants,
            "summary_finding": (
                "T_layman macro-ECE SD (+/-23.51 pp, ddof=0) is driven by severe single-class logit collapses on "
                "v=3,4 (96.3%-96.6% mildDR, ECE 69.48%-74.98%), v=8,10 (82.9%-88.9% sevDR, ECE 59.02%-61.07%), "
                "and v=5 (99.6% noDR, ECE 40.53% at 98.14% confidence), versus lower-ECE variants v=1,9 (13.31%-13.71%)."
            ),
        },
    }


def compute_latent_geometry_check(
    vis_u: np.ndarray, txt_u: np.ndarray, txt_meta: pd.DataFrame
) -> Dict[str, Any]:
    """Verify Section 4.3 latent geometry and main.tex:304 cosine similarity vs distance labels."""
    all_embeds = np.vstack([vis_u, txt_u])
    labels = np.array([0] * len(vis_u) + [1] * len(txt_u), dtype=int)
    sil_cosine = float(silhouette_score(all_embeds, labels, metric="cosine"))
    vis_txt_dist = float(cosine_distances(vis_u, txt_u).mean())
    vis_txt_sim = float((vis_u @ txt_u.T).mean())

    fam_embs = {}
    for fam in FAMILIES:
        idx = txt_meta.index[txt_meta["Family"] == fam].values
        assert len(idx) == 50
        fam_embs[fam] = txt_u[idx]

    intra_offdiag_sim: Dict[str, float] = {}
    intra_offdiag_dist: Dict[str, float] = {}
    intra_incl_diag_sim: Dict[str, float] = {}
    intra_incl_diag_dist: Dict[str, float] = {}

    for fam in FAMILIES:
        e = fam_embs[fam]
        sim_mat = e @ e.T
        triu_idx = np.triu_indices(len(e), k=1)
        off_sim = float(np.mean(sim_mat[triu_idx]))
        full_sim = float(np.mean(sim_mat))
        intra_offdiag_sim[fam] = off_sim
        intra_offdiag_dist[fam] = float(1.0 - off_sim)
        intra_incl_diag_sim[fam] = full_sim
        intra_incl_diag_dist[fam] = float(1.0 - full_sim)

    cross_sim: Dict[str, float] = {}
    cross_dist: Dict[str, float] = {}
    for f1, f2 in TAXONOMY_PAIRS:
        sim_val = float(np.mean(fam_embs[f1] @ fam_embs[f2].T))
        key = f"{f1}__{f2}"
        cross_sim[key] = sim_val
        cross_dist[key] = float(1.0 - sim_val)

    # Assert exact match to main.tex L302 & L304 figures
    assert f"{sil_cosine:.4f}" == "0.6393"
    assert f"{vis_txt_dist:.4f}" == "0.8801"
    assert f"{intra_offdiag_sim['T_label']:.4f}" == "0.6482"
    assert f"{intra_offdiag_sim['T_layman']:.4f}" == "0.6879"
    assert f"{cross_sim['T_label__T_layman']:.4f}" == "0.3434"

    return {
        "vision_vs_text_silhouette_cosine": sil_cosine,
        "vision_vs_text_mean_cosine_distance": vis_txt_dist,
        "vision_vs_text_mean_cosine_similarity": vis_txt_sim,
        "intra_family_offdiag_cosine_similarity": intra_offdiag_sim,
        "intra_family_offdiag_cosine_distance": intra_offdiag_dist,
        "intra_family_incl_diag_cosine_similarity": intra_incl_diag_sim,
        "intra_family_incl_diag_cosine_distance": intra_incl_diag_dist,
        "cross_family_cosine_similarity": cross_sim,
        "cross_family_cosine_distance": cross_dist,
        "main_tex_L304_note": (
            "Values 0.6482 (T_label intra off-diag), 0.6879 (T_layman intra off-diag), and 0.3434 "
            "(T_label vs T_layman cross-family) reported in main.tex:304 are pairwise cosine similarities "
            "(not cosine distances). Corresponding cosine distances (1 - cos) are 0.3518, 0.3121, and 0.6566."
        ),
    }


def main() -> None:
    t_start = time.perf_counter()
    print("[M2] Loading cached Week 4 tensors, metadata, MESSIDOR-2 CSV, and M1 ledger...")

    vis_raw = np.load(WEEK4_DIR / "cached_vision_embeddings_1744.npy")
    txt_raw = np.load(WEEK4_DIR / "cached_text_embeddings_custom.npy")
    tau_native = float(np.load(WEEK4_DIR / "flair_logit_scale.npy").item())

    # L2-normalize in native float32 before casting to float64 for exact M1 Gate 0 & 3_inference_and_audit.py parity
    vis_u = (vis_raw / np.linalg.norm(vis_raw, axis=1, keepdims=True)).astype(np.float64)
    txt_u = (txt_raw / np.linalg.norm(txt_raw, axis=1, keepdims=True)).astype(np.float64)

    vis_meta = pd.read_csv(WEEK4_DIR / "cached_vision_metadata_1744.csv")
    txt_meta = pd.read_csv(WEEK4_DIR / "cached_text_metadata_custom.csv")
    txt_meta["numerical_grade"] = txt_meta["Grade"].map(GRADE_TO_IDX)
    y_true = vis_meta["adjudicated_dr_grade"].astype(int).values

    messidor_df = pd.read_csv(MESSIDOR_CSV)
    thesis_csv = pd.read_csv(WEEK4_DIR / "thesis_audit_results_1744.csv")

    with open(ARTIFACTS_DIR / "m1_verified_metrics.json", "r", encoding="utf-8") as f:
        m1_data = json.load(f)

    # Precompute cos_sim and softmax probabilities for all (4 families x 10 variants)
    cos_by_fam: Dict[str, np.ndarray] = {}
    probs_by_fam: Dict[str, np.ndarray] = {}
    for fam in FAMILIES:
        cos_list = []
        probs_list = []
        for v_num in range(1, 11):
            sub = txt_meta[(txt_meta["Family"] == fam) & (txt_meta["Variation_Number"] == v_num)]
            sub = sub.sort_values("numerical_grade")
            assert len(sub) == 5
            v_emb = txt_u[sub.index.values]  # (5, 512)
            cos_mat = vis_u @ v_emb.T  # (1744, 5)
            logits = cos_mat * tau_native
            l_max = np.max(logits, axis=1, keepdims=True)
            exp_l = np.exp(logits - l_max)
            probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
            cos_list.append(cos_mat)
            probs_list.append(probs)
        cos_by_fam[fam] = np.stack(cos_list, axis=0)    # (10, 1744, 5)
        probs_by_fam[fam] = np.stack(probs_list, axis=0)  # (10, 1744, 5)

    # 1. Deterministic Reconstruction & Gate Assertions
    print("[M2] Step 1: Deterministic reconstruction & Table 2-3 / M1 Gate 0 assertions...")
    core_table2, point_by_fam, variants_by_fam = verify_and_build_core_table2(
        probs_by_fam, cos_by_fam, y_true, tau_native, vis_meta, messidor_df, thesis_csv, m1_data
    )

    # 2. Cluster-Robust Bootstrap (B=10,000, seed=42)
    print(f"[M2] Step 2: Running B={BOOTSTRAP_B} image-level cluster bootstrap (seed={BOOTSTRAP_SEED})...")
    t_boot_start = time.perf_counter()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot_idx = rng.integers(0, len(y_true), size=(BOOTSTRAP_B, len(y_true)))
    W_all = np.zeros((BOOTSTRAP_B, len(y_true)), dtype=np.float64)
    for b in range(BOOTSTRAP_B):
        W_all[b] = np.bincount(boot_idx[b], minlength=len(y_true))

    boot_series_by_fam: Dict[str, Dict[str, np.ndarray]] = {}
    ci_by_fam: Dict[str, Dict[str, Dict[str, float]]] = {}
    max_spot_err = 0.0

    for fam in FAMILIES:
        t_fam = time.perf_counter()
        ing = precompute_bootstrap_ingredients(probs_by_fam[fam], cos_by_fam[fam], y_true, tau_native)
        b_ser = run_vectorized_bootstrap_for_family(ing, W_all, boot_idx)
        spot_err = spot_check_vectorization(ing, b_ser, boot_idx)
        max_spot_err = max(max_spot_err, spot_err)
        boot_series_by_fam[fam] = b_ser

        pt = point_by_fam[fam]
        fam_ci: Dict[str, Dict[str, float]] = {}
        for m_name, arr in b_ser.items():
            lo, hi = np.percentile(arr, [2.5, 97.5])
            fam_ci[m_name] = {
                "point": float(pt[m_name]),
                "ci95_low": float(lo),
                "ci95_high": float(hi),
                "boot_se": float(np.std(arr, ddof=1)),
            }
        ci_by_fam[fam] = fam_ci
        print(f"  - {fam}: completed in {time.perf_counter() - t_fam:.2f}s (spot-check max err={spot_err:.2e})")

    boot_elapsed = time.perf_counter() - t_boot_start
    print(f"[M2] Cluster bootstrap total: {boot_elapsed:.2f}s (overall spot-check max err={max_spot_err:.2e})")

    # Assert exact parity with M1 control_c_results.json cluster_bootstrap_95ci if present
    ctrl_c_path = ARTIFACTS_DIR / "control_c_results.json"
    if ctrl_c_path.exists():
        with open(ctrl_c_path, "r", encoding="utf-8") as f:
            ctrl_c = json.load(f)
        m1_raw_fams = ctrl_c["protocols"]["raw_single_string"]["families"]
        for fam in FAMILIES:
            m1_ci = m1_raw_fams[fam]["cluster_bootstrap_95ci"]
            m2_ci = ci_by_fam[fam]
            key_map = {
                "accuracy": "accuracy",
                "aca": "aca",
                "macro_f1": "macro_f1",
                "qwk": "qwk",
                "mean_confidence": "mean_confidence",
                "micro_ece_m10": "micro_ece_m10",
                "referable_dr_auroc": "referable_dr_auroc",
                "referable_dr_sensitivity": "referable_dr_p05_sensitivity",
                "referable_dr_specificity": "referable_dr_p05_specificity",
                "referable_dr_f1": "referable_dr_p05_f1",
            }
            for k1, k2 in key_map.items():
                assert abs(m1_ci[k1]["ci95_low"] - m2_ci[k2]["ci95_low"]) < 1e-12
                assert abs(m1_ci[k1]["ci95_high"] - m2_ci[k2]["ci95_high"]) < 1e-12
                assert abs(m1_ci[k1]["boot_se"] - m2_ci[k2]["boot_se"]) < 1e-12
        print("[M2] Verified exact (<1e-12) CI parity against M1 control_c_results.json.")

    # Paired bootstrap deltas across all 6 pairs
    delta_metrics = [
        "accuracy",
        "aca",
        "macro_f1",
        "qwk",
        "referable_dr_auroc",
        "micro_ece_m10",
        "adaptive_ece_m10",
        "brier_score",
        "mean_confidence",
    ]
    paired_deltas: Dict[str, Any] = {}
    for f1, f2 in TAXONOMY_PAIRS:
        comp: Dict[str, Any] = {}
        for m_name in delta_metrics:
            pt_delta = float(point_by_fam[f1][m_name] - point_by_fam[f2][m_name])
            diff_arr = boot_series_by_fam[f1][m_name] - boot_series_by_fam[f2][m_name]
            lo, hi = np.percentile(diff_arr, [2.5, 97.5])
            c_le = int(np.sum(diff_arr <= 0.0))
            c_ge = int(np.sum(diff_arr >= 0.0))
            p_raw = float(min(1.0, 2.0 * min(c_le / BOOTSTRAP_B, c_ge / BOOTSTRAP_B)))
            p_plus1 = float(min(1.0, 2.0 * (min(c_le, c_ge) + 1.0) / (BOOTSTRAP_B + 1.0)))
            comp[m_name] = {
                "delta": pt_delta,
                "ci95_low": float(lo),
                "ci95_high": float(hi),
                "boot_se": float(np.std(diff_arr, ddof=1)),
                "boot_p_raw": p_raw,
                "boot_p_plus1": p_plus1,
            }
        paired_deltas[f"{f1}_vs_{f2}"] = comp

    # 3. GEE Logistic Regression & Legacy McNemar
    print("[M2] Step 3: Fitting image-clustered GEE logistic regressions & legacy McNemar...")
    gee_joint, gee_pairwise, legacy_mcnemar = run_gee_and_mcnemar(probs_by_fam, y_true)
    print(f"  - Joint 4-level GEE converged={gee_joint['converged']} in {gee_joint['elapsed_seconds']:.3f}s (rho={gee_joint['exchangeable_corr_rho']:.4f})")

    cluster_significance = {
        "bootstrap_config": {
            "B": BOOTSTRAP_B,
            "seed": BOOTSTRAP_SEED,
            "rng": "np.random.default_rng(42)",
            "resampling_unit": "image_id (1,744 clusters, keeping all 10 prompt variants per image together)",
            "ci_method": "percentile [2.5, 97.5]",
            "p_value_note": "boot_p_raw matches M1 control_c; boot_p_plus1 applies finite-sample Phipson-Smyth (min_count+1)/(B+1) floor (~2e-4 at B=10,000).",
            "runtime_seconds": float(boot_elapsed),
            "vectorization_max_abs_err": float(max_spot_err),
        },
        "per_taxonomy_bootstrap": ci_by_fam,
        "paired_deltas": paired_deltas,
        "gee_joint_4level": gee_joint,
        "gee_pairwise_2level": gee_pairwise,
        "legacy_pooled_mcnemar_invalid": legacy_mcnemar,
    }

    # 4. Standard MESSIDOR-2 Clinical Benchmarks
    print("[M2] Step 4: Assembling standard MESSIDOR-2 clinical benchmarks...")
    clinical_benchmarks_by_fam: Dict[str, Any] = {}
    for fam in FAMILIES:
        c = ci_by_fam[fam]
        clinical_benchmarks_by_fam[fam] = {
            "accuracy": c["accuracy"],
            "aca": c["aca"],
            "macro_f1": c["macro_f1"],
            "qwk": c["qwk"],
            "per_grade_f1": {
                IDX_TO_GRADE[g]: c[f"per_grade_f1_{g}"] for g in range(5)
            },
            "referable_dr": {
                "definition": "Adjudicated DR grade >= 2 vs 0-1; P(RDR) = P(modDR) + P(sevDR) + P(prolDR)",
                "auroc": c["referable_dr_auroc"],
                "threshold_p05": {
                    "rule": "P(RDR) >= 0.5 (primary, matching M1)",
                    "sensitivity": c["referable_dr_p05_sensitivity"],
                    "specificity": c["referable_dr_p05_specificity"],
                    "f1": c["referable_dr_p05_f1"],
                },
                "threshold_argmax": {
                    "rule": "argmax(probs) >= 2 (secondary sensitivity check)",
                    "sensitivity": c["referable_dr_argmax_sensitivity"],
                    "specificity": c["referable_dr_argmax_specificity"],
                    "f1": c["referable_dr_argmax_f1"],
                },
            },
        }
    clinical_benchmarks = {
        "positive_prevalence_referable_dr": float(np.mean(y_true >= 2)),
        "n_referable_dr_images": int(np.sum(y_true >= 2)),
        "n_non_referable_images": int(np.sum(y_true < 2)),
        "by_family": clinical_benchmarks_by_fam,
    }

    # 5. Calibration Robustness & 5-Fold Stratified CV Temperature Scaling
    print("[M2] Step 5: Running calibration robustness & 5-fold stratified image-level CV temperature scaling...")
    calibration_robustness = run_calibration_robustness(
        probs_by_fam, cos_by_fam, point_by_fam, ci_by_fam, y_true, tau_native
    )

    # 6. Complete 40-Prompt Variance Breakdown
    print("[M2] Step 6: Building 40-prompt variance breakdown & T_layman diagnosis...")
    per_prompt_variance = build_per_prompt_variance(variants_by_fam, txt_meta)
    assert abs(per_prompt_variance["by_family"]["T_layman"]["macro_summary"]["ece_m10"]["sd_ddof0"] * 100.0 - 23.51) < 0.01

    # 7. Latent Geometry Check
    print("[M2] Step 7: Computing Section 4.3 latent geometry check...")
    latent_geometry_check = compute_latent_geometry_check(vis_u, txt_u, txt_meta)

    # 8. Save M2 and Consolidated Verified Metrics JSONs
    print("[M2] Step 8: Consolidating artifacts/m2_verified_metrics.json & artifacts/verified_metrics.json...")
    now_utc = datetime.now(timezone.utc).isoformat()
    m2_metadata = {
        "generated_utc": now_utc,
        "milestone": "M2",
        "script": "code/revision_experiments/stats_reanalysis.py",
        "n_images": int(len(y_true)),
        "n_variants_per_family": 10,
        "n_predictions_per_family": int(len(y_true) * 10),
        "logit_scale_native": float(tau_native),
        "bootstrap_b": BOOTSTRAP_B,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "cv_n_splits": CV_N_SPLITS,
        "cv_seed": CV_SEED,
        "total_runtime_seconds": float(time.perf_counter() - t_start),
        "artifacts": {
            "m1_verified_metrics": "artifacts/m1_verified_metrics.json",
            "m2_verified_metrics": "artifacts/m2_verified_metrics.json",
            "verified_metrics": "artifacts/verified_metrics.json",
        },
    }

    m2_payload = {
        "metadata": m2_metadata,
        "core_table2": core_table2,
        "cluster_significance": cluster_significance,
        "clinical_benchmarks": clinical_benchmarks,
        "calibration_robustness": calibration_robustness,
        "per_prompt_variance": per_prompt_variance,
        "latent_geometry_check": latent_geometry_check,
    }

    unified_payload = {
        "metadata": {
            "generated_utc": now_utc,
            "schema_version": "1.0",
            "scope_lock": "FLAIR-MESSIDOR-2 only (Control D omitted per scope)",
            "m1_metadata": copy.deepcopy(m1_data["metadata"]),
            "m2_metadata": copy.deepcopy(m2_metadata),
        },
        "control_a": copy.deepcopy(m1_data["control_a"]),
        "control_b": copy.deepcopy(m1_data["control_b"]),
        "control_c": copy.deepcopy(m1_data["control_c"]),
        "umls_telemetry": copy.deepcopy(m1_data["umls_telemetry"]),
        "core_table2": copy.deepcopy(core_table2),
        "cluster_significance": copy.deepcopy(cluster_significance),
        "clinical_benchmarks": copy.deepcopy(clinical_benchmarks),
        "calibration_robustness": copy.deepcopy(calibration_robustness),
        "per_prompt_variance": copy.deepcopy(per_prompt_variance),
        "latent_geometry_check": copy.deepcopy(latent_geometry_check),
    }

    # Final schema and integrity assertions
    assert list(unified_payload.keys()) == EXPECTED_TOP_LEVEL_KEYS
    for k in ("control_a", "control_b", "control_c", "umls_telemetry"):
        assert unified_payload[k] == m1_data[k], f"Verbatim M1 preservation failed for key: {k}"
    for fam in FAMILIES:
        assert (
            abs(
                calibration_robustness["by_family"][fam]["native_tau"]["ece_equal_width"]["M10"]
                - m1_data["control_a"]["gate0_verification"]["cached_baseline_reproduction_matches_csv"][fam]["ece_m10"]
            )
            < 1e-12
        )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    m2_path = ARTIFACTS_DIR / "m2_verified_metrics.json"
    unified_path = ARTIFACTS_DIR / "verified_metrics.json"
    with open(m2_path, "w", encoding="utf-8") as f:
        json.dump(m2_payload, f, indent=2)
    with open(unified_path, "w", encoding="utf-8") as f:
        json.dump(unified_payload, f, indent=2)

    total_elapsed = time.perf_counter() - t_start
    print(f"[M2] Wrote {m2_path} and {unified_path} cleanly in {total_elapsed:.2f}s.")
    print("[M2] Summary:")
    for fam in FAMILIES:
        c = ci_by_fam[fam]
        cv = calibration_robustness["by_family"][fam]["cv_temperature_scaling"]
        print(
            f"  {fam:12s} | Acc={c['accuracy']['point']*100:5.2f}% [{c['accuracy']['ci95_low']*100:5.2f},{c['accuracy']['ci95_high']*100:5.2f}] "
            f"| QWK={c['qwk']['point']:.4f} | AUROC={c['referable_dr_auroc']['point']:.4f} "
            f"| ECE10={c['micro_ece_m10']['point']*100:5.2f}% -> OOF={cv['oof_pooled']['ece_equal_width']['M10']*100:5.2f}% (tau*={cv['tau_star_mean']:.2f})"
        )
    print(
        f"  GEE Joint (T_label vs T_expert): OR={gee_joint['pairwise_wald_contrasts']['T_label_vs_T_expert']['odds_ratio_f1_over_f2']:.4f}, "
        f"Wald chi2={gee_joint['pairwise_wald_contrasts']['T_label_vs_T_expert']['wald_chi2']:.2f}, "
        f"p={gee_joint['pairwise_wald_contrasts']['T_label_vs_T_expert']['p_value']:.2e} "
        f"(vs invalid pooled McNemar chi2={legacy_mcnemar['pairs']['T_label_vs_T_expert']['chi2_corrected']:.2f})"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Image-clustered statistical re-analysis, QWK/RDR benchmarks, and CV temperature scaling.")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--cache-dir", type=Path, default=WEEK4_DIR)
    p.add_argument("--messidor-csv", type=Path, default=MESSIDOR_CSV)
    p.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    DATA_DIR = args.data_dir
    WEEK4_DIR = args.cache_dir
    MESSIDOR_CSV = args.messidor_csv
    ARTIFACTS_DIR = args.artifacts_dir
    main()
