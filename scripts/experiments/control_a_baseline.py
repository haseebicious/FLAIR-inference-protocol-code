"""Milestone 1 — Gate 0 Sanity Check & Control A Official FLAIR Baseline Verification.

Implements:
1. Gate 0 verification:
   - Full 1,748 -> 1,744 messidor_data.csv alignment & class count assertions.
   - Row-by-row check against cached embeddings directory/cached_vision_metadata_1744.csv.
   - File existence check for all 1,744 images in data/Messidor-2/IMAGES/.
   - Offline loading of FLAIRModel ('jusiro2/FLAIR', local_files_only=LOCAL_FILES_ONLY).
   - Full 1,744-image GPU re-extraction via FLAIRModel.preprocess_image + vision_model
     compared against cached embeddings directory/cached_vision_embeddings_1744.npy.
   - Exact logit_scale check vs cached embeddings directory/flair_logit_scale.npy.
   - Verification of data/final_prompts.json == cached_text_metadata_custom.csv (200 rows)
     and reproduction of thesis_audit_results_1744.csv cached baseline metrics.
2. Control A 24-row factorial ablation table (A01..A24) decomposing the T_expert vs T_label gap.
3. Per-variation diagnosis (variations 1..10) for A17 (T_label) and A19 (T_expert).

Outputs:
- artifacts/control_a_results.json
"""

from __future__ import annotations

import copy
import itertools
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

# Enforce strict offline HuggingFace loading before importing flair/transformers
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)

# Configurable paths relative to repository root (override via CLI flags)
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = WORKSPACE_ROOT / "data"
PROMPTS_DIR = WORKSPACE_ROOT / "prompts"
WEEK4_DIR = (WORKSPACE_ROOT / "results") if (WORKSPACE_ROOT / "results" / "cached_vision_embeddings_1744.npy").exists() else (
    (DATA_DIR / "Week 4 Work") if (DATA_DIR / "Week 4 Work").exists() else DATA_DIR
)
IMAGES_DIR = (DATA_DIR / "IMAGES") if (DATA_DIR / "IMAGES").exists() else (DATA_DIR / "Messidor-2" / "IMAGES")
MESSIDOR_CSV = (DATA_DIR / "messidor_data.csv") if (DATA_DIR / "messidor_data.csv").exists() else (DATA_DIR / "Messidor-2" / "Third Party Grading" / "messidor_data.csv")
ARTIFACTS_DIR = WORKSPACE_ROOT / "artifacts"
LOCAL_FILES_ONLY = False

# Official published FLAIR domain-knowledge descriptions for the 5 ICDR DR grades (Silva-Rodriguez et al., 2025)
FLAIR_PUBLISHED_DR_DESCRIPTIONS: List[List[str]] = [
    ["no relevant haemorrhages, microaneurysms or exudates", "no microaneurysms", "no referable lesions"],
    ["few microaneurysms", "few hard exudates", "few retinal haemorrhages"],
    ["retinal haemorrhages in few quadrants", "many haemorrhages", "cotton wool spots"],
    ["severe haemorrhages in all four quadrants", "venous beading", "intraretinal microvascular abnormalities"],
    ["diabetic retinopathy with neovascularization at the disk", "neovascularization"],
]

FLAIR_CATEGORIES = [
    "no diabetic retinopathy",
    "mild diabetic retinopathy",
    "moderate diabetic retinopathy",
    "severe diabetic retinopathy",
    "proliferative diabetic retinopathy",
]
GRADE_TO_IDX = {"noDR": 0, "mildDR": 1, "modDR": 2, "sevDR": 3, "prolDR": 4}
EXPECTED_CLASS_COUNTS = {0: 1017, 1: 270, 2: 347, 3: 75, 4: 35}


def calculate_ece(confidences: np.ndarray, accuracies: np.ndarray, num_bins: int = 10) -> float:
    """Uniform-bin ECE matching code/FLAIR-inference-protocol-code-main/scripts/3_inference_and_audit.py."""
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        if i == num_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        prop_in_bin = float(np.mean(in_bin))
        if prop_in_bin > 0.0:
            acc_in_bin = float(np.mean(accuracies[in_bin]))
            conf_in_bin = float(np.mean(confidences[in_bin]))
            ece += abs(acc_in_bin - conf_in_bin) * prop_in_bin
    return float(ece)


def compute_classification_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
) -> Dict[str, Any]:
    """Compute all Control A classification, calibration, and Referable DR metrics."""
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    correct = (preds == y_true).astype(float)

    acc = float(np.mean(correct))
    aca = float(balanced_accuracy_score(y_true, preds))
    macro_f1 = float(f1_score(y_true, preds, labels=[0, 1, 2, 3, 4], average="macro", zero_division=0))
    qwk = float(cohen_kappa_score(y_true, preds, labels=[0, 1, 2, 3, 4], weights="quadratic"))
    mean_conf = float(np.mean(confidences))
    ece_m10 = calculate_ece(confidences, correct, num_bins=10)

    # Referable DR: grade >= 2 vs 0-1
    y_ref = (y_true >= 2).astype(int)
    p_ref = probs[:, 2] + probs[:, 3] + probs[:, 4]
    pred_ref = (p_ref >= 0.5).astype(int)
    ref_auroc = float(roc_auc_score(y_ref, p_ref))
    tp = int(np.sum((y_ref == 1) & (pred_ref == 1)))
    fn = int(np.sum((y_ref == 1) & (pred_ref == 0)))
    tn = int(np.sum((y_ref == 0) & (pred_ref == 0)))
    fp = int(np.sum((y_ref == 0) & (pred_ref == 1)))
    ref_sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    ref_spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    ref_f1 = float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0

    pred_counts = {str(c): int(np.sum(preds == c)) for c in range(5)}

    return {
        "n_predictions": int(len(y_true)),
        "accuracy": acc,
        "aca": aca,
        "macro_f1": macro_f1,
        "qwk": qwk,
        "referable_dr_auroc": ref_auroc,
        "referable_dr_sensitivity": ref_sens,
        "referable_dr_specificity": ref_spec,
        "referable_dr_f1": ref_f1,
        "ece_m10": ece_m10,
        "mean_confidence": mean_conf,
        "predicted_class_counts": pred_counts,
    }


def eval_logits_from_prototypes(
    vis_embeds: np.ndarray,
    class_prototypes: np.ndarray,
    logit_scale: float,
    y_true: np.ndarray,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Compute softmax probabilities and metrics for (N, D) vis_embeds and (5, D) prototypes."""
    scaled_logits = np.dot(vis_embeds, class_prototypes.T) * logit_scale
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return compute_classification_metrics(y_true, probs), probs


def encode_texts_single_string(
    model: Any,
    texts: Sequence[str],
    device: torch.device,
) -> np.ndarray:
    """Encode each text prompt individually (batch_size=1, zero [PAD]) -> L2-normalized (N, 512)."""
    embs = []
    with torch.no_grad():
        for txt in texts:
            tokens = model.text_model.tokenizer(
                [txt], truncation=True, padding=True, return_tensors="pt"
            )
            input_ids = tokens["input_ids"].to(device).to(torch.long)
            attention_mask = tokens["attention_mask"].to(device).to(torch.long)
            emb = model.text_model(input_ids, attention_mask)
            embs.append(emb.squeeze(0).detach().cpu().numpy())
    arr = np.vstack(embs).astype(np.float64)
    arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
    return arr


def encode_class_batched(
    model: Any,
    per_class_texts: Sequence[Sequence[str]],
    device: torch.device,
    renormalize: bool = True,
) -> np.ndarray:
    """Encode each class's prompt list together with padding=True (matching FLAIRModel.compute_text_embeddings)."""
    prototypes = []
    with torch.no_grad():
        for desc_list in per_class_texts:
            tokens = model.text_model.tokenizer(
                list(desc_list), truncation=True, padding=True, return_tensors="pt"
            )
            input_ids = tokens["input_ids"].to(device).to(torch.long)
            attention_mask = tokens["attention_mask"].to(device).to(torch.long)
            text_embeds = model.text_model(input_ids, attention_mask)
            proto = text_embeds.mean(0).detach().cpu().numpy().astype(np.float64)
            if renormalize:
                proto = proto / np.linalg.norm(proto)
            prototypes.append(proto)
    return np.vstack(prototypes)


def encode_class_single_mean(
    model: Any,
    per_class_texts: Sequence[Sequence[str]],
    device: torch.device,
    renormalize: bool = True,
) -> np.ndarray:
    """Encode each prompt single-string (zero [PAD]) and average per class."""
    prototypes = []
    for desc_list in per_class_texts:
        embs = encode_texts_single_string(model, desc_list, device)
        proto = np.mean(embs, axis=0)
        if renormalize:
            proto = proto / np.linalg.norm(proto)
        prototypes.append(proto)
    return np.vstack(prototypes)


def wrap_caption(texts: Sequence[str], caption: str = "A fundus photograph of [CLS]") -> List[str]:
    return [caption.replace("[CLS]", t) for t in texts]


def get_published_dr_descriptions() -> List[List[str]]:
    """Return the official published 5-grade DR domain-knowledge descriptions (Silva-Rodriguez et al., 2025)."""
    return [list(x) for x in FLAIR_PUBLISHED_DR_DESCRIPTIONS]


def _unused_parse_official_prompts_txt(path: Path) -> List[List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    assert len(lines) >= 5, f"Expected >=5 lines in {path}, got {len(lines)}"
    dr_lists = []
    for i in range(5):
        phrases = re.findall(r'"([^"]+)"', lines[i])
        assert len(phrases) >= 2, f"Unexpected line {i+1} format in {path}: {lines[i]}"
        dr_lists.append(phrases)
    assert [len(x) for x in dr_lists] == [3, 3, 3, 3, 2], f"Unexpected counts: {[len(x) for x in dr_lists]}"
    return dr_lists


def run_gate0_verification(
    model: Any,
    device: torch.device,
    vis_embeds_cached: np.ndarray,
    vis_meta: pd.DataFrame,
    txt_embeds_custom: np.ndarray,
    txt_meta_custom: pd.DataFrame,
    logit_scale: float,
) -> Dict[str, Any]:
    """Run full Gate 0 assertions including full 1,744-image GPU re-extraction."""
    print("[Gate 0] Checking messidor_data.csv & cached_vision_metadata_1744.csv...")
    messidor_df = pd.read_csv(MESSIDOR_CSV)
    assert len(messidor_df) == 1748, f"Expected 1748 rows in messidor_data.csv, got {len(messidor_df)}"

    ungradable_df = messidor_df[
        (messidor_df["adjudicated_gradable"] == 0) | (messidor_df["adjudicated_dr_grade"].isna())
    ]
    assert len(ungradable_df) == 4, f"Expected 4 ungradable rows, got {len(ungradable_df)}"
    excluded_ids = ungradable_df["image_id"].tolist()

    gradable_df = messidor_df[
        messidor_df["adjudicated_dr_grade"].isin([0.0, 1.0, 2.0, 3.0, 4.0])
    ].reset_index(drop=True)
    assert len(gradable_df) == 1744, f"Expected 1744 gradable rows, got {len(gradable_df)}"

    assert len(vis_meta) == 1744, f"Expected 1744 rows in cached_vision_metadata_1744.csv, got {len(vis_meta)}"
    assert (vis_meta["image_id"].values == gradable_df["image_id"].values).all(), "image_id mismatch!"
    assert (
        vis_meta["adjudicated_dr_grade"].astype(int).values
        == gradable_df["adjudicated_dr_grade"].astype(int).values
    ).all(), "adjudicated_dr_grade mismatch!"

    actual_counts = {
        int(k): int(v) for k, v in vis_meta["adjudicated_dr_grade"].astype(int).value_counts().sort_index().items()
    }
    assert actual_counts == EXPECTED_CLASS_COUNTS, f"Class count mismatch: {actual_counts}"

    for grade_idx, expected_label in enumerate(FLAIR_CATEGORIES):
        mapped = vis_meta.loc[vis_meta["adjudicated_dr_grade"].astype(int) == grade_idx, "mapped_flair_class"].unique()
        assert list(mapped) == [expected_label], f"Label mapping mismatch for {grade_idx}: {mapped}"

    # Verify all 1,744 image files exist
    image_paths = [IMAGES_DIR / img_id for img_id in vis_meta["image_id"].tolist()]
    missing_files = [str(p) for p in image_paths if not p.exists()]
    assert len(missing_files) == 0, f"Missing {len(missing_files)} image files, e.g. {missing_files[:3]}"

    # Check logit_scale
    model_logit_scale = float(model.logit_scale.exp().item())
    logit_scale_diff = abs(model_logit_scale - logit_scale)
    assert logit_scale_diff < 1e-6, f"logit_scale diff {logit_scale_diff} >= 1e-6"

    # Full 1,744-image GPU re-extraction
    print("[Gate 0] Re-extracting all 1,744 vision embeddings via FLAIRModel.preprocess_image + vision_model...")
    t0 = time.time()
    reextracted = np.zeros_like(vis_embeds_cached, dtype=np.float32)
    with torch.no_grad():
        for idx, p in enumerate(image_paths):
            img = np.array(Image.open(p).convert("RGB"), dtype=float)
            img_tensor = model.preprocess_image(img).to(device)
            emb = model.vision_model(img_tensor)
            reextracted[idx] = emb.squeeze(0).detach().cpu().numpy()
            if (idx + 1) % 500 == 0 or (idx + 1) == len(image_paths):
                print(f"  Processed {idx + 1}/{len(image_paths)} images ({time.time() - t0:.1f}s)")

    abs_diffs = np.abs(reextracted - vis_embeds_cached)
    max_abs_diff = float(np.max(abs_diffs))
    mean_abs_diff = float(np.mean(abs_diffs))

    # Cosine similarity against cached embeddings
    re_norm = reextracted / np.linalg.norm(reextracted, axis=1, keepdims=True)
    ca_norm = vis_embeds_cached / np.linalg.norm(vis_embeds_cached, axis=1, keepdims=True)
    cos_sims = np.sum(re_norm * ca_norm, axis=1)
    min_cos_sim = float(np.min(cos_sims))
    mean_cos_sim = float(np.mean(cos_sims))
    l2_norms = np.linalg.norm(reextracted, axis=1)
    max_norm_dev = float(np.max(np.abs(l2_norms - 1.0)))

    assert max_abs_diff < 1e-5, f"max_abs_diff={max_abs_diff} >= 1e-5"
    assert min_cos_sim > 0.99999, f"min_cos_sim={min_cos_sim} <= 0.99999"
    assert max_norm_dev < 1e-5, f"max_norm_dev={max_norm_dev} >= 1e-5"

    # Check final_prompts.json == cached_text_metadata_custom.csv
    fp_path = (PROMPTS_DIR / "final_prompts.json") if (PROMPTS_DIR / "final_prompts.json").exists() else (DATA_DIR / "final_prompts.json")
    with open(fp_path, "r", encoding="utf-8") as f:
        final_prompts = json.load(f)
    assert len(final_prompts) == 200 and len(txt_meta_custom) == 200
    fp_df = pd.DataFrame(final_prompts)
    for col in ["Grade", "Family", "Variation_Number", "Prompt_Text"]:
        assert (fp_df[col].values == txt_meta_custom[col].values).all(), f"final_prompts mismatch on {col}"

    # Verify cached pooled metrics (cross-checking thesis_audit_results_1744.csv when present)
    audit_csv_path = WEEK4_DIR / "thesis_audit_results_1744.csv"
    audit_csv = pd.read_csv(audit_csv_path).set_index("Prompt Family") if audit_csv_path.exists() else None
    vis_unit = ca_norm.astype(np.float64)
    txt_unit = (txt_embeds_custom / np.linalg.norm(txt_embeds_custom, axis=1, keepdims=True)).astype(np.float64)
    y_true = vis_meta["adjudicated_dr_grade"].astype(int).values
    txt_meta_work = txt_meta_custom.copy()
    txt_meta_work["numerical_grade"] = txt_meta_work["Grade"].map(GRADE_TO_IDX)

    cached_repro = {}
    for fam in ["T_label", "T_expert", "T_layman", "T_exclusion"]:
        all_probs, all_y = [], []
        for var_num in range(1, 11):
            mask = (txt_meta_work["Family"] == fam) & (txt_meta_work["Variation_Number"] == var_num)
            vmeta = txt_meta_work[mask].sort_values("numerical_grade")
            assert len(vmeta) == 5
            v_emb = txt_unit[vmeta.index.values]
            scaled = np.dot(vis_unit, v_emb.T) * logit_scale
            exp_l = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
            probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
            all_probs.append(probs)
            all_y.append(y_true)
        pooled_m = compute_classification_metrics(np.concatenate(all_y), np.vstack(all_probs))
        cached_repro[fam] = pooled_m
        if audit_csv is not None:
            assert f"{pooled_m['accuracy']*100:.2f}%" == audit_csv.loc[fam, "Accuracy"], f"Acc mismatch {fam}"
            assert f"{pooled_m['ece_m10']*100:.2f}%" == audit_csv.loc[fam, "ECE"], f"ECE mismatch {fam}"
            assert f"{pooled_m['mean_confidence']*100:.2f}%" == audit_csv.loc[fam, "Avg Confidence"], f"Conf mismatch {fam}"

    print(
        f"[Gate 0 PASS] 1,744 images verified: max_abs_diff={max_abs_diff:.2e}, "
        f"min_cos_sim={min_cos_sim:.8f}, logit_scale={model_logit_scale:.6f}"
    )

    return {
        "messidor_csv_total_rows": int(len(messidor_df)),
        "excluded_ungradable_count": int(len(ungradable_df)),
        "excluded_image_ids": excluded_ids,
        "gradable_image_count": int(len(gradable_df)),
        "class_counts": {str(k): v for k, v in actual_counts.items()},
        "metadata_row_by_row_match": True,
        "all_1744_image_files_exist": True,
        "logit_scale_model": model_logit_scale,
        "logit_scale_cached": logit_scale,
        "logit_scale_abs_diff": logit_scale_diff,
        "full_1744_gpu_reextraction": {
            "n_images": int(len(image_paths)),
            "max_abs_diff_vs_cached": max_abs_diff,
            "mean_abs_diff_vs_cached": mean_abs_diff,
            "min_cos_sim_vs_cached": min_cos_sim,
            "mean_cos_sim_vs_cached": mean_cos_sim,
            "max_l2_norm_deviation": max_norm_dev,
            "elapsed_seconds": float(time.time() - t0),
        },
        "final_prompts_matches_week4_metadata_csv": True,
        "cached_baseline_reproduction_matches_csv": cached_repro,
    }


def main() -> Dict[str, Any]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Control A] Using device: {device}")

    # Load FLAIR strictly offline
    from flair.modeling.dictionary import definitions
    from flair.modeling.model import FLAIRModel

    original_definitions = copy.deepcopy(definitions)
    model = FLAIRModel.from_pretrained("jusiro2/FLAIR", local_files_only=LOCAL_FILES_ONLY).eval().to(device)

    vis_embeds_raw = np.load(WEEK4_DIR / "cached_vision_embeddings_1744.npy")
    vis_meta = pd.read_csv(WEEK4_DIR / "cached_vision_metadata_1744.csv")
    txt_embeds_custom = np.load(WEEK4_DIR / "cached_text_embeddings_custom.npy")
    txt_meta_custom = pd.read_csv(WEEK4_DIR / "cached_text_metadata_custom.csv")
    logit_scale = float(np.load(WEEK4_DIR / "flair_logit_scale.npy").item())

    # 1. Gate 0
    gate0 = run_gate0_verification(
        model=model,
        device=device,
        vis_embeds_cached=vis_embeds_raw,
        vis_meta=vis_meta,
        txt_embeds_custom=txt_embeds_custom,
        txt_meta_custom=txt_meta_custom,
        logit_scale=logit_scale,
    )

    vis_unit = (vis_embeds_raw / np.linalg.norm(vis_embeds_raw, axis=1, keepdims=True)).astype(np.float64)
    y_true = vis_meta["adjudicated_dr_grade"].astype(int).values
    txt_unit_custom = (txt_embeds_custom / np.linalg.norm(txt_embeds_custom, axis=1, keepdims=True)).astype(np.float64)

    txt_meta_work = txt_meta_custom.copy()
    txt_meta_work["numerical_grade"] = txt_meta_work["Grade"].map(GRADE_TO_IDX)

    official_txt_lists = get_published_dr_descriptions()

    # Helper to extract per-class custom prompts for T_label and T_expert
    tlabel_by_class: List[List[str]] = []
    texpert_by_class: List[List[str]] = []
    tlabel_by_var: List[List[str]] = []
    texpert_by_var: List[List[str]] = []

    for g_idx in range(5):
        tl_rows = txt_meta_work[(txt_meta_work["Family"] == "T_label") & (txt_meta_work["numerical_grade"] == g_idx)].sort_values("Variation_Number")
        te_rows = txt_meta_work[(txt_meta_work["Family"] == "T_expert") & (txt_meta_work["numerical_grade"] == g_idx)].sort_values("Variation_Number")
        tlabel_by_class.append(tl_rows["Prompt_Text"].tolist())
        texpert_by_class.append(te_rows["Prompt_Text"].tolist())

    for v in range(1, 11):
        tl_v = txt_meta_work[(txt_meta_work["Family"] == "T_label") & (txt_meta_work["Variation_Number"] == v)].sort_values("numerical_grade")
        te_v = txt_meta_work[(txt_meta_work["Family"] == "T_expert") & (txt_meta_work["Variation_Number"] == v)].sort_values("numerical_grade")
        tlabel_by_var.append(tl_v["Prompt_Text"].tolist())
        texpert_by_var.append(te_v["Prompt_Text"].tolist())

    ablation_table: Dict[str, Dict[str, Any]] = {}

    def record_row(row_id: str, meta_info: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        entry = {**meta_info, **metrics}
        ablation_table[row_id] = entry
        print(
            f"  [{row_id}] Acc={metrics['accuracy']*100:.2f}% | ACA={metrics['aca']*100:.2f}% | "
            f"MF1={metrics['macro_f1']:.4f} | QWK={metrics['qwk']:.4f} | "
            f"RefAUROC={metrics['referable_dr_auroc']:.4f} | ECE={metrics['ece_m10']*100:.2f}%"
        )

    print("\n[Control A] Running 24-row factorial ablation table (A01..A24)...")

    # A01: compute_text_embeddings(dk=False)
    with torch.no_grad():
        _, emb_a01 = model.compute_text_embeddings(FLAIR_CATEGORIES, domain_knowledge=False)
    proto_a01 = emb_a01.detach().cpu().numpy().astype(np.float64)
    m_a01, _ = eval_logits_from_prototypes(vis_unit, proto_a01, logit_scale, y_true)
    record_row(
        "A01_official_dk_false_batched",
        {
            "prompt_source": "compute_text_embeddings(dk=False)",
            "template": True,
            "base_label_appended": True,
            "aggregation": "single_prompt_per_class",
            "tokenization": "batched_padding",
            "prototype_norm": "unit",
        },
        m_a01,
    )

    # A02: cats + template, single-string
    proto_a02 = encode_texts_single_string(model, wrap_caption(FLAIR_CATEGORIES), device)
    m_a02, _ = eval_logits_from_prototypes(vis_unit, proto_a02, logit_scale, y_true)
    record_row(
        "A02_official_dk_false_single",
        {
            "prompt_source": "FLAIR_CATEGORIES + template",
            "template": True,
            "base_label_appended": True,
            "aggregation": "single_prompt_per_class",
            "tokenization": "single_string_no_pad",
            "prototype_norm": "unit",
        },
        m_a02,
    )

    # A03: raw base labels, single-string
    proto_a03 = encode_texts_single_string(model, FLAIR_CATEGORIES, device)
    m_a03, _ = eval_logits_from_prototypes(vis_unit, proto_a03, logit_scale, y_true)
    record_row(
        "A03_raw_base_labels_single",
        {
            "prompt_source": "FLAIR_CATEGORIES (raw)",
            "template": False,
            "base_label_appended": True,
            "aggregation": "single_prompt_per_class",
            "tokenization": "single_string_no_pad",
            "prototype_norm": "unit",
        },
        m_a03,
    )

    # A04 & A05: compute_text_embeddings(dk=True) — restore definitions cleanly afterward!
    assert len(definitions["mild diabetic retinopathy"]) == 1, "definitions already mutated before A04!"
    with torch.no_grad():
        _, emb_a04 = model.compute_text_embeddings(FLAIR_CATEGORIES, domain_knowledge=True)
    proto_a04 = emb_a04.detach().cpu().numpy().astype(np.float64)
    proto_a04_norms = [float(x) for x in np.linalg.norm(proto_a04, axis=1)]

    # Restore definitions immediately to clean original state
    definitions.clear()
    definitions.update(copy.deepcopy(original_definitions))
    assert len(definitions["mild diabetic retinopathy"]) == 1, "Failed to restore definitions!"

    m_a04, _ = eval_logits_from_prototypes(vis_unit, proto_a04, logit_scale, y_true)
    record_row(
        "A04_official_dk_true_unrenorm",
        {
            "prompt_source": "compute_text_embeddings(dk=True) (dictionary.py)",
            "template": True,
            "base_label_appended": True,
            "aggregation": "class_mean",
            "tokenization": "class_batched_padding",
            "prototype_norm": "unrenormalized",
            "prototype_l2_norms": proto_a04_norms,
        },
        m_a04,
    )

    proto_a05 = proto_a04 / np.linalg.norm(proto_a04, axis=1, keepdims=True)
    m_a05, _ = eval_logits_from_prototypes(vis_unit, proto_a05, logit_scale, y_true)
    record_row(
        "A05_official_dk_true_renorm",
        {
            "prompt_source": "compute_text_embeddings(dk=True) (dictionary.py)",
            "template": True,
            "base_label_appended": True,
            "aggregation": "class_mean",
            "tokenization": "class_batched_padding",
            "prototype_norm": "l2_renormalized",
        },
        m_a05,
    )

    # Prepare clean dictionary.py lists (with and without base label)
    dict_with_lbl = []
    dict_no_lbl = []
    for cat in FLAIR_CATEGORIES:
        base_descs = copy.deepcopy(original_definitions[cat])
        dict_no_lbl.append(list(base_descs))
        with_lbl = list(base_descs)
        if cat not in with_lbl:
            with_lbl.append(cat)
        dict_with_lbl.append(with_lbl)

    # A06: dictionary.py + base label, template, single-string, class-mean, L2-renormalized
    proto_a06 = encode_class_single_mean(model, [wrap_caption(d) for d in dict_with_lbl], device, renormalize=True)
    m_a06, _ = eval_logits_from_prototypes(vis_unit, proto_a06, logit_scale, y_true)
    record_row(
        "A06_dict_dk_true_single_renorm",
        {
            "prompt_source": "dictionary.py + base label",
            "template": True,
            "base_label_appended": True,
            "aggregation": "class_mean",
            "tokenization": "single_string_no_pad",
            "prototype_norm": "l2_renormalized",
        },
        m_a06,
    )

    # Assert definitions still unmutated before A07
    assert len(definitions["mild diabetic retinopathy"]) == 1, "definitions mutated before A07!"

    # A07: dictionary.py ONLY, template, class-batched, L2-renormalized
    proto_a07 = encode_class_batched(model, [wrap_caption(d) for d in dict_no_lbl], device, renormalize=True)
    m_a07, _ = eval_logits_from_prototypes(vis_unit, proto_a07, logit_scale, y_true)
    record_row(
        "A07_dict_nolabel_tmpl_batched",
        {
            "prompt_source": "dictionary.py only",
            "template": True,
            "base_label_appended": False,
            "aggregation": "class_mean",
            "tokenization": "class_batched_padding",
            "prototype_norm": "l2_renormalized",
        },
        m_a07,
    )

    # A08: dictionary.py ONLY, template, single-string, L2-renormalized
    proto_a08 = encode_class_single_mean(model, [wrap_caption(d) for d in dict_no_lbl], device, renormalize=True)
    m_a08, _ = eval_logits_from_prototypes(vis_unit, proto_a08, logit_scale, y_true)
    record_row(
        "A08_dict_nolabel_tmpl_single",
        {
            "prompt_source": "dictionary.py only",
            "template": True,
            "base_label_appended": False,
            "aggregation": "class_mean",
            "tokenization": "single_string_no_pad",
            "prototype_norm": "l2_renormalized",
        },
        m_a08,
    )

    # A09: dictionary.py ONLY, raw (no template), single-string, L2-renormalized
    proto_a09 = encode_class_single_mean(model, dict_no_lbl, device, renormalize=True)
    m_a09, _ = eval_logits_from_prototypes(vis_unit, proto_a09, logit_scale, y_true)
    record_row(
        "A09_dict_nolabel_notmpl_single",
        {
            "prompt_source": "dictionary.py only (raw)",
            "template": False,
            "base_label_appended": False,
            "aggregation": "class_mean",
            "tokenization": "single_string_no_pad",
            "prototype_norm": "l2_renormalized",
        },
        m_a09,
    )

    # FLAIR published DR descriptions with & without base label
    offtxt_with_lbl = []
    for cat, descs in zip(FLAIR_CATEGORIES, official_txt_lists):
        d_copy = list(descs)
        if cat not in d_copy:
            d_copy.append(cat)
        offtxt_with_lbl.append(d_copy)

    # A10
    proto_a10 = encode_class_batched(model, [wrap_caption(d) for d in offtxt_with_lbl], device, renormalize=True)
    m_a10, _ = eval_logits_from_prototypes(vis_unit, proto_a10, logit_scale, y_true)
    record_row(
        "A10_offtxt_lbl_tmpl_batched",
        {
            "prompt_source": "FLAIR published DR descriptions + base label",
            "template": True,
            "base_label_appended": True,
            "aggregation": "class_mean",
            "tokenization": "class_batched_padding",
            "prototype_norm": "l2_renormalized",
        },
        m_a10,
    )

    # A11
    proto_a11 = encode_class_single_mean(model, [wrap_caption(d) for d in offtxt_with_lbl], device, renormalize=True)
    m_a11, _ = eval_logits_from_prototypes(vis_unit, proto_a11, logit_scale, y_true)
    record_row(
        "A11_offtxt_lbl_tmpl_single",
        {
            "prompt_source": "FLAIR published DR descriptions + base label",
            "template": True,
            "base_label_appended": True,
            "aggregation": "class_mean",
            "tokenization": "single_string_no_pad",
            "prototype_norm": "l2_renormalized",
        },
        m_a11,
    )

    # A12
    proto_a12 = encode_class_batched(model, [wrap_caption(d) for d in official_txt_lists], device, renormalize=True)
    m_a12, _ = eval_logits_from_prototypes(vis_unit, proto_a12, logit_scale, y_true)
    record_row(
        "A12_offtxt_nolbl_tmpl_batched",
        {
            "prompt_source": "FLAIR published DR descriptions (no base label)",
            "template": True,
            "base_label_appended": False,
            "aggregation": "class_mean",
            "tokenization": "class_batched_padding",
            "prototype_norm": "l2_renormalized",
        },
        m_a12,
    )

    # A13
    proto_a13 = encode_class_single_mean(model, [wrap_caption(d) for d in official_txt_lists], device, renormalize=True)
    m_a13, _ = eval_logits_from_prototypes(vis_unit, proto_a13, logit_scale, y_true)
    record_row(
        "A13_offtxt_nolbl_tmpl_single",
        {
            "prompt_source": "FLAIR published DR descriptions (no base label)",
            "template": True,
            "base_label_appended": False,
            "aggregation": "class_mean",
            "tokenization": "single_string_no_pad",
            "prototype_norm": "l2_renormalized",
        },
        m_a13,
    )

    # A14
    proto_a14 = encode_class_single_mean(model, official_txt_lists, device, renormalize=True)
    m_a14, _ = eval_logits_from_prototypes(vis_unit, proto_a14, logit_scale, y_true)
    record_row(
        "A14_offtxt_nolbl_notmpl_single",
        {
            "prompt_source": "FLAIR published DR descriptions (raw)",
            "template": False,
            "base_label_appended": False,
            "aggregation": "class_mean",
            "tokenization": "single_string_no_pad",
            "prototype_norm": "l2_renormalized",
        },
        m_a14,
    )

    # Pre-encode all 14 official_txt phrases raw & wrapped (single-string) for A15 & A16 (162 Cartesian tuples)
    offtxt_raw_embs = [encode_texts_single_string(model, cls_phrases, device) for cls_phrases in official_txt_lists]
    offtxt_tmpl_embs = [
        encode_texts_single_string(model, wrap_caption(cls_phrases), device) for cls_phrases in official_txt_lists
    ]

    def run_cartesian_tuples(cls_emb_lists: List[np.ndarray]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        index_ranges = [range(len(c)) for c in cls_emb_lists]
        all_probs, all_y = [], []
        tuple_accs, tuple_acas, tuple_mf1s, tuple_qwks, tuple_eces = [], [], [], [], []
        for combo in itertools.product(*index_ranges):
            proto = np.vstack([cls_emb_lists[c_idx][p_idx] for c_idx, p_idx in enumerate(combo)])
            tm, tprobs = eval_logits_from_prototypes(vis_unit, proto, logit_scale, y_true)
            all_probs.append(tprobs)
            all_y.append(y_true)
            tuple_accs.append(tm["accuracy"])
            tuple_acas.append(tm["aca"])
            tuple_mf1s.append(tm["macro_f1"])
            tuple_qwks.append(tm["qwk"])
            tuple_eces.append(tm["ece_m10"])
        pooled_m = compute_classification_metrics(np.concatenate(all_y), np.vstack(all_probs))
        summary = {
            "n_tuples": len(tuple_accs),
            "accuracy_mean": float(np.mean(tuple_accs)),
            "accuracy_sd": float(np.std(tuple_accs, ddof=1)),
            "accuracy_min": float(np.min(tuple_accs)),
            "accuracy_max": float(np.max(tuple_accs)),
            "aca_mean": float(np.mean(tuple_acas)),
            "aca_sd": float(np.std(tuple_acas, ddof=1)),
            "aca_min": float(np.min(tuple_acas)),
            "aca_max": float(np.max(tuple_acas)),
            "macro_f1_mean": float(np.mean(tuple_mf1s)),
            "qwk_mean": float(np.mean(tuple_qwks)),
            "ece_m10_mean": float(np.mean(tuple_eces)),
        }
        return pooled_m, summary

    # A15
    m_a15, sum_a15 = run_cartesian_tuples(offtxt_raw_embs)
    record_row(
        "A15_offtxt_unensembled_notmpl",
        {
            "prompt_source": "FLAIR published DR descriptions (162 tuples)",
            "template": False,
            "base_label_appended": False,
            "aggregation": "unensembled_162_tuples_pooled",
            "tokenization": "single_string_no_pad",
            "prototype_norm": "unit",
            "tuple_summary": sum_a15,
        },
        m_a15,
    )

    # A16
    m_a16, sum_a16 = run_cartesian_tuples(offtxt_tmpl_embs)
    record_row(
        "A16_offtxt_unensembled_tmpl",
        {
            "prompt_source": "FLAIR published DR descriptions (162 tuples)",
            "template": True,
            "base_label_appended": False,
            "aggregation": "unensembled_162_tuples_pooled",
            "tokenization": "single_string_no_pad",
            "prototype_norm": "unit",
            "tuple_summary": sum_a16,
        },
        m_a16,
    )

    # Helper for pooled evaluation of 10 variations (cached or newly encoded)
    def eval_cached_pooled_and_per_var(family_name: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        all_probs, all_y = [], []
        per_var = []
        for v in range(1, 11):
            mask = (txt_meta_work["Family"] == family_name) & (txt_meta_work["Variation_Number"] == v)
            vmeta = txt_meta_work[mask].sort_values("numerical_grade")
            assert len(vmeta) == 5
            proto = txt_unit_custom[vmeta.index.values]
            vm, vprobs = eval_logits_from_prototypes(vis_unit, proto, logit_scale, y_true)
            all_probs.append(vprobs)
            all_y.append(y_true)
            per_var.append({
                "variation_number": v,
                "prompts_by_grade": {row["Grade"]: row["Prompt_Text"] for _, row in vmeta.iterrows()},
                **vm,
            })
        pooled = compute_classification_metrics(np.concatenate(all_y), np.vstack(all_probs))
        return pooled, per_var

    # A17: Custom T_label raw pooled
    m_a17, per_var_tlabel = eval_cached_pooled_and_per_var("T_label")
    record_row(
        "A17_custom_tlabel_raw_pooled",
        {
            "prompt_source": "Custom T_label (10 vars, cached)",
            "template": False,
            "base_label_appended": True,
            "aggregation": "unensembled_10_vars_pooled",
            "tokenization": "single_string_no_pad",
            "prototype_norm": "unit",
        },
        m_a17,
    )

    # A18: Custom T_label + template, single-string, 10 vars pooled
    all_probs_a18, all_y_a18 = [], []
    for v_texts in tlabel_by_var:
        proto = encode_texts_single_string(model, wrap_caption(v_texts), device)
        _, vprobs = eval_logits_from_prototypes(vis_unit, proto, logit_scale, y_true)
        all_probs_a18.append(vprobs)
        all_y_a18.append(y_true)
    m_a18 = compute_classification_metrics(np.concatenate(all_y_a18), np.vstack(all_probs_a18))
    record_row(
        "A18_custom_tlabel_tmpl_pooled",
        {
            "prompt_source": "Custom T_label (10 vars)",
            "template": True,
            "base_label_appended": True,
            "aggregation": "unensembled_10_vars_pooled",
            "tokenization": "single_string_no_pad",
            "prototype_norm": "unit",
        },
        m_a18,
    )

    # A19: Custom T_expert raw pooled
    m_a19, per_var_texpert = eval_cached_pooled_and_per_var("T_expert")
    record_row(
        "A19_custom_texpert_raw_pooled",
        {
            "prompt_source": "Custom T_expert (10 vars, cached)",
            "template": False,
            "base_label_appended": False,
            "aggregation": "unensembled_10_vars_pooled",
            "tokenization": "single_string_no_pad",
            "prototype_norm": "unit",
        },
        m_a19,
    )

    # A20: Custom T_expert raw, class-mean ensembled (10/cls), L2-renormalized
    texpert_raw_protos = []
    for g_idx in range(5):
        g_rows = txt_meta_work[(txt_meta_work["Family"] == "T_expert") & (txt_meta_work["numerical_grade"] == g_idx)]
        assert len(g_rows) == 10
        proto = np.mean(txt_unit_custom[g_rows.index.values], axis=0)
        proto = proto / np.linalg.norm(proto)
        texpert_raw_protos.append(proto)
    proto_a20 = np.vstack(texpert_raw_protos)
    m_a20, _ = eval_logits_from_prototypes(vis_unit, proto_a20, logit_scale, y_true)
    record_row(
        "A20_custom_texpert_raw_ens",
        {
            "prompt_source": "Custom T_expert (cached)",
            "template": False,
            "base_label_appended": False,
            "aggregation": "class_mean_10_per_cls",
            "tokenization": "single_string_no_pad",
            "prototype_norm": "l2_renormalized",
        },
        m_a20,
    )

    # A21: Custom T_expert + template, single-string, un-ensembled pooled
    all_probs_a21, all_y_a21 = [], []
    for v_texts in texpert_by_var:
        proto = encode_texts_single_string(model, wrap_caption(v_texts), device)
        _, vprobs = eval_logits_from_prototypes(vis_unit, proto, logit_scale, y_true)
        all_probs_a21.append(vprobs)
        all_y_a21.append(y_true)
    m_a21 = compute_classification_metrics(np.concatenate(all_y_a21), np.vstack(all_probs_a21))
    record_row(
        "A21_custom_texpert_tmpl_pooled",
        {
            "prompt_source": "Custom T_expert (10 vars)",
            "template": True,
            "base_label_appended": False,
            "aggregation": "unensembled_10_vars_pooled",
            "tokenization": "single_string_no_pad",
            "prototype_norm": "unit",
        },
        m_a21,
    )

    # A22: Custom T_expert + template, single-string, class-mean (10/cls), L2-renormalized
    proto_a22 = encode_class_single_mean(model, [wrap_caption(d) for d in texpert_by_class], device, renormalize=True)
    m_a22, _ = eval_logits_from_prototypes(vis_unit, proto_a22, logit_scale, y_true)
    record_row(
        "A22_custom_texpert_tmpl_ens_single",
        {
            "prompt_source": "Custom T_expert",
            "template": True,
            "base_label_appended": False,
            "aggregation": "class_mean_10_per_cls",
            "tokenization": "single_string_no_pad",
            "prototype_norm": "l2_renormalized",
        },
        m_a22,
    )

    # A23: Custom T_expert + template, class-batched (padding=True), class-mean (10/cls), L2-renormalized
    proto_a23 = encode_class_batched(model, [wrap_caption(d) for d in texpert_by_class], device, renormalize=True)
    m_a23, _ = eval_logits_from_prototypes(vis_unit, proto_a23, logit_scale, y_true)
    record_row(
        "A23_custom_texpert_tmpl_ens_batched",
        {
            "prompt_source": "Custom T_expert",
            "template": True,
            "base_label_appended": False,
            "aggregation": "class_mean_10_per_cls",
            "tokenization": "class_batched_padding",
            "prototype_norm": "l2_renormalized",
        },
        m_a23,
    )

    # A24: Custom T_expert + base label + template, class-batched, class-mean (11/cls), L2-renormalized
    texpert_with_lbl = [list(d) + [cat] for cat, d in zip(FLAIR_CATEGORIES, texpert_by_class)]
    proto_a24 = encode_class_batched(model, [wrap_caption(d) for d in texpert_with_lbl], device, renormalize=True)
    m_a24, _ = eval_logits_from_prototypes(vis_unit, proto_a24, logit_scale, y_true)
    record_row(
        "A24_custom_texpert_tmpl_lbl_ens_batched",
        {
            "prompt_source": "Custom T_expert + base label",
            "template": True,
            "base_label_appended": True,
            "aggregation": "class_mean_11_per_cls",
            "tokenization": "class_batched_padding",
            "prototype_norm": "l2_renormalized",
        },
        m_a24,
    )

    # Assert key baseline sanity checks
    assert len(ablation_table) == 24
    assert m_a04["accuracy"] > m_a01["accuracy"], "Expected A04 dk=True > A01 dk=False on Accuracy"
    assert m_a04["macro_f1"] > m_a01["macro_f1"], "Expected A04 dk=True > A01 dk=False on Macro-F1"
    assert m_a04["qwk"] > m_a01["qwk"], "Expected A04 dk=True > A01 dk=False on QWK"
    assert m_a04["referable_dr_auroc"] > m_a01["referable_dr_auroc"], "Expected A04 dk=True > A01 dk=False on Ref-DR AUROC"
    assert m_a04["ece_m10"] < m_a01["ece_m10"], "Expected A04 dk=True < A01 dk=False on ECE"

    # Key step-wise deltas for clear reporting
    factor_deltas = {
        "official_dk_true_minus_false_acc_pp": (m_a04["accuracy"] - m_a01["accuracy"]) * 100.0,
        "official_dk_true_minus_false_mf1": m_a04["macro_f1"] - m_a01["macro_f1"],
        "official_dk_true_minus_false_qwk": m_a04["qwk"] - m_a01["qwk"],
        "official_dk_true_minus_false_ref_auroc": m_a04["referable_dr_auroc"] - m_a01["referable_dr_auroc"],
        "official_dk_true_minus_false_ece_pp": (m_a04["ece_m10"] - m_a01["ece_m10"]) * 100.0,
        "texpert_ensembling_alone_acc_pp": (m_a20["accuracy"] - m_a19["accuracy"]) * 100.0,
        "texpert_ensembling_alone_ece_pp": (m_a20["ece_m10"] - m_a19["ece_m10"]) * 100.0,
        "texpert_template_alone_acc_pp": (m_a21["accuracy"] - m_a19["accuracy"]) * 100.0,
        "texpert_full_official_style_minus_raw_pooled_acc_pp": (m_a24["accuracy"] - m_a19["accuracy"]) * 100.0,
        "texpert_full_official_style_minus_raw_pooled_aca_pp": (m_a24["aca"] - m_a19["aca"]) * 100.0,
        "texpert_full_official_style_minus_raw_pooled_ece_pp": (m_a24["ece_m10"] - m_a19["ece_m10"]) * 100.0,
        "offtxt_template_removal_single_aca_pp": (m_a14["aca"] - m_a13["aca"]) * 100.0,
        "dict_pad_effect_acc_pp": (m_a05["accuracy"] - m_a06["accuracy"]) * 100.0,
        "offtxt_pad_effect_nolbl_acc_pp": (m_a12["accuracy"] - m_a13["accuracy"]) * 100.0,
    }

    output = {
        "metadata": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "model_id": "jusiro2/FLAIR",
            "local_files_only": LOCAL_FILES_ONLY,
            "device": str(device),
            "n_images": 1744,
            "ablation_rows_count": len(ablation_table),
        },
        "gate0_verification": gate0,
        "ablation_table": ablation_table,
        "factor_deltas": factor_deltas,
        "per_variation_diagnosis": {
            "A17_T_label": per_var_tlabel,
            "A19_T_expert": per_var_texpert,
        },
    }

    out_path = ARTIFACTS_DIR / "control_a_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n[Control A COMPLETE] Wrote {out_path}")
    return output


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gate 0 verification & Control A official FLAIR baseline ablation.")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--cache-dir", type=Path, default=WEEK4_DIR)
    p.add_argument("--prompts-dir", type=Path, default=PROMPTS_DIR)
    p.add_argument("--images-dir", type=Path, default=IMAGES_DIR)
    p.add_argument("--messidor-csv", type=Path, default=MESSIDOR_CSV)
    p.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    DATA_DIR = args.data_dir
    WEEK4_DIR = args.cache_dir
    PROMPTS_DIR = args.prompts_dir
    IMAGES_DIR = args.images_dir
    MESSIDOR_CSV = args.messidor_csv
    ARTIFACTS_DIR = args.artifacts_dir
    LOCAL_FILES_ONLY = args.local_files_only
    main()
