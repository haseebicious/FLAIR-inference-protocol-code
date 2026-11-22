"""Milestone 1 — Control B: Matched Minimal-Pair Negation Ablation.

Constructs the exact a priori 50 minimal pairs (CB01..CB50) across 5 lesion tiers x
2 registers (Clinical, Lay) x 5 negation operators, verifies all pre-inference
invariants (operator-only diff, <77 Bio_ClinicalBERT tokens), runs frozen FLAIR
inference on all 1,744 MESSIDOR-2 images under both raw_single_string and
wrapped_caption ("A fundus photograph of [CLS]") protocols, and computes:
- Delta S = S_neg - S_aff and discriminative DeltaDelta S with B=10,000 image-level
  bootstrap 95% CIs & paired Wilcoxon p-values
- Text embedding cosine similarity r_text and score correlation r_score
- Pairwise 2-way FPR / FNR / balanced accuracy
- Neutral-anchor binary FPR shift ("no diabetic retinopathy" and "normal healthy retina")
- 5-way paired grade-prompt swap FPR shift (both T_label v1 and T_label 10-var centroid)

Outputs:
- artifacts/control_b_pairs.json
- artifacts/control_b_results.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, wilcoxon

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = WORKSPACE_ROOT / "data"
WEEK4_DIR = (WORKSPACE_ROOT / "results") if (WORKSPACE_ROOT / "results" / "cached_vision_embeddings_1744.npy").exists() else (
    (DATA_DIR / "Week 4 Work") if (DATA_DIR / "Week 4 Work").exists() else DATA_DIR
)
ARTIFACTS_DIR = WORKSPACE_ROOT / "artifacts"
LOCAL_FILES_ONLY = False

CAPTION_TEMPLATE = "A fundus photograph of [CLS]"
BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 42

FLAIR_CATEGORIES = [
    "no diabetic retinopathy",
    "mild diabetic retinopathy",
    "moderate diabetic retinopathy",
    "severe diabetic retinopathy",
    "proliferative diabetic retinopathy",
]
GRADE_TO_IDX = {"noDR": 0, "mildDR": 1, "modDR": 2, "sevDR": 3, "prolDR": 4}

# Exact 50 a priori minimal pairs (CB01-CB50)
RAW_CONTROL_B_SPEC: List[Tuple[str, str, int, str, str, str, str, str]] = [
    # T1: Mild (g_min = 1)
    ("CB01", "T1: Mild", 1, "Clinical", "with / without", "with", "without", "Fundus with microaneurysms.", "Fundus without microaneurysms."),
    ("CB02", "T1: Mild", 1, "Clinical", "Presence of / Absence of", "Presence of", "Absence of", "Presence of retinal microaneurysms.", "Absence of retinal microaneurysms."),
    ("CB03", "T1: Mild", 1, "Clinical", "showing / showing no", "showing", "showing no", "Fundus photograph showing capillary microaneurysms.", "Fundus photograph showing no capillary microaneurysms."),
    ("CB04", "T1: Mild", 1, "Clinical", "reveals / reveals no", "reveals", "reveals no", "Retinal examination reveals isolated retinal microaneurysms.", "Retinal examination reveals no isolated retinal microaneurysms."),
    ("CB05", "T1: Mild", 1, "Clinical", "exhibits / exhibits no", "exhibits", "exhibits no", "The retina exhibits scattered macular microaneurysms.", "The retina exhibits no scattered macular microaneurysms."),
    ("CB06", "T1: Mild", 1, "Lay", "with / without", "with", "without", "Back of the eye with tiny red dots.", "Back of the eye without tiny red dots."),
    ("CB07", "T1: Mild", 1, "Lay", "Presence of / Absence of", "Presence of", "Absence of", "Presence of small red spots in the back of the eye.", "Absence of small red spots in the back of the eye."),
    ("CB08", "T1: Mild", 1, "Lay", "showing / showing no", "showing", "showing no", "Eye photograph showing tiny swollen blood vessels.", "Eye photograph showing no tiny swollen blood vessels."),
    ("CB09", "T1: Mild", 1, "Lay", "reveals / reveals no", "reveals", "reveals no", "Eye examination reveals pinpoint red blood blisters.", "Eye examination reveals no pinpoint red blood blisters."),
    ("CB10", "T1: Mild", 1, "Lay", "exhibits / exhibits no", "exhibits", "exhibits no", "The back of the eye exhibits isolated tiny red marks.", "The back of the eye exhibits no isolated tiny red marks."),
    # T2: Mod (g_min = 2)
    ("CB11", "T2: Mod", 2, "Clinical", "with / without", "with", "without", "Fundus with intraretinal haemorrhages.", "Fundus without intraretinal haemorrhages."),
    ("CB12", "T2: Mod", 2, "Clinical", "Presence of / Absence of", "Presence of", "Absence of", "Presence of hard exudates in the macular region.", "Absence of hard exudates in the macular region."),
    ("CB13", "T2: Mod", 2, "Clinical", "showing / showing no", "showing", "showing no", "Fundus photograph showing cotton wool spots.", "Fundus photograph showing no cotton wool spots."),
    ("CB14", "T2: Mod", 2, "Clinical", "reveals / reveals no", "reveals", "reveals no", "Retinal examination reveals dot-and-blot retinal haemorrhages.", "Retinal examination reveals no dot-and-blot retinal haemorrhages."),
    ("CB15", "T2: Mod", 2, "Clinical", "exhibits / exhibits no", "exhibits", "exhibits no", "The retina exhibits hard exudates and cotton wool spots.", "The retina exhibits no hard exudates and cotton wool spots."),
    ("CB16", "T2: Mod", 2, "Lay", "with / without", "with", "without", "Back of the eye with red fluid spots in several areas.", "Back of the eye without red fluid spots in several areas."),
    ("CB17", "T2: Mod", 2, "Lay", "Presence of / Absence of", "Presence of", "Absence of", "Presence of yellow fluid deposits.", "Absence of yellow fluid deposits."),
    ("CB18", "T2: Mod", 2, "Lay", "showing / showing no", "showing", "showing no", "Eye photograph showing fluffy white cotton-like patches.", "Eye photograph showing no fluffy white cotton-like patches."),
    ("CB19", "T2: Mod", 2, "Lay", "reveals / reveals no", "reveals", "reveals no", "Eye examination reveals multiple small patches of blood leakage.", "Eye examination reveals no multiple small patches of blood leakage."),
    ("CB20", "T2: Mod", 2, "Lay", "exhibits / exhibits no", "exhibits", "exhibits no", "The back of the eye exhibits hard yellow leakage spots.", "The back of the eye exhibits no hard yellow leakage spots."),
    # T3: Sev (g_min = 3)
    ("CB21", "T3: Sev", 3, "Clinical", "with / without", "with", "without", "Fundus with severe haemorrhages in all four quadrants.", "Fundus without severe haemorrhages in all four quadrants."),
    ("CB22", "T3: Sev", 3, "Clinical", "Presence of / Absence of", "Presence of", "Absence of", "Presence of venous beading.", "Absence of venous beading."),
    ("CB23", "T3: Sev", 3, "Clinical", "showing / showing no", "showing", "showing no", "Fundus photograph showing intraretinal microvascular abnormalities.", "Fundus photograph showing no intraretinal microvascular abnormalities."),
    ("CB24", "T3: Sev", 3, "Clinical", "reveals / reveals no", "reveals", "reveals no", "Retinal examination reveals widespread severe retinal haemorrhages across four quadrants.", "Retinal examination reveals no widespread severe retinal haemorrhages across four quadrants."),
    ("CB25", "T3: Sev", 3, "Clinical", "exhibits / exhibits no", "exhibits", "exhibits no", "The retina exhibits venous beading and intraretinal microvascular abnormalities.", "The retina exhibits no venous beading and intraretinal microvascular abnormalities."),
    ("CB26", "T3: Sev", 3, "Lay", "with / without", "with", "without", "Back of the eye with heavy red fluid in all four sections.", "Back of the eye without heavy red fluid in all four sections."),
    ("CB27", "T3: Sev", 3, "Lay", "Presence of / Absence of", "Presence of", "Absence of", "Presence of lumpy veins shaped like a string of beads.", "Absence of lumpy veins shaped like a string of beads."),
    ("CB28", "T3: Sev", 3, "Lay", "showing / showing no", "showing", "showing no", "Eye photograph showing abnormal looping tiny blood paths inside the eye wall.", "Eye photograph showing no abnormal looping tiny blood paths inside the eye wall."),
    ("CB29", "T3: Sev", 3, "Lay", "reveals / reveals no", "reveals", "reveals no", "Eye examination reveals widespread heavy blood patches in every corner.", "Eye examination reveals no widespread heavy blood patches in every corner."),
    ("CB30", "T3: Sev", 3, "Lay", "exhibits / exhibits no", "exhibits", "exhibits no", "The back of the eye exhibits pinched and swollen veins.", "The back of the eye exhibits no pinched and swollen veins."),
    # T4: Prol (g_min = 4)
    ("CB31", "T4: Prol", 4, "Clinical", "with / without", "with", "without", "Fundus with neovascularization at the optic disc.", "Fundus without neovascularization at the optic disc."),
    ("CB32", "T4: Prol", 4, "Clinical", "Presence of / Absence of", "Presence of", "Absence of", "Presence of retinal neovascularization.", "Absence of retinal neovascularization."),
    ("CB33", "T4: Prol", 4, "Clinical", "showing / showing no", "showing", "showing no", "Fundus photograph showing vitreous haemorrhage.", "Fundus photograph showing no vitreous haemorrhage."),
    ("CB34", "T4: Prol", 4, "Clinical", "reveals / reveals no", "reveals", "reveals no", "Retinal examination reveals preretinal haemorrhage.", "Retinal examination reveals no preretinal haemorrhage."),
    ("CB35", "T4: Prol", 4, "Clinical", "exhibits / exhibits no", "exhibits", "exhibits no", "The retina exhibits preretinal neovascularization elsewhere.", "The retina exhibits no preretinal neovascularization elsewhere."),
    ("CB36", "T4: Prol", 4, "Lay", "with / without", "with", "without", "Back of the eye with new fragile blood vessels growing on the main nerve.", "Back of the eye without new fragile blood vessels growing on the main nerve."),
    ("CB37", "T4: Prol", 4, "Lay", "Presence of / Absence of", "Presence of", "Absence of", "Presence of abnormal new blood vessel growth.", "Absence of abnormal new blood vessel growth."),
    ("CB38", "T4: Prol", 4, "Lay", "showing / showing no", "showing", "showing no", "Eye photograph showing leaking blood inside the clear jelly of the eye.", "Eye photograph showing no leaking blood inside the clear jelly of the eye."),
    ("CB39", "T4: Prol", 4, "Lay", "reveals / reveals no", "reveals", "reveals no", "Eye examination reveals fresh tangles of weak new blood vessels.", "Eye examination reveals no fresh tangles of weak new blood vessels."),
    ("CB40", "T4: Prol", 4, "Lay", "exhibits / exhibits no", "exhibits", "exhibits no", "The back of the eye exhibits leaky new blood vessels branching on the surface.", "The back of the eye exhibits no leaky new blood vessels branching on the surface."),
    # T5: Holistic (g_min varies 1, 2, 4)
    ("CB41", "T5: Holistic", 1, "Clinical", "with / without", "with", "without", "Fundus with diabetic retinopathy.", "Fundus without diabetic retinopathy."),
    ("CB42", "T5: Holistic", 2, "Clinical", "Presence of / Absence of", "Presence of", "Absence of", "Presence of referable diabetic retinopathy lesions.", "Absence of referable diabetic retinopathy lesions."),
    ("CB43", "T5: Holistic", 4, "Clinical", "showing / showing no", "showing", "showing no", "Fundus photograph showing proliferative diabetic retinopathy.", "Fundus photograph showing no proliferative diabetic retinopathy."),
    ("CB44", "T5: Holistic", 2, "Clinical", "reveals / reveals no", "reveals", "reveals no", "Retinal examination reveals clinically significant retinal exudates.", "Retinal examination reveals no clinically significant retinal exudates."),
    ("CB45", "T5: Holistic", 1, "Clinical", "exhibits / exhibits no", "exhibits", "exhibits no", "The retina exhibits diabetic microvascular lesions.", "The retina exhibits no diabetic microvascular lesions."),
    ("CB46", "T5: Holistic", 1, "Lay", "with / without", "with", "without", "Back of the eye with unhealthy blood vessel changes.", "Back of the eye without unhealthy blood vessel changes."),
    ("CB47", "T5: Holistic", 2, "Lay", "Presence of / Absence of", "Presence of", "Absence of", "Presence of concerning fluid leakage and red blood spots.", "Absence of concerning fluid leakage and red blood spots."),
    ("CB48", "T5: Holistic", 4, "Lay", "showing / showing no", "showing", "showing no", "Eye photograph showing dangerous new blood vessel growth.", "Eye photograph showing no dangerous new blood vessel growth."),
    ("CB49", "T5: Holistic", 2, "Lay", "reveals / reveals no", "reveals", "reveals no", "Eye examination reveals abnormal white and yellow fluid spots.", "Eye examination reveals no abnormal white and yellow fluid spots."),
    ("CB50", "T5: Holistic", 1, "Lay", "exhibits / exhibits no", "exhibits", "exhibits no", "The back of the eye exhibits visible blood or fluid marks.", "The back of the eye exhibits no visible blood or fluid marks."),
]


def build_and_validate_pairs(tokenizer: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Construct pairs list and run all strict pre-inference invariants."""
    pairs = []
    max_raw_tok = 0
    max_wrap_tok = 0

    for pid, tier, g_min, reg, op_label, op_aff, op_neg, aff, neg in RAW_CONTROL_B_SPEC:
        # Strict minimal-pair check: operator appears exactly once and replacement is character-exact
        assert aff.count(op_aff) == 1, f"{pid}: affirmative operator '{op_aff}' count != 1 in '{aff}'"
        assert neg.count(op_neg) == 1, f"{pid}: negated operator '{op_neg}' count != 1 in '{neg}'"
        reconstructed_neg = aff.replace(op_aff, op_neg, 1)
        assert reconstructed_neg == neg, f"{pid}: non-minimal diff '{reconstructed_neg}' != '{neg}'"
        stripped_aff = aff.replace(op_aff, "", 1)
        stripped_neg = neg.replace(op_neg, "", 1)
        assert stripped_aff == stripped_neg, f"{pid}: stripped diff '{stripped_aff}' != '{stripped_neg}'"

        for txt in (aff, neg):
            raw_ids = tokenizer(txt, truncation=False, add_special_tokens=True)["input_ids"]
            wrap_txt = CAPTION_TEMPLATE.replace("[CLS]", txt)
            wrap_ids = tokenizer(wrap_txt, truncation=False, add_special_tokens=True)["input_ids"]
            max_raw_tok = max(max_raw_tok, len(raw_ids))
            max_wrap_tok = max(max_wrap_tok, len(wrap_ids))
            assert len(raw_ids) < 77, f"{pid} raw token len {len(raw_ids)} >= 77"
            assert len(wrap_ids) < 77, f"{pid} wrap token len {len(wrap_ids)} >= 77"

        pairs.append({
            "pair_id": pid,
            "tier": tier,
            "g_min": g_min,
            "register": reg,
            "operator": op_label,
            "op_aff": op_aff,
            "op_neg": op_neg,
            "affirmative_prompt": aff,
            "negated_prompt": neg,
        })

    assert len(pairs) == 50
    assert sum(1 for p in pairs if p["register"] == "Clinical") == 25
    assert sum(1 for p in pairs if p["register"] == "Lay") == 25
    tiers = pd.Series([p["tier"] for p in pairs]).value_counts().to_dict()
    ops = pd.Series([p["operator"] for p in pairs]).value_counts().to_dict()
    assert all(v == 10 for v in tiers.values()) and len(tiers) == 5
    assert all(v == 10 for v in ops.values()) and len(ops) == 5

    invariants = {
        "n_pairs": 50,
        "clinical_count": 25,
        "lay_count": 25,
        "tier_counts": tiers,
        "operator_counts": ops,
        "all_operator_only_diffs_exact": True,
        "max_raw_tokens": max_raw_tok,
        "max_wrapped_tokens": max_wrap_tok,
        "tokenizer_limit": 77,
        "zero_truncation": True,
    }
    return pairs, invariants


def encode_single_strings(model: Any, texts: Sequence[str], device: torch.device) -> np.ndarray:
    """Encode strings individually (batch_size=1, zero [PAD]) -> L2-normalized (N, 512)."""
    embs = []
    with torch.no_grad():
        for txt in texts:
            toks = model.text_model.tokenizer([txt], truncation=True, padding=True, return_tensors="pt")
            input_ids = toks["input_ids"].to(device).to(torch.long)
            attention_mask = toks["attention_mask"].to(device).to(torch.long)
            emb = model.text_model(input_ids, attention_mask)
            embs.append(emb.squeeze(0).detach().cpu().numpy())
    arr = np.vstack(embs).astype(np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / norms
    assert np.max(np.abs(np.linalg.norm(arr, axis=1) - 1.0)) < 1e-6
    return arr


def bootstrap_mean_ci(per_image_vals: np.ndarray, boot_idx: np.ndarray) -> Dict[str, Any]:
    """Compute mean, SD (over images), and B-resample 95% CI from per-image values."""
    vals = per_image_vals[~np.isnan(per_image_vals)]
    point = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1))
    # Resample within valid image rows
    n = len(vals)
    if n == len(per_image_vals):
        boot_means = np.mean(vals[boot_idx], axis=1)
    else:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        b_idx = rng.integers(0, n, size=(boot_idx.shape[0], n))
        boot_means = np.mean(vals[b_idx], axis=1)
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    return {
        "n_images": int(n),
        "mean": point,
        "sd": sd,
        "ci95_low": float(ci_lo),
        "ci95_high": float(ci_hi),
    }


def safe_wilcoxon(x: np.ndarray) -> float:
    x_clean = x[~np.isnan(x)]
    if len(x_clean) == 0 or np.allclose(x_clean, 0.0):
        return 1.0
    _, p = wilcoxon(x_clean)
    return float(p)


def analyze_protocol(
    protocol_name: str,
    model: Any,
    device: torch.device,
    vis_unit: np.ndarray,
    y_true: np.ndarray,
    logit_scale: float,
    pairs: List[Dict[str, Any]],
    txt_meta_custom: pd.DataFrame,
    use_wrapper: bool,
) -> Dict[str, Any]:
    """Run full Control B inference and metrics under raw_single_string or wrapped_caption."""
    aff_texts = [p["affirmative_prompt"] for p in pairs]
    neg_texts = [p["negated_prompt"] for p in pairs]
    if use_wrapper:
        aff_texts_enc = [CAPTION_TEMPLATE.replace("[CLS]", t) for t in aff_texts]
        neg_texts_enc = [CAPTION_TEMPLATE.replace("[CLS]", t) for t in neg_texts]
    else:
        aff_texts_enc = aff_texts
        neg_texts_enc = neg_texts

    emb_aff = encode_single_strings(model, aff_texts_enc, device)  # (50, 512)
    emb_neg = encode_single_strings(model, neg_texts_enc, device)  # (50, 512)

    S_aff = np.dot(vis_unit, emb_aff.T)  # (1744, 50)
    S_neg = np.dot(vis_unit, emb_neg.T)  # (1744, 50)
    assert S_aff.shape == (1744, 50) and S_neg.shape == (1744, 50)
    delta_S = S_neg - S_aff  # (1744, 50)

    # Check floating-point symmetry invariant
    assert abs(float(np.mean(S_neg) - np.mean(S_aff)) - float(np.mean(delta_S))) < 1e-12

    g_min_arr = np.array([p["g_min"] for p in pairs], dtype=int)  # (50,)
    registers = np.array([p["register"] for p in pairs])
    tiers = np.array([p["tier"] for p in pairs])
    operators = np.array([p["operator"] for p in pairs])

    # Pre-generate image-level bootstrap indices for full 1744, noDR (1017), and anyDR (727)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx_all = np.arange(len(y_true))
    idx_nodr = np.where(y_true == 0)[0]
    idx_anydr = np.where(y_true >= 1)[0]
    boot_all = rng.integers(0, len(idx_all), size=(BOOTSTRAP_B, len(idx_all)))
    boot_nodr = rng.integers(0, len(idx_nodr), size=(BOOTSTRAP_B, len(idx_nodr)))
    boot_anydr = rng.integers(0, len(idx_anydr), size=(BOOTSTRAP_B, len(idx_anydr)))

    # 1. r_text and r_score per pair
    r_text = np.sum(emb_aff * emb_neg, axis=1)  # (50,)
    assert np.all((r_text >= -1.000001) & (r_text <= 1.000001))
    r_score = np.array([pearsonr(S_aff[:, k], S_neg[:, k])[0] for k in range(50)], dtype=np.float64)

    def summarize_pair_vector(vec: np.ndarray, mask: np.ndarray | None = None) -> Dict[str, Any]:
        v = vec if mask is None else vec[mask]
        return {
            "n_pairs": int(len(v)),
            "mean": float(np.mean(v)),
            "sd": float(np.std(v, ddof=1)),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
        }

    # 2. Delta S summaries across subsets of pairs
    def eval_delta_s_subset(pair_mask: np.ndarray) -> Dict[str, Any]:
        dS_sub = delta_S[:, pair_mask]  # (1744, K)
        g_sub = g_min_arr[pair_mask]    # (K,)
        S_aff_sub = S_aff[:, pair_mask]
        S_neg_sub = S_neg[:, pair_mask]

        # Image-level means
        img_mean_all = np.mean(dS_sub, axis=1)
        img_mean_nodr = img_mean_all[idx_nodr]
        img_mean_anydr = img_mean_all[idx_anydr]

        # Tier-conditioned masks (1744, K)
        absent_mask = y_true[:, None] < g_sub[None, :]
        present_mask = y_true[:, None] >= g_sub[None, :]

        # Per-image mean over absent / present pairs for that image (NaN if image has 0 pairs in that state)
        img_absent = np.array([
            np.mean(dS_sub[i, absent_mask[i]]) if np.any(absent_mask[i]) else np.nan
            for i in range(len(y_true))
        ])
        img_present = np.array([
            np.mean(dS_sub[i, present_mask[i]]) if np.any(present_mask[i]) else np.nan
            for i in range(len(y_true))
        ])

        # Discriminative interaction DeltaDelta S = E[Delta S | noDR] - E[Delta S | y >= g_min]
        # Bootstrap jointly over all 1744 images (keeping image clusters intact)
        boot_dd_nodr_vs_present = []
        boot_dd_absent_vs_present = []
        for b in range(BOOTSTRAP_B):
            bi = boot_all[b]
            m_nodr = float(np.nanmean(img_mean_all[bi][y_true[bi] == 0]))
            m_pres = float(np.nanmean(img_present[bi]))
            m_abs = float(np.nanmean(img_absent[bi]))
            boot_dd_nodr_vs_present.append(m_nodr - m_pres)
            boot_dd_absent_vs_present.append(m_abs - m_pres)

        dd_nodr_pt = float(np.mean(img_mean_nodr)) - float(np.nanmean(img_present))
        dd_abs_pt = float(np.nanmean(img_absent)) - float(np.nanmean(img_present))

        # Pairwise 2-way FPR/FNR
        aff_gt_neg = (S_aff_sub > S_neg_sub).astype(float)
        neg_gt_aff = (S_neg_sub > S_aff_sub).astype(float)
        fpr_2way_nodr_per_img = np.mean(aff_gt_neg[idx_nodr], axis=1)
        fpr_2way_absent_per_img = np.array([
            np.mean(aff_gt_neg[i, absent_mask[i]]) if np.any(absent_mask[i]) else np.nan
            for i in range(len(y_true))
        ])
        fnr_2way_present_per_img = np.array([
            np.mean(neg_gt_aff[i, present_mask[i]]) if np.any(present_mask[i]) else np.nan
            for i in range(len(y_true))
        ])

        fpr_abs_stat = bootstrap_mean_ci(fpr_2way_absent_per_img, boot_all)
        fnr_pres_stat = bootstrap_mean_ci(fnr_2way_present_per_img, boot_all)
        bal_acc_2way = 0.5 * ((1.0 - fpr_abs_stat["mean"]) + (1.0 - fnr_pres_stat["mean"]))

        return {
            "n_pairs": int(np.sum(pair_mask)),
            "delta_s_all_images": {
                **bootstrap_mean_ci(img_mean_all, boot_all),
                "wilcoxon_p": safe_wilcoxon(img_mean_all),
            },
            "delta_s_nodr": {
                **bootstrap_mean_ci(img_mean_nodr, boot_nodr),
                "wilcoxon_p": safe_wilcoxon(img_mean_nodr),
            },
            "delta_s_anydr": {
                **bootstrap_mean_ci(img_mean_anydr, boot_anydr),
                "wilcoxon_p": safe_wilcoxon(img_mean_anydr),
            },
            "delta_s_lesion_absent_tier": {
                **bootstrap_mean_ci(img_absent, boot_all),
                "wilcoxon_p": safe_wilcoxon(img_absent),
            },
            "delta_s_lesion_present_tier": {
                **bootstrap_mean_ci(img_present, boot_all),
                "wilcoxon_p": safe_wilcoxon(img_present),
            },
            "delta_delta_s_nodr_minus_present": {
                "mean": dd_nodr_pt,
                "ci95_low": float(np.percentile(boot_dd_nodr_vs_present, 2.5)),
                "ci95_high": float(np.percentile(boot_dd_nodr_vs_present, 97.5)),
            },
            "delta_delta_s_absent_minus_present": {
                "mean": dd_abs_pt,
                "ci95_low": float(np.percentile(boot_dd_absent_vs_present, 2.5)),
                "ci95_high": float(np.percentile(boot_dd_absent_vs_present, 97.5)),
            },
            "pairwise_2way_fpr_nodr": bootstrap_mean_ci(fpr_2way_nodr_per_img, boot_nodr),
            "pairwise_2way_fpr_absent_tier": fpr_abs_stat,
            "pairwise_2way_fnr_present_tier": fnr_pres_stat,
            "pairwise_2way_balanced_accuracy": float(bal_acc_2way),
        }

    overall_mask = np.ones(50, dtype=bool)
    by_register = {
        reg: eval_delta_s_subset(registers == reg) for reg in ["Clinical", "Lay"]
    }
    by_tier = {
        t: eval_delta_s_subset(tiers == t) for t in ["T1: Mild", "T2: Mod", "T3: Sev", "T4: Prol", "T5: Holistic"]
    }
    by_operator = {
        op: eval_delta_s_subset(operators == op)
        for op in [
            "with / without",
            "Presence of / Absence of",
            "showing / showing no",
            "reveals / reveals no",
            "exhibits / exhibits no",
        ]
    }

    # 3. Neutral-anchor binary FPR shift
    anchors = {
        "primary_no_dr": "no diabetic retinopathy",
        "sensitivity_healthy_retina": "normal healthy retina",
    }
    anchor_results = {}
    for a_key, a_str in anchors.items():
        a_enc = CAPTION_TEMPLATE.replace("[CLS]", a_str) if use_wrapper else a_str
        emb_a = encode_single_strings(model, [a_enc], device)[0]  # (512,)
        S_a = np.dot(vis_unit, emb_a)  # (1744,)

        fpr_aff_nodr_mat = (S_aff[idx_nodr] > S_a[idx_nodr, None]).astype(float)
        fpr_neg_nodr_mat = (S_neg[idx_nodr] > S_a[idx_nodr, None]).astype(float)
        fpr_aff_img = np.mean(fpr_aff_nodr_mat, axis=1)
        fpr_neg_img = np.mean(fpr_neg_nodr_mat, axis=1)
        dfpr_img = fpr_neg_img - fpr_aff_img

        # Also split Clinical vs Lay
        clin_mask = registers == "Clinical"
        lay_mask = registers == "Lay"

        absent_mask = y_true[:, None] < g_min_arr[None, :]
        fpr_aff_abs_img = np.array([
            np.mean((S_aff[i, absent_mask[i]] > S_a[i]).astype(float)) if np.any(absent_mask[i]) else np.nan
            for i in range(len(y_true))
        ])
        fpr_neg_abs_img = np.array([
            np.mean((S_neg[i, absent_mask[i]] > S_a[i]).astype(float)) if np.any(absent_mask[i]) else np.nan
            for i in range(len(y_true))
        ])
        dfpr_abs_img = fpr_neg_abs_img - fpr_aff_abs_img

        anchor_results[a_key] = {
            "anchor_text": a_enc,
            "nodr_overall": {
                "fpr_aff": bootstrap_mean_ci(fpr_aff_img, boot_nodr),
                "fpr_neg": bootstrap_mean_ci(fpr_neg_img, boot_nodr),
                "delta_fpr": {
                    **bootstrap_mean_ci(dfpr_img, boot_nodr),
                    "wilcoxon_p": safe_wilcoxon(dfpr_img),
                },
            },
            "nodr_clinical": {
                "fpr_aff": bootstrap_mean_ci(np.mean(fpr_aff_nodr_mat[:, clin_mask], axis=1), boot_nodr),
                "fpr_neg": bootstrap_mean_ci(np.mean(fpr_neg_nodr_mat[:, clin_mask], axis=1), boot_nodr),
                "delta_fpr": bootstrap_mean_ci(
                    np.mean(fpr_neg_nodr_mat[:, clin_mask] - fpr_aff_nodr_mat[:, clin_mask], axis=1), boot_nodr
                ),
            },
            "nodr_lay": {
                "fpr_aff": bootstrap_mean_ci(np.mean(fpr_aff_nodr_mat[:, lay_mask], axis=1), boot_nodr),
                "fpr_neg": bootstrap_mean_ci(np.mean(fpr_neg_nodr_mat[:, lay_mask], axis=1), boot_nodr),
                "delta_fpr": bootstrap_mean_ci(
                    np.mean(fpr_neg_nodr_mat[:, lay_mask] - fpr_aff_nodr_mat[:, lay_mask], axis=1), boot_nodr
                ),
            },
            "absent_tier_overall": {
                "fpr_aff": bootstrap_mean_ci(fpr_aff_abs_img, boot_all),
                "fpr_neg": bootstrap_mean_ci(fpr_neg_abs_img, boot_all),
                "delta_fpr": {
                    **bootstrap_mean_ci(dfpr_abs_img, boot_all),
                    "wilcoxon_p": safe_wilcoxon(dfpr_abs_img),
                },
            },
        }

    # 4. 5-Way Paired Grade-Prompt Swap FPR Shift
    txt_meta_work = txt_meta_custom.copy()
    txt_meta_work["numerical_grade"] = txt_meta_work["Grade"].map(GRADE_TO_IDX)

    # Baseline 1: T_label Variation 1
    tl_v1_texts = txt_meta_work[
        (txt_meta_work["Family"] == "T_label") & (txt_meta_work["Variation_Number"] == 1)
    ].sort_values("numerical_grade")["Prompt_Text"].tolist()
    if use_wrapper:
        tl_v1_texts = [CAPTION_TEMPLATE.replace("[CLS]", t) for t in tl_v1_texts]
    base_v1 = encode_single_strings(model, tl_v1_texts, device)  # (5, 512)

    # Baseline 2: T_label 10-variant centroid (L2-renormalized)
    base_centroid_list = []
    for g_idx in range(5):
        g_texts = txt_meta_work[
            (txt_meta_work["Family"] == "T_label") & (txt_meta_work["numerical_grade"] == g_idx)
        ].sort_values("Variation_Number")["Prompt_Text"].tolist()
        if use_wrapper:
            g_texts = [CAPTION_TEMPLATE.replace("[CLS]", t) for t in g_texts]
        g_embs = encode_single_strings(model, g_texts, device)
        c = np.mean(g_embs, axis=0)
        c = c / np.linalg.norm(c)
        base_centroid_list.append(c)
    base_centroid = np.vstack(base_centroid_list)  # (5, 512)

    five_way_results = {}
    for b_name, base_protos in [("tlabel_v1", base_v1), ("tlabel_10var_centroid", base_centroid)]:
        base_S = np.dot(vis_unit, base_protos.T)  # (1744, 5)
        # Note: softmax argmax == cosine argmax since logit_scale > 0 is monotonic
        fpr_aff_nodr_all = np.zeros((len(idx_nodr), 50), dtype=float)
        fpr_neg_nodr_all = np.zeros((len(idx_nodr), 50), dtype=float)

        for k in range(50):
            slot = g_min_arr[k]
            S_swap_aff = base_S[idx_nodr].copy()
            S_swap_aff[:, slot] = S_aff[idx_nodr, k]
            preds_aff = np.argmax(S_swap_aff, axis=1)
            fpr_aff_nodr_all[:, k] = (preds_aff == slot).astype(float)

            S_swap_neg = base_S[idx_nodr].copy()
            S_swap_neg[:, slot] = S_neg[idx_nodr, k]
            preds_neg = np.argmax(S_swap_neg, axis=1)
            fpr_neg_nodr_all[:, k] = (preds_neg == slot).astype(float)

        # Overall 50 pairs
        aff_img_all = np.mean(fpr_aff_nodr_all, axis=1)
        neg_img_all = np.mean(fpr_neg_nodr_all, axis=1)
        diff_img_all = neg_img_all - aff_img_all

        # Proliferative slot (g_min == 4: CB31..CB40 + CB43, CB48 => 12 pairs; also T4 only CB31..CB40)
        prol_all_mask = g_min_arr == 4
        t4_mask = tiers == "T4: Prol"

        five_way_results[b_name] = {
            "all_50_pairs_nodr_to_slot": {
                "fpr_aff": bootstrap_mean_ci(aff_img_all, boot_nodr),
                "fpr_neg": bootstrap_mean_ci(neg_img_all, boot_nodr),
                "delta_fpr": {
                    **bootstrap_mean_ci(diff_img_all, boot_nodr),
                    "wilcoxon_p": safe_wilcoxon(diff_img_all),
                },
            },
            "prol_slot_all_12_pairs_nodr_to_prolDR": {
                "n_pairs": int(np.sum(prol_all_mask)),
                "fpr_aff": bootstrap_mean_ci(np.mean(fpr_aff_nodr_all[:, prol_all_mask], axis=1), boot_nodr),
                "fpr_neg": bootstrap_mean_ci(np.mean(fpr_neg_nodr_all[:, prol_all_mask], axis=1), boot_nodr),
                "delta_fpr": bootstrap_mean_ci(
                    np.mean(fpr_neg_nodr_all[:, prol_all_mask] - fpr_aff_nodr_all[:, prol_all_mask], axis=1),
                    boot_nodr,
                ),
            },
            "t4_prol_10_pairs_nodr_to_prolDR": {
                "n_pairs": int(np.sum(t4_mask)),
                "fpr_aff": bootstrap_mean_ci(np.mean(fpr_aff_nodr_all[:, t4_mask], axis=1), boot_nodr),
                "fpr_neg": bootstrap_mean_ci(np.mean(fpr_neg_nodr_all[:, t4_mask], axis=1), boot_nodr),
                "delta_fpr": bootstrap_mean_ci(
                    np.mean(fpr_neg_nodr_all[:, t4_mask] - fpr_aff_nodr_all[:, t4_mask], axis=1),
                    boot_nodr,
                ),
            },
        }

    # Per-pair summary (compact, helpful for inspection & supplement)
    per_pair_metrics = []
    for k, p in enumerate(pairs):
        nodr_d = delta_S[idx_nodr, k]
        pres_mask_k = y_true >= p["g_min"]
        pres_d = delta_S[pres_mask_k, k]
        per_pair_metrics.append({
            "pair_id": p["pair_id"],
            "tier": p["tier"],
            "g_min": p["g_min"],
            "register": p["register"],
            "operator": p["operator"],
            "r_text": float(r_text[k]),
            "r_score": float(r_score[k]),
            "delta_s_nodr_mean": float(np.mean(nodr_d)),
            "delta_s_present_mean": float(np.mean(pres_d)),
            "delta_delta_s": float(np.mean(nodr_d) - np.mean(pres_d)),
            "pairwise_2way_fpr_nodr": float(np.mean(S_aff[idx_nodr, k] > S_neg[idx_nodr, k])),
            "pairwise_2way_fnr_present": float(np.mean(S_neg[pres_mask_k, k] > S_aff[pres_mask_k, k])),
        })

    return {
        "protocol": protocol_name,
        "caption_wrapped": use_wrapper,
        "r_text": {
            "all_50": summarize_pair_vector(r_text),
            "clinical_25": summarize_pair_vector(r_text, registers == "Clinical"),
            "lay_25": summarize_pair_vector(r_text, registers == "Lay"),
        },
        "r_score": {
            "all_50": summarize_pair_vector(r_score),
            "clinical_25": summarize_pair_vector(r_score, registers == "Clinical"),
            "lay_25": summarize_pair_vector(r_score, registers == "Lay"),
        },
        "overall": eval_delta_s_subset(overall_mask),
        "by_register": by_register,
        "by_tier": by_tier,
        "by_operator": by_operator,
        "neutral_anchor_fpr": anchor_results,
        "five_way_grade_swap_fpr": five_way_results,
        "per_pair": per_pair_metrics,
    }


def main() -> Dict[str, Any]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Control B] Using device: {device}")

    from flair.modeling.model import FLAIRModel

    model = FLAIRModel.from_pretrained("jusiro2/FLAIR", local_files_only=LOCAL_FILES_ONLY).eval().to(device)
    pairs, invariants = build_and_validate_pairs(model.text_model.tokenizer)
    print(
        f"[Control B] Pre-inference invariants PASS: {invariants['n_pairs']} pairs, "
        f"max_wrapped_tokens={invariants['max_wrapped_tokens']} (< 77)"
    )

    vis_embeds_raw = np.load(WEEK4_DIR / "cached_vision_embeddings_1744.npy")
    vis_unit = (vis_embeds_raw / np.linalg.norm(vis_embeds_raw, axis=1, keepdims=True)).astype(np.float64)
    vis_meta = pd.read_csv(WEEK4_DIR / "cached_vision_metadata_1744.csv")
    txt_meta_custom = pd.read_csv(WEEK4_DIR / "cached_text_metadata_custom.csv")
    logit_scale = float(np.load(WEEK4_DIR / "flair_logit_scale.npy").item())
    y_true = vis_meta["adjudicated_dr_grade"].astype(int).values

    # Save control_b_pairs.json
    pairs_payload = {
        "metadata": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "design_note": "Strictly a priori 50 matched minimal negation pairs; zero test-set tuning.",
            "invariants": invariants,
        },
        "pairs": pairs,
    }
    pairs_path = ARTIFACTS_DIR / "control_b_pairs.json"
    with open(pairs_path, "w", encoding="utf-8") as f:
        json.dump(pairs_payload, f, indent=2)

    raw_res = analyze_protocol(
        "raw_single_string", model, device, vis_unit, y_true, logit_scale, pairs, txt_meta_custom, use_wrapper=False
    )
    wrap_res = analyze_protocol(
        "wrapped_caption", model, device, vis_unit, y_true, logit_scale, pairs, txt_meta_custom, use_wrapper=True
    )

    for pname, res in [("raw_single_string", raw_res), ("wrapped_caption", wrap_res)]:
        ov = res["overall"]
        print(
            f"  [{pname}] r_text={res['r_text']['all_50']['mean']:.4f}, "
            f"r_score={res['r_score']['all_50']['mean']:.4f}, "
            f"DeltaS(noDR)={ov['delta_s_nodr']['mean']:.4f}, "
            f"DeltaS(present)={ov['delta_s_lesion_present_tier']['mean']:.4f}, "
            f"DeltaDeltaS={ov['delta_delta_s_nodr_minus_present']['mean']:.4f}, "
            f"2way_FPR(noDR)={ov['pairwise_2way_fpr_nodr']['mean']*100:.2f}%, "
            f"Anchor_FPR_aff={res['neutral_anchor_fpr']['primary_no_dr']['nodr_overall']['fpr_aff']['mean']*100:.2f}% -> "
            f"neg={res['neutral_anchor_fpr']['primary_no_dr']['nodr_overall']['fpr_neg']['mean']*100:.2f}%"
        )

    output = {
        "metadata": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "n_images": 1744,
            "n_nodr": int(np.sum(y_true == 0)),
            "n_anydr": int(np.sum(y_true >= 1)),
            "bootstrap_b": BOOTSTRAP_B,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "invariants": invariants,
        },
        "raw_single_string": raw_res,
        "wrapped_caption": wrap_res,
    }

    out_path = ARTIFACTS_DIR / "control_b_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"[Control B COMPLETE] Wrote {pairs_path} and {out_path}")
    return output


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Control B: 50 matched minimal-pair negation ablation.")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--cache-dir", type=Path, default=WEEK4_DIR)
    p.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    DATA_DIR = args.data_dir
    WEEK4_DIR = args.cache_dir
    ARTIFACTS_DIR = args.artifacts_dir
    LOCAL_FILES_ONLY = args.local_files_only
    main()
