"""Regenerate row-normalized Figure 2 (Confusion Matrices) and N=1,744 Figure 3 (Temperature Scaling KDE).

Reads cached frozen FLAIR embeddings and metadata from --cache-dir (default: ./results,
falling back to ./data) and writes 600-DPI vector PDFs to --out-dir (default: ./figures).
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax
from sklearn.metrics import confusion_matrix

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = (ROOT / "results") if (ROOT / "results" / "cached_vision_embeddings_1744.npy").exists() else (
    (ROOT / "data" / "Week 4 Work") if (ROOT / "data" / "Week 4 Work").exists() else (ROOT / "data")
)
OUT_DIR = ROOT / "figures"


def regenerate_figure_2() -> None:
    """Regenerate Figure 2: Row-normalized Aggregated Confusion Matrices (N = 1,744)."""
    plt.rcParams.update({"pdf.fonttype": 42})
    sns.set_context("talk")
    sns.set_style("white")

    if (DATA_DIR / "cached_vision_embeddings_1744.npy").exists():
        vision_emb = np.load(DATA_DIR / "cached_vision_embeddings_1744.npy")
        vision_meta = pd.read_csv(DATA_DIR / "cached_vision_metadata_1744.csv")
    else:
        vision_emb = np.load(DATA_DIR / "cached_vision_embeddings.npy")
        vision_meta = pd.read_csv(DATA_DIR / "cached_vision_metadata.csv")
        valid_idx = vision_meta["adjudicated_dr_grade"].notna()
        vision_emb = vision_emb[valid_idx]
        vision_meta = vision_meta[valid_idx].reset_index(drop=True)
    text_emb = np.load(DATA_DIR / "cached_text_embeddings_custom.npy")
    logit_scale = float(np.load(DATA_DIR / "flair_logit_scale.npy").item())
    text_meta = pd.read_csv(DATA_DIR / "cached_text_metadata_custom.csv")
    y_true = vision_meta["adjudicated_dr_grade"].astype(int).values
    assert len(y_true) == 1744, f"Expected 1,744 gradable images, got {len(y_true)}"

    v_norm = vision_emb / np.linalg.norm(vision_emb, axis=1, keepdims=True)
    families = ["T_label", "T_expert", "T_layman", "T_exclusion"]
    grade_order = ["noDR", "mildDR", "modDR", "sevDR", "prolDR"]

    fig, axes = plt.subplots(2, 2, figsize=(16.5, 14.5))
    axes = axes.flatten()

    for idx, fam in enumerate(families):
        agg_cm = np.zeros((5, 5), dtype=int)

        for var_num in range(1, 11):
            t_sub = np.zeros((5, 512))
            for g_idx, grade in enumerate(grade_order):
                mask = (
                    (text_meta["Family"] == fam)
                    & (text_meta["Variation_Number"] == var_num)
                    & (text_meta["Grade"] == grade)
                )
                t_idx = text_meta.index[mask].tolist()[0]
                t_sub[g_idx] = text_emb[t_idx]

            t_norm = t_sub / np.linalg.norm(t_sub, axis=1, keepdims=True)
            logits = (v_norm @ t_norm.T) * logit_scale
            probs = softmax(logits, axis=1)
            preds = np.argmax(probs, axis=1)  # Shape: (1744,)

            cm = confusion_matrix(y_true, preds, labels=range(5))
            agg_cm += cm

        assert agg_cm.sum() == 17440, f"{fam} matrix sum {agg_cm.sum()} != 17440"
        # Row-normalized (class-conditional) percentages
        row_sums = agg_cm.sum(axis=1, keepdims=True)
        cm_perc = (agg_cm / row_sums) * 100.0
        assert np.allclose(cm_perc.sum(axis=1), 100.0, atol=1e-9)

        if fam == "T_exclusion":
            nodr_prol_cnt = int(agg_cm[0, 4])
            nodr_total = int(row_sums[0, 0])
            nodr_prol_pct = float(cm_perc[0, 4])
            assert nodr_prol_cnt == 2444 and nodr_total == 10170
            print(
                f"[FIG 2 VERIFY] T_exclusion noDR->prolDR: {nodr_prol_cnt}/{nodr_total} "
                f"({nodr_prol_pct:.2f}%)"
            )

        annot_labels = np.empty_like(agg_cm, dtype=object)
        for i in range(5):
            for j in range(5):
                annot_labels[i, j] = f"{agg_cm[i, j]}\n({cm_perc[i, j]:.2f}%)"

        sns.heatmap(
            agg_cm,
            annot=annot_labels,
            fmt="",
            cmap="YlGnBu",
            cbar=False,
            xticklabels=grade_order,
            yticklabels=grade_order,
            ax=axes[idx],
            annot_kws={"size": 12},
        )
        axes[idx].set_title(f"Prompt Family: {fam} (Row %)", pad=14, fontweight="bold")
        axes[idx].set_xlabel("Predicted DR Grade")
        axes[idx].set_ylabel("True DR Grade")

    plt.suptitle(
        "Aggregated Confusion Matrices Across Prompt Variations (Counts & Row %)",
        y=1.01,
        fontsize=21,
        fontweight="bold",
    )
    plt.tight_layout()

    pdf_out = OUT_DIR / "new_thesis_confusion_matrices.pdf"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(pdf_out, format="pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG 2 DONE] Wrote {pdf_out} ({pdf_out.stat().st_size:,} bytes)")


def regenerate_figure_3() -> None:
    """Regenerate Figure 3: Temperature Scaling Density Shift strictly on N = 1,744."""
    plt.rcParams.update({"pdf.fonttype": 42})

    vision_emb_1744 = np.load(DATA_DIR / "cached_vision_embeddings_1744.npy")
    if (DATA_DIR / "cached_vision_embeddings.npy").exists() and (DATA_DIR / "cached_vision_metadata.csv").exists():
        vision_emb_1748 = np.load(DATA_DIR / "cached_vision_embeddings.npy")
        vision_meta_1748 = pd.read_csv(DATA_DIR / "cached_vision_metadata.csv")
        valid_idx = vision_meta_1748["adjudicated_dr_grade"].notna().values
        assert np.array_equal(vision_emb_1748[valid_idx], vision_emb_1744), (
            "cached_vision_embeddings_1744.npy mismatch vs filtered 1,748!"
        )
    assert vision_emb_1744.shape == (1744, 512)

    text_embeddings = np.load(DATA_DIR / "cached_text_embeddings_custom.npy")
    logit_scale = float(np.load(DATA_DIR / "flair_logit_scale.npy").item())
    assert text_embeddings.shape == (200, 512)

    vision_norm = vision_emb_1744 / np.linalg.norm(vision_emb_1744, axis=1, keepdims=True)
    text_norm = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)

    raw_similarity = vision_norm @ text_norm.T  # Shape: (1744, 200)
    max_raw_sim = raw_similarity.max(axis=1)

    scaled_similarity = raw_similarity * logit_scale
    probabilities = softmax(scaled_similarity, axis=1)
    max_probs = probabilities.max(axis=1)

    print(
        f"[FIG 3 VERIFY] N={len(max_raw_sim)}, tau={logit_scale:.6f}, "
        f"max_raw_sim mean={max_raw_sim.mean():.6f}, max_prob mean={max_probs.mean():.6f}"
    )

    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.kdeplot(max_raw_sim, ax=axes[0], fill=True, color="#2c7bb6", linewidth=2, cut=0)
    axes[0].set_title(
        "Raw Cosine Similarity Distribution\n(Uncalibrated Top-1 Confidence)",
        fontsize=12,
        pad=10,
    )
    axes[0].set_xlabel("Max Cosine Similarity $S(v_i, t_j)$", fontsize=11)
    axes[0].set_ylabel("Density", fontsize=11)

    sns.kdeplot(max_probs, ax=axes[1], fill=True, color="#d7191c", linewidth=2, cut=0)
    axes[1].set_title(
        "Calibrated Probability Distribution\n(Post-Scaling Softmax)",
        fontsize=12,
        pad=10,
    )
    axes[1].set_xlabel("Max Softmax Probability $P(y|v_i)$", fontsize=11)
    axes[1].set_ylabel("Density", fontsize=11)

    plt.suptitle(
        "Density Shift via Temperature Scaling in Vision-Language Latent Space",
        fontsize=14,
        y=1.05,
    )
    plt.tight_layout()

    pdf_out = OUT_DIR / "new_scaling_effect.pdf"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(pdf_out, format="pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG 3 DONE] Wrote {pdf_out} ({pdf_out.stat().st_size:,} bytes)")


def main() -> None:
    global DATA_DIR, OUT_DIR
    p = argparse.ArgumentParser(description="Regenerate audited Figures 2 and 3 from cached embeddings.")
    p.add_argument("--cache-dir", type=pathlib.Path, default=DATA_DIR)
    p.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    args = p.parse_args()
    DATA_DIR = args.cache_dir
    OUT_DIR = args.out_dir
    regenerate_figure_2()
    regenerate_figure_3()


if __name__ == "__main__":
    main()
