import numpy as np
import pandas as pd
import os
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from statsmodels.stats.contingency_tables import mcnemar

OUT_DIR = "../results/"

def calculate_ece(confidences, accuracies, num_bins=10):
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i+1] if i == num_bins - 1 else bin_boundaries[i+1]
        
        if i == num_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    return ece

def run_evaluation():
    print(">>> INITIATING THESIS MATHEMATICAL AUDIT <<<\n")
    
    vis_embeds = np.load(f"{OUT_DIR}cached_vision_embeddings_1744.npy")
    txt_embeds = np.load(f"{OUT_DIR}cached_text_embeddings_custom.npy")
    logit_scale = np.load(f"{OUT_DIR}flair_logit_scale.npy").item()
    
    vis_meta = pd.read_csv(f"{OUT_DIR}cached_vision_metadata_1744.csv")
    txt_meta = pd.read_csv(f"{OUT_DIR}cached_text_metadata_custom.csv")
    
    # L2 Normalize
    vis_embeds = vis_embeds / np.linalg.norm(vis_embeds, axis=1, keepdims=True)
    txt_embeds = txt_embeds / np.linalg.norm(txt_embeds, axis=1, keepdims=True)
    
    grade_mapping = {'noDR': 0, 'mildDR': 1, 'modDR': 2, 'sevDR': 3, 'prolDR': 4}
    txt_meta['numerical_grade'] = txt_meta['Grade'].map(grade_mapping)
    true_labels = vis_meta['adjudicated_dr_grade'].values
    families = ['T_label', 'T_expert', 'T_layman', 'T_exclusion']
    
    results = []
    family_correctness = {}

    print("1. EVALUATING ECE AND ACCURACY")
    for family in families:
        print(f"\nEvaluating Prompt Family: {family}...")
        family_accuracies, family_confidences, correctness = [], [], []
        for var_num in range(1, 11):
            var_mask = (txt_meta['Family'] == family) & (txt_meta['Variation_Number'] == var_num)
            var_meta = txt_meta[var_mask].sort_values('numerical_grade')
            if len(var_meta) != 5: continue
                
            var_txt_embeds = txt_embeds[var_meta.index.values]
            scaled_logits = np.dot(vis_embeds, var_txt_embeds.T) * logit_scale
            
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            predictions = np.argmax(probs, axis=1)
            accuracies = (predictions == true_labels).astype(float)
            
            family_accuracies.extend(accuracies)
            family_confidences.extend(np.max(probs, axis=1))
            correctness.extend(accuracies.astype(int))
            
        overall_acc = np.mean(family_accuracies)
        overall_ece = calculate_ece(np.array(family_confidences), np.array(family_accuracies))
        family_correctness[family] = np.array(correctness)
        
        results.append({'Prompt Family': family, 'Accuracy': f"{overall_acc*100:.2f}%", 'ECE': f"{overall_ece*100:.2f}%", 'Avg Confidence': f"{np.mean(family_confidences)*100:.2f}%"})
        print(f"  [{family}] Accuracy: {overall_acc*100:.2f}% | ECE: {overall_ece*100:.2f}%")

    pd.DataFrame(results).to_csv(f"{OUT_DIR}thesis_audit_results_1744.csv", index=False)

    print("\n2. QUANTIFYING THE MODALITY GAP")
    all_embeds = np.vstack([vis_embeds, txt_embeds])
    labels = np.array([0] * len(vis_embeds) + [1] * len(txt_embeds))
    print(f"[*] Silhouette Score: {silhouette_score(all_embeds, labels, metric='cosine'):.4f}")
    print(f"[*] Avg Inter-cluster Cosine Distance: {cosine_distances(vis_embeds, txt_embeds).mean():.4f}")

    print("\n3. MCNEMAR'S STATISTICAL SIGNIFICANCE TESTS")
    for f1, f2 in [('T_label', 'T_expert'), ('T_expert', 'T_layman'), ('T_layman', 'T_exclusion'), ('T_label', 'T_layman')]:
        c1, c2 = family_correctness[f1], family_correctness[f2]
        table = [[np.sum((c1==1)&(c2==1)), np.sum((c1==1)&(c2==0))], [np.sum((c1==0)&(c2==1)), np.sum((c1==0)&(c2==0))]]
        res = mcnemar(table, exact=False, correction=True)
        sig = "***" if res.pvalue < 0.001 else "**" if res.pvalue < 0.01 else "*" if res.pvalue < 0.05 else "ns"
        print(f"  {f1} vs {f2}: p-value = {res.pvalue:.4e} | {sig}")

if __name__ == "__main__":
    run_evaluation()
