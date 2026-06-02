import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUT_DIR = "../results/"

def calculate_binned_metrics(confidences, accuracies, num_bins=10):
    bin_bounds = np.linspace(0, 1, num_bins + 1)
    bin_accs, bin_confs = [], []
    for i in range(num_bins):
        if i == num_bins - 1:
            in_bin = (confidences >= bin_bounds[i]) & (confidences <= bin_bounds[i+1])
        else:
            in_bin = (confidences >= bin_bounds[i]) & (confidences < bin_bounds[i+1])
            
        if np.sum(in_bin) > 0:
            bin_accs.append(np.mean(accuracies[in_bin]))
            bin_confs.append(np.mean(confidences[in_bin]))
        else:
            bin_accs.append(np.nan)
            bin_confs.append(np.nan)
    return np.array(bin_accs), np.array(bin_confs)

def generate_reliability_diagrams():
    print("Generating Publication-Quality Reliability Diagrams...")
    vis_embeds = np.load(f"{OUT_DIR}cached_vision_embeddings_1744.npy")
    txt_embeds = np.load(f"{OUT_DIR}cached_text_embeddings_custom.npy")
    logit_scale = np.load(f"{OUT_DIR}flair_logit_scale.npy").item()
    
    vis_meta = pd.read_csv(f"{OUT_DIR}cached_vision_metadata_1744.csv")
    txt_meta = pd.read_csv(f"{OUT_DIR}cached_text_metadata_custom.csv")
    
    vis_embeds = vis_embeds / np.linalg.norm(vis_embeds, axis=1, keepdims=True)
    txt_embeds = txt_embeds / np.linalg.norm(txt_embeds, axis=1, keepdims=True)
    
    txt_meta['numerical_grade'] = txt_meta['Grade'].map({'noDR':0, 'mildDR':1, 'modDR':2, 'sevDR':3, 'prolDR':4})
    true_labels = vis_meta['adjudicated_dr_grade'].values
    
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), sharex=True, sharey=True)
    axes = axes.flatten()
    
    families = ['T_label', 'T_expert', 'T_layman', 'T_exclusion']
    titles = ['T_label (Baseline Control)', 'T_expert (Clinical Phrasing)', 
              'T_layman (Patient Phrasing)', 'T_exclusion (Negative Constraints)']

    for idx, family in enumerate(families):
        family_accs, family_confs = [], []
        for var_num in range(1, 11):
            var_mask = (txt_meta['Family'] == family) & (txt_meta['Variation_Number'] == var_num)
            var_meta = txt_meta[var_mask].sort_values('numerical_grade')
            if len(var_meta) != 5: continue
                
            scaled_logits = np.dot(vis_embeds, txt_embeds[var_meta.index.values].T) * logit_scale
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            family_accs.extend((np.argmax(probs, axis=1) == true_labels).astype(float))
            family_confs.extend(np.max(probs, axis=1))
            
        bin_accs, bin_confs = calculate_binned_metrics(np.array(family_confs), np.array(family_accs))
        
        ax = axes[idx]
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
        ax.plot(bin_confs, bin_accs, marker='o', linewidth=3, markersize=8, 
                color='crimson' if family != 'T_label' else 'teal', label='Model Output')
        
        ax.set_title(titles[idx], fontweight='bold', pad=15)
        ax.set_xlim([0.0, 1.05])
        ax.set_ylim([0.0, 1.05])
        
        if idx >= 2: ax.set_xlabel('Predicted Confidence')
        if idx % 2 == 0: ax.set_ylabel('True Accuracy')
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='lightgray')
        textstr = f'Accuracy: {np.mean(family_accs)*100:.1f}%\nAvg Conf: {np.mean(family_confs)*100:.1f}%'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
        ax.legend(loc="lower right")

    plt.suptitle("Impact of Lexical Variation on Model Calibration (MESSIDOR-2)", fontsize=22, y=0.98, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}thesis_calibration_curves.png", dpi=300, bbox_inches='tight')
    print(f"Success! Plot saved to {OUT_DIR}thesis_calibration_curves.png")

if __name__ == "__main__":
    generate_reliability_diagrams()
