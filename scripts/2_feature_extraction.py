import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from flair.modeling.model import FLAIRModel

# --- CONFIGURATION PATHS ---
IMAGE_DIR = "../data/IMAGES"
CSV_PATH = "../data/messidor_data.csv"
PROMPTS_JSON = "../data/final_prompts.json"
OUT_DIR = "../results"

def extract_features():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(">>> INITIATING THESIS FEATURE EXTRACTION PIPELINE <<<")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("[1/3] Loading FLAIR Model from Hugging Face...")
    model = FLAIRModel.from_pretrained("jusiro2/FLAIR").eval().to(device)
    
    # Save Temperature parameter
    logit_scale = model.logit_scale.exp().item()
    np.save(os.path.join(OUT_DIR, "flair_logit_scale.npy"), logit_scale)
    
    print("\n[2/3] Processing Custom Text Prompts...")
    try:
        with open(PROMPTS_JSON, 'r') as f:
            prompts_data = json.load(f)
    except FileNotFoundError:
        print(f"[CRITICAL ERROR] Could not find '{PROMPTS_JSON}'.")
        sys.exit(1)
        
    all_text_embeds, metadata = [], []
    with torch.no_grad():
        for item in tqdm(prompts_data, desc="Text Prompts"):
            raw_text = item["Prompt_Text"]
            text_tokens = model.text_model.tokenize([raw_text])
            input_ids = text_tokens["input_ids"].to(device).to(torch.long)
            attention_mask = text_tokens["attention_mask"].to(device).to(torch.long)
            
            embed = model.text_model(input_ids, attention_mask)
            all_text_embeds.append(embed.cpu().numpy())
            metadata.append({
                'Grade': item["Grade"], 'Family': item["Family"],
                'Variation_Number': item["Variation_Number"], 'Prompt_Text': raw_text
            })
            
    np.save(os.path.join(OUT_DIR, "cached_text_embeddings_custom.npy"), np.vstack(all_text_embeds))
    pd.DataFrame(metadata).to_csv(os.path.join(OUT_DIR, "cached_text_metadata_custom.csv"), index=False)
    
    print("\n[3/3] Extracting Vision Embeddings from MESSIDOR-2...")
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"[CRITICAL ERROR] Could not find '{CSV_PATH}'.")
        sys.exit(1)
        
    # --- SANITIZATION LOGIC ---
    # Purge 4 ungradable images to isolate the valid subset (1748 -> 1744)
    df = df.dropna(subset=['adjudicated_dr_grade'])
    df = df[df['adjudicated_dr_grade'].isin([0.0, 1.0, 2.0, 3.0, 4.0])]
    assert len(df) == 1744, f"Sanitization failure: Expected 1744 valid images, found {len(df)}."
    
    dr_mapping = {0.0: "noDR", 1.0: "mildDR", 2.0: "modDR", 3.0: "sevDR", 4.0: "prolDR"}
    image_embeddings, image_metadata = [], []
    
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Images"):
            img_filename, ground_truth_grade = row['image_id'], row['adjudicated_dr_grade']
            img_path = os.path.join(IMAGE_DIR, img_filename)
            
            if not os.path.exists(img_path): continue
            
            img = np.array(Image.open(img_path).convert('RGB'), dtype=float)
            img_tensor = model.preprocess_image(img).to(device)
            
            vis_embed = model.vision_model(img_tensor)
            image_embeddings.append(vis_embed.cpu().numpy())
            image_metadata.append({
                'image_id': img_filename, 'adjudicated_dr_grade': ground_truth_grade,
                'mapped_flair_class': dr_mapping.get(ground_truth_grade, "Unknown")
            })

    np.save(os.path.join(OUT_DIR, "cached_vision_embeddings_1744.npy"), np.vstack(image_embeddings))
    pd.DataFrame(image_metadata).to_csv(os.path.join(OUT_DIR, "cached_vision_metadata_1744.csv"), index=False)
    print("\n>>> PIPELINE EXECUTED FLAWLESSLY. MATRICES CACHED. <<<")

if __name__ == "__main__":
    extract_features()
