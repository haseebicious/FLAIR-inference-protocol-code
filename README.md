# Lexical Overfitting in Zero-Shot Retinal Diagnosis: A Study of Medical Vision-Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official codebase, evaluation pipeline, and prompt taxonomies for the paper: **"Lexical Overfitting in Zero-Shot Retinal Diagnosis: A Study of Medical Vision-Language Models."** 

This project investigates the semantic brittleness and "epistemic arrogance" of medical Vision-Language Models (VLMs), specifically evaluating the [FLAIR](https://github.com/jusiro/FLAIR) foundation model. By implementing a **Strictly-Frozen Inference Protocol** and an **Automated UMLS Sanitization Pipeline**, we isolate cross-modal representations from gradient-driven memorization to demonstrate how models overfit to clinical vocabulary.

## 📌 Prerequisites & Installation

1. Clone this repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/VLM-Lexical-Overfitting.git](https://github.com/YOUR_USERNAME/VLM-Lexical-Overfitting.git)
   cd VLM-Lexical-Overfitting

2. Install the required Python packages:
   ```bash
   pip install torch torchvision numpy pandas tqdm pillow transformers aiohttp matplotlib seaborn scikit-learn statsmodels
   pip install git+[https://github.com/jusiro/FLAIR.git](https://github.com/jusiro/FLAIR.git)

3. Request an NIH UMLS API Key (required only if running the automated semantic sanitization pipeline).

🗂️ Dataset Preparation
1. Download the MESSIDOR-2 images and place them in the data/IMAGES/ directory.
2. Ensure your metadata file (messidor_data.csv) is placed in data/.
Note: Our pipeline automatically drops four ungradable images, enforcing a strict 1,744 valid image subset for exact replication of our tensor space.

🚀 Step-by-Step Pipeline Execution

The codebase is highly modular. All outputs, cached matrices, and generated graphs will be saved automatically to the results/ directory.

Step 1: Automated UMLS Prompt Sanitization
To guarantee the strict removal of latent medical ontologies from non-expert prompts ($T_{layman}$ and $T_{exclusion}$), we utilize an asynchronous pipeline interfacing with the NIH UMLS REST API.(Open scripts/1_umls_sanitization.py and replace the placeholder API key with your own, or pass it via environment variable).

  ```bash
  python scripts/1_umls_sanitization.py


Step 2: Frozen-Inference Feature Extraction
Extracts the visual embeddings from the 1,744 valid MESSIDOR-2 images and the text embeddings from the custom taxonomy prompts using the frozen FLAIR model.

  ```bash
    python scripts/2_feature_extraction.py

Step 3: Mathematical Audit & Statistics
Measures the Expected Calibration Error (ECE), computes the Modality Gap (Silhouette score), and calculates McNemar's statistical significance tests across the different prompt taxonomies.
  ```bash
  python scripts/3_inference_and_audit.py

Step 4: Visualizations
Generates publication-quality reliability diagrams (ECE curves).
  ```bash
python scripts/4_generate_figures.py


📊 Reproducing UMAP Topography
To reproduce the 2D UMAP projection demonstrating the Vision-Language Modality Gap discussed in our paper, utilize the following deterministic parameters on the cached embeddings:
n_neighbors = 15, min_dist = 0.1, metric = 'cosine', random_state = 42.




