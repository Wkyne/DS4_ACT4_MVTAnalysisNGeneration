# Act 4 — Multi-Variant Text Analysis and Generation
Group 1 CSS182-04 CM2

> **Course Activity** | BERT, GPT, and GANs  
> **Dataset:** | [Twitter Financial News Sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) — 11,932 finance tweets, 3-class sentiment (Bearish / Bullish / Neutral)
---
Submitted by:  
Granada, Dustin Joaquin B.  
Dequinon, Rob Eugene  
Gallor, Michael Edward II  

---

Three NLP architectures implemented on the same corpus for direct comparison: BERT (classification), DistilGPT-2 (generation), and a custom Text GAN (adversarial generation).

---

## Repository Structure

```
DS4_ACT4_MVTAnalysisNGeneration/
├── data/
│   └── README.md               # Dataset source, license, and loading instructions
├── notebooks/
│   ├── 00_preprocessing.ipynb  # Shared cleaning, splitting, and EDA
│   ├── 01_bert_variant.ipynb   # Fine-tuning + classification evaluation
│   ├── 02_gpt_variant.ipynb    # DistilGPT-2 fine-tuning + text generation
│   └── 03_gan_variant.ipynb    # Text GAN training + adversarial evaluation
├── results/
│   ├── training_histories/     # Loss/accuracy curves (PNG + CSV) per variant
│   ├── sample_outputs/         # Generated text samples and BERT predictions
│   └── figures/                # Confusion matrix, comparison plots
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/DS4_ACT4_MVTAnalysisNGeneration.git
cd DS4_ACT4_MVTAnalysisNGeneration
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Google Colab users:** Run this at the top of each notebook instead:
> ```python
> !pip install -r requirements.txt
> ```
> All notebooks are designed to run end-to-end on a Colab T4 GPU runtime.

### 3. Run notebooks in order

| Order | Notebook | Purpose |
|-------|----------|---------|
| 1 | `00_preprocessing.ipynb` | Load dataset, clean, EDA, export shared splits |
| 2 | `01_bert_variant.ipynb` | BERT fine-tuning and classification metrics |
| 3 | `02_gpt_variant.ipynb` | DistilGPT-2 fine-tuning and text generation |
| 4 | `03_gan_variant.ipynb` | Text GAN training and adversarial evaluation |

Each variant notebook is self-contained and can be run independently after `00_preprocessing.ipynb` has been run once.

---

## Standardized Shared Scripts

```python
import sys
sys.path.append('../') # adds root directory to path
from utils import compute_classification_metrics, compute_generation_metrics, calculate_perplexity
```

## Model Variants

### Variant 1 — BERT (`01_bert_variant.ipynb`)
- **Model:** `bert-base-uncased` via HuggingFace Transformers
- **Task:** Binary sentiment classification (positive / negative)
- **Head:** `BertForSequenceClassification` with `num_labels=2`
- **Metrics:** Precision, Recall, F1-score (per class + macro avg)

### Variant 2 — DistilGPT-2 (`02_gpt_variant.ipynb`)
- **Model:** `distilgpt2` via HuggingFace Transformers
- **Task:** Causal language modeling — domain-conditioned movie review generation
- **Metrics:** BLEU, ROUGE-L, Perplexity (from validation loss)

### Variant 3 — Text GAN (`03_gan_variant.ipynb`)
- **Architecture:** Embedding → 1D-CNN/GRU Generator + 1D-CNN Discriminator
- **Task:** Synthetic adversarial sentence generation
- **Metrics:** Discriminator Precision/Recall/F1, Generator BLEU/ROUGE vs. real test sentences

---

## Performance Comparison Matrix

> Populated after all notebooks are run. See `results/figures/metrics_comparison_matrix.png` for the rendered version.

| Model Variant | Precision / Recall / F1 | BLEU / ROUGE / Perplexity | Training Time / Epoch | Key Observations |
|---------------|------------------------|---------------------------|-----------------------|-----------------|
| BERT | TBD | N/A (Classification) | TBD | |
| DistilGPT-2 | N/A (Generative) | TBD | TBD | |
| Text GAN | TBD (Discriminator) | TBD (Generator) | TBD | |

---

## Results Shortcuts

From the repo root, open any results folder directly:

```bash
# View training histories
open results/training_histories/        # macOS
xdg-open results/training_histories/    # Linux
start results\training_histories\       # Windows

# View generated text samples
cat results/sample_outputs/gpt_generated_samples.txt
cat results/sample_outputs/gan_generated_samples.txt

# View BERT predictions
cat results/sample_outputs/bert_test_predictions.csv
```

Or use Python from anywhere in the repo:

```python
import subprocess, os
subprocess.Popen(["open", os.path.join(os.path.dirname(__file__), "results")])
```

---

## Citation

```bibtex
@dataset{zeroshot_twitter_financial_news_sentiment,
  title     = {Twitter Financial News Sentiment},
  author    = {Zeroshot},
  year      = {2022},
  publisher = {HuggingFace Datasets},
  url       = {https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment}
}
```