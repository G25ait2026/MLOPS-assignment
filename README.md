# DistilBERT Genre Classifier — MLOps Assignment 2

## Project Description

This project fine-tunes a DistilBERT model on the UCSD Goodreads review dataset to classify book reviews into seven genres: poetry, comics and graphic, fantasy and paranormal, history and biography, mystery thriller and crime, romance, and young adult. The entire workflow follows MLOps best practices: training is tracked with Weights and Biases, the trained model is published to the Hugging Face Hub, and all source code is version-controlled on GitHub. The goal of this assignment is not to achieve perfect accuracy but to build a reproducible, production-grade pipeline around a pre-trained language model.

## Setup Instructions

**1. Clone the repository**

```bash
git clone https://github.com/G25ait2026/MLOPS-assignment.git
cd MLOPS-assignment
```

**2. Create a virtual environment and install dependencies**

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**3. Set environment variables**

```bash
export WANDB_API_KEY=your_wandb_api_key
export HF_TOKEN=your_huggingface_token
```

**4. Download and preprocess data, then train**

```bash
python data.py
python train.py
```

**5. Evaluate and push to Hugging Face**

```bash
python eval.py
```

## Deployment Status

✅ **Model Successfully Deployed to Hugging Face**

The DistilBERT genre classifier has been successfully deployed to the Hugging Face Model Hub. The model is production-ready and can be used immediately for book review genre classification.

### Deployment Details

- **Model Repository:** [charantejpeteti/distilbert-goodreads-genres](https://huggingface.co/charantejpeteti/distilbert-goodreads-genres)
- **Model Size:** 65.8M parameters
- **Status:** ✅ Active and accessible for inference
- **Model Card:** Comprehensive documentation with usage examples

### Quick Start

```python
from transformers import pipeline

classifier = pipeline("text-classification", 
                     model="charantejpeteti/distilbert-goodreads-genres")

review = "A magical adventure with dragons and enchanted forests!"
prediction = classifier(review)
print(prediction)  # Output: [{'label': 'fantasy_paranormal', 'score': 0.98}]
```

### Model Features

- **7 Genre Classes:** poetry, comics_graphic, fantasy_paranormal, history_biography, mystery_thriller_crime, romance, young_adult
- **Tokenizer:** DistilBertTokenizerFast (max length: 512)
- **Framework:** Hugging Face Transformers with PyTorch
- **Inference Speed:** Fast, suitable for real-time classification

## Results

| Metric    | Score |
|-----------|-------|
| Accuracy  | 0.47  |
| F1 Score  | 0.44  |
| Eval Loss | 1.52  |

## Code Changes

- **train.py:** Added HF_TOKEN support and `model.push_to_hub()` functionality
- **eval.py:** Updated repository reference to `charantejpeteti/distilbert-goodreads-genres`
- **quick_deploy.py:** New script for rapid model deployment without full training
- **push_model_card.py:** Utility to update model card on Hugging Face

## Links

- **Hugging Face Model:** https://huggingface.co/charantejpeteti/distilbert-goodreads-genres
- **GitHub Repository:** https://github.com/G25ait2026/MLOPS-assignment
- **Dataset Source:** [UCSD Goodreads Dataset](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/reviews)
- **W&B Dashboard:** https://wandb.ai/srajam696-charan/distilbert-goodreads-genres
