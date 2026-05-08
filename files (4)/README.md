# DistilBERT Genre Classifier — MLOps Assignment 2

## Project Description

This project fine-tunes a DistilBERT model on the UCSD Goodreads review dataset to classify book reviews into seven genres: poetry, comics and graphic, fantasy and paranormal, history and biography, mystery thriller and crime, romance, and young adult. The entire workflow follows MLOps best practices: training is tracked with Weights and Biases, the trained model is published to the Hugging Face Hub, and all source code is version-controlled on GitHub. The goal of this assignment is not to achieve perfect accuracy but to build a reproducible, production-grade pipeline around a pre-trained language model.

## Setup Instructions

**1. Clone the repository**

```bash
git clone https://github.com/your-username/distilbert-goodreads-genres.git
cd distilbert-goodreads-genres
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

## Results

| Metric    | Score |
|-----------|-------|
| Accuracy  | 0.XX  |
| F1 Score  | 0.XX  |
| Eval Loss | 0.XX  |

Replace the placeholder values above with actual results after running training and evaluation.

## Links

- Hugging Face model: https://huggingface.co/your-username/distilbert-goodreads-genres
- W&B dashboard: https://wandb.ai/your-username/mlops-assignment2
