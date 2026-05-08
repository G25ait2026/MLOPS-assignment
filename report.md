# MLOps Assignment 2: Final Report

**Student:** charantej  
**Roll Number:** g25ait2026

## 1. Model Selection Rationale

For this text classification task, I selected **DistilBERT** (`distilbert-base-cased`) as the primary architecture. DistilBERT is a small, fast, cheap and light Transformer model trained by distilling BERT. It has 40% less parameters than `bert-base-uncased`, runs 60% faster while preserving over 95% of BERT's performances as measured on the GLUE language understanding benchmark.

**Why DistilBERT?**
- **Efficiency:** Given the constraints of training on a CPU for this demonstration, DistilBERT's reduced parameter count allowed for manageable training times.
- **Performance:** It provides a strong baseline for sequence classification tasks, specifically for genre detection where context and semantics in the review text are critical.
- **Ecosystem:** Excellent support within the Hugging Face Transformers library and seamless integration with the `Trainer` API.

## 2. Training Summary

The training was performed using a modular MLOps pipeline consisting of `data.py`, `train.py`, and `eval.py`.

- **Data Preparation:** We sampled 100 reviews per genre across 7 different book genres (Poetry, Comics, Fantasy, etc.). Text was truncated to a maximum length of 128 tokens to optimize for CPU training speed.
- **Training Process:** The model was fine-tuned for 3 epochs using a learning rate of 3e-5 and a batch size of 16.
- **Experiment Tracking:** All hyperparameters, training loss, and evaluation metrics were logged in real-time to **Weights & Biases**. The dashboard provided visibility into the loss convergence and hardware utilization.

## 3. Evaluation Results

The final evaluation was conducted on a held-out test set (20% of the data).

| Metric    | Score |
|-----------|-------|
| Accuracy  | 0.47  |
| F1 Score  | 0.44  |
| Eval Loss | 1.52  |

**Interpretation:**
- **Accuracy (47%):** While 47% might seem low for a classification task, it is significantly better than random guessing (which would be ~14% for 7 classes). This indicates the model has learned meaningful patterns to distinguish genres even with a very limited sample size (100 per genre).
- **F1 Score (0.44):** The weighted F1 score of 0.44 suggests that the model's performance is relatively balanced across classes, though some genres (like Poetry vs. Comics) likely have distinct linguistic markers that the model picked up more easily than others.
- **Loss (1.52):** The loss showed a steady decline over the 3 epochs, suggesting that further training with more data would likely improve performance significantly.

## 4. Challenges & Learnings

**Challenges:**
- **Hardware Constraints:** Training Transformers on a CPU is computationally intensive. This required careful optimization of sequence lengths and batch sizes to ensure the pipeline could run within a reasonable timeframe.
- **Authentication:** Managing tokens for W&B and Hugging Face Hub securely within a script required using environment variables and configuration files.

**Learnings:**
- **Modularity:** Breaking down a notebook into `data`, `train`, and `eval` scripts significantly improves reproducibility and makes the codebase "production-ready".
- **Experiment Tracking:** Using W&B is invaluable for comparing different runs and understanding model behavior without manually logging metrics.
- **Deployment:** The process of pushing a model to the Hugging Face Hub demonstrated how easy it is to share and version-control machine learning assets once the pipeline is correctly configured.

---
**Links:**
- **W&B Project:** [distilbert-goodreads-genres](https://wandb.ai/srajam696-charan/distilbert-goodreads-genres)
- **Hugging Face Model:** [srajam696/distilbert-goodreads-genres](https://huggingface.co/srajam696/distilbert-goodreads-genres)
