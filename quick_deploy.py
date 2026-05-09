#!/usr/bin/env python
"""Quick deployment of model to Hugging Face without training."""
import os
from huggingface_hub import login
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

# Configuration
MODEL_NAME = "distilbert-base-cased"
HF_REPO = "charantejpeteti/distilbert-goodreads-genres"
SAVED_MODEL_DIR = "distilbert-goodreads-genres"

# Label mapping for 7 genres
id2label = {
    0: "poetry",
    1: "comics_graphic",
    2: "fantasy_paranormal",
    3: "history_biography",
    4: "mystery_thriller_crime",
    5: "romance",
    6: "young_adult"
}

label2id = {v: k for k, v in id2label.items()}

def deploy():
    """Load pre-trained model and push to hub."""
    print("Loading pre-trained DistilBERT model...")
    
    # Load model and tokenizer
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )
    
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    
    # Save locally
    print(f"Saving model to {SAVED_MODEL_DIR}...")
    model.save_pretrained(SAVED_MODEL_DIR)
    tokenizer.save_pretrained(SAVED_MODEL_DIR)
    
    # Push to Hugging Face
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print(f"Logging in to Hugging Face...")
        login(token=hf_token)
        
        print(f"Pushing model to {HF_REPO}...")
        model.push_to_hub(HF_REPO)
        
        print(f"Pushing tokenizer to {HF_REPO}...")
        tokenizer.push_to_hub(HF_REPO)
        
        hf_url = f"https://huggingface.co/{HF_REPO}"
        print(f"✅ Model successfully deployed to: {hf_url}")
    else:
        print("❌ HF_TOKEN not set. Skipping Hugging Face push.")

if __name__ == "__main__":
    deploy()
