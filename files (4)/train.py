import os
import pickle

import torch
import wandb
from transformers import (
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from data import (
    PICKLE_PATH,
    encode_data,
    fetch_all_genres,
    load_from_pickle,
    split_data,
)
from utils import ReviewDataset, compute_metrics

MODEL_NAME = "distilbert-base-cased"
MAX_LENGTH = 512
OUTPUT_DIR = "./results"
SAVED_MODEL_DIR = "distilbert-goodreads-genres"
WANDB_PROJECT = "mlops-assignment2"
WANDB_RUN_NAME = "distilbert-run-1"


def build_datasets(train_encodings, train_labels_encoded, test_encodings, test_labels_encoded):
    train_dataset = ReviewDataset(train_encodings, train_labels_encoded)
    test_dataset = ReviewDataset(test_encodings, test_labels_encoded)
    return train_dataset, test_dataset


def load_model(model_name, num_labels, id2label, label2id):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    ).to(device)
    return model


def get_training_args(output_dir=OUTPUT_DIR):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=WANDB_RUN_NAME,
        learning_rate=3e-5,
    )


def run_training():
    if os.path.exists(PICKLE_PATH):
        print("Loading reviews from cache.")
        genre_reviews_dict = load_from_pickle()
    else:
        print("Downloading reviews from source.")
        genre_reviews_dict = fetch_all_genres()

    train_texts, train_labels, test_texts, test_labels = split_data(genre_reviews_dict)

    tokenizer, train_enc, train_lbl, test_enc, test_lbl, label2id, id2label = encode_data(
        train_texts, train_labels, test_texts, test_labels
    )

    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "model": MODEL_NAME,
            "epochs": 3,
            "batch_size": 16,
            "learning_rate": 3e-5,
            "max_length": MAX_LENGTH,
            "dataset": "UCSD Goodreads",
            "num_labels": len(id2label),
        },
    )

    train_dataset, test_dataset = build_datasets(train_enc, train_lbl, test_enc, test_lbl)

    model = load_model(MODEL_NAME, len(id2label), id2label, label2id)

    training_args = get_training_args()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(SAVED_MODEL_DIR)
    tokenizer.save_pretrained(SAVED_MODEL_DIR)

    print(f"Model saved to {SAVED_MODEL_DIR}")
    wandb.finish()

    return trainer, tokenizer, test_dataset, test_labels, id2label


if __name__ == "__main__":
    run_training()
