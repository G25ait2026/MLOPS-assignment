import json
import os

import wandb
from huggingface_hub import login
from sklearn.metrics import classification_report
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from data import (
    PICKLE_PATH,
    encode_data,
    fetch_all_genres,
    load_from_pickle,
    split_data,
)
from train import (
    SAVED_MODEL_DIR,
    WANDB_ENTITY,
    WANDB_PROJECT,
    WANDB_RUN_NAME,
    build_datasets,
    get_training_args,
    load_model,
)
from utils import ReviewDataset, compute_metrics, build_label_maps

from transformers import Trainer

HF_REPO = "charantejpeteti/distilbert-goodreads-genres"
EVAL_REPORT_PATH = "eval_report.json"


def run_evaluation():
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

    train_dataset, test_dataset = build_datasets(train_enc, train_lbl, test_enc, test_lbl)

    model = DistilBertForSequenceClassification.from_pretrained(SAVED_MODEL_DIR)

    training_args = get_training_args()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME + "-eval",
    )

    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)

    wandb.log(
        {
            "final/loss": eval_results["eval_loss"],
            "final/accuracy": eval_results["eval_accuracy"],
            "final/f1": eval_results["eval_f1"],
        }
    )

    predictions = trainer.predict(test_dataset).predictions.argmax(-1)
    true_labels = [item["labels"].item() for item in test_dataset]

    report = classification_report(
        true_labels,
        predictions,
        target_names=list(id2label.values()),
        output_dict=True,
    )

    with open(EVAL_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(classification_report(true_labels, predictions, target_names=list(id2label.values())))

    artifact = wandb.Artifact("eval-report", type="evaluation")
    artifact.add_file(EVAL_REPORT_PATH)
    wandb.log_artifact(artifact)

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import HfApi
        api = HfApi()
        user_info = api.whoami(token=hf_token)
        username = user_info["name"]
        dynamic_repo = f"{username}/distilbert-goodreads-genres"
        
        login(token=hf_token)
        model.push_to_hub(dynamic_repo)
        tokenizer.push_to_hub(dynamic_repo)
        hf_url = f"https://huggingface.co/{dynamic_repo}"
        wandb.run.summary["huggingface_model"] = hf_url
        print(f"Model pushed to Hugging Face: {hf_url}")
    else:
        print("HF_TOKEN not set. Skipping Hugging Face push.")

    wandb.finish()

    return eval_results, report


if __name__ == "__main__":
    run_evaluation()
