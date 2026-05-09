"""Push model card to Hugging Face Hub."""
import os
from huggingface_hub import HfApi, create_repo

HF_REPO = "charantejpeteti/distilbert-goodreads-genres"
README_PATH = "distilbert-goodreads-genres/README.md"

def push_model_card():
    """Upload model card to existing repository."""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("❌ HF_TOKEN not set.")
        return
    
    api = HfApi(token=hf_token)
    
    # Read the README
    with open(README_PATH, "r") as f:
        readme_content = f.read()
    
    print(f"Uploading model card to {HF_REPO}...")
    
    try:
        # Upload the README
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=HF_REPO,
            repo_type="model",
        )
        print(f"✅ Model card successfully updated!")
        print(f"View it at: https://huggingface.co/{HF_REPO}")
    except Exception as e:
        print(f"❌ Error uploading model card: {e}")

if __name__ == "__main__":
    push_model_card()
