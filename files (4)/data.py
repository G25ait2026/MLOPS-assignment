import gzip
import json
import pickle
import random
import requests

from transformers import DistilBertTokenizerFast

from utils import build_label_maps

GENRE_URL_DICT = {
    "poetry":                 "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz",
    "comics_graphic":         "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz",
    "fantasy_paranormal":     "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz",
    "history_biography":      "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz",
    "mystery_thriller_crime": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz",
    "romance":                "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz",
    "young_adult":            "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz",
}

MODEL_NAME = "distilbert-base-cased"
MAX_LENGTH = 512
HEAD = 10000
SAMPLE_SIZE = 2000
REVIEWS_PER_GENRE = 1000
TRAIN_SPLIT = 800
PICKLE_PATH = "genre_reviews_dict.pickle"


def load_reviews(url, head=HEAD, sample_size=SAMPLE_SIZE):
    reviews = []
    count = 0
    response = requests.get(url, stream=True)
    with gzip.open(response.raw, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            reviews.append(d["review_text"])
            count += 1
            if head is not None and count >= head:
                break
    return random.sample(reviews, min(sample_size, len(reviews)))


def fetch_all_genres(pickle_path=PICKLE_PATH):
    genre_reviews_dict = {}
    for genre, url in GENRE_URL_DICT.items():
        print(f"Loading reviews for genre: {genre}")
        genre_reviews_dict[genre] = load_reviews(url)
    with open(pickle_path, "wb") as f:
        pickle.dump(genre_reviews_dict, f)
    return genre_reviews_dict


def load_from_pickle(pickle_path=PICKLE_PATH):
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def split_data(genre_reviews_dict, reviews_per_genre=REVIEWS_PER_GENRE, train_split=TRAIN_SPLIT):
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []
    for genre, reviews in genre_reviews_dict.items():
        sampled = random.sample(reviews, min(reviews_per_genre, len(reviews)))
        for review in sampled[:train_split]:
            train_texts.append(review)
            train_labels.append(genre)
        for review in sampled[train_split:]:
            test_texts.append(review)
            test_labels.append(genre)
    return train_texts, train_labels, test_texts, test_labels


def encode_data(train_texts, train_labels, test_texts, test_labels, model_name=MODEL_NAME, max_length=MAX_LENGTH):
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    label2id, id2label = build_label_maps(train_labels + test_labels)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)

    train_labels_encoded = [label2id[y] for y in train_labels]
    test_labels_encoded = [label2id[y] for y in test_labels]

    return tokenizer, train_encodings, train_labels_encoded, test_encodings, test_labels_encoded, label2id, id2label


if __name__ == "__main__":
    import os

    if os.path.exists(PICKLE_PATH):
        print("Loading reviews from cache.")
        genre_reviews_dict = load_from_pickle()
    else:
        print("Downloading reviews from source.")
        genre_reviews_dict = fetch_all_genres()

    train_texts, train_labels, test_texts, test_labels = split_data(genre_reviews_dict)
    print(f"Train size: {len(train_texts)}, Test size: {len(test_texts)}")

    tokenizer, train_enc, train_lbl, test_enc, test_lbl, label2id, id2label = encode_data(
        train_texts, train_labels, test_texts, test_labels
    )
    print("Encoding complete.")
    print(f"Label map: {label2id}")
