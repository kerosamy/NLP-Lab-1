"""Download, clean, and split the emotion dataset before model training."""

import re

import numpy as np

DATASET_ID = "avsolatorio/mteb-emotion-avs_triplets"
DATASET_REVISION = "7af7023e3f8be456bfe17846aaed40301048d0fe"

# The position in this list is the dataset's numeric label.
LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def preprocess(text):
    """Clean any English sentence; use .split() on the result to get word tokens."""
    words = re.findall(r"[a-z]+", text.lower())
    return " ".join(words)


def clean_dataset(rows):
    """Apply basic preprocessing and keep every dataset row with its label."""
    texts = []
    labels = []
    for row in rows:
        texts.append(preprocess(row["text"]))
        labels.append(int(row["label"]))

    print(f"Dataset: {len(texts):,} sentences")
    return texts, np.asarray(labels, dtype=int)


def random_split(texts, labels, test_size=0.1, seed=42):
    """Shuffle all rows and reserve 10% for testing by default.

    Returns train_texts, test_texts, train_labels, and test_labels.
    The same seed produces the same split and sentence order.
    """
    labels = np.asarray(labels)
    random_generator = np.random.default_rng(seed)
    shuffled_indices = random_generator.permutation(len(texts))
    number_of_test_samples = int(np.ceil(len(texts) * test_size))

    test_indices = shuffled_indices[:number_of_test_samples]
    train_indices = shuffled_indices[number_of_test_samples:]

    train_texts = [texts[index] for index in train_indices]
    test_texts = [texts[index] for index in test_indices]
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]
    return train_texts, test_texts, train_labels, test_labels


def load_and_split(test_size=0.1, seed=42):
    """Download once into Hugging Face's cache, then clean and split the data."""
    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, split="train", revision=DATASET_REVISION)
    texts, labels = clean_dataset(dataset)
    return random_split(texts, labels, test_size=test_size, seed=seed)
