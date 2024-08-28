from collections import Counter
from typing import List
import torch.utils
from src.data_loader import SentimentExample, DataLoader, pad_collate_fn
import os
import torch
from typing import List, Tuple


# Builds a vocab based on the min_freq
def build_vocab(examples: List[SentimentExample], min_freq: int = 2) -> List[str]:
    word_counter = Counter(word for ex in examples for word in ex.words)
    vocab = ["PAD", "UNK"] + [word for word, count in word_counter.items() if count >= min_freq]
    return vocab


# Reads a sentiment file and creates SentimentExample objects to load to DataLoader
def read_sentiment_examples(filepath: str, labeled: bool = True) -> List[SentimentExample]:
    examples = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                if labeled:
                    label, sent = line.split("\t")
                    label = int(label)
                else:
                    sent = line
                    label = None
                sent = sent.lower().split()
                examples.append(SentimentExample(sent, label))
    return examples

def preprocess_sentence_FFNN(sentence: str, vocab: List[str], PAD_IDX: int, UNK_IDX: int, max_length=20) -> List[List[int]]:
    tokens = sentence.lower().split()
    indicies = [vocab.index(word) if word in vocab else UNK_IDX for word in tokens]
    if len(indicies) < max_length:
        indicies += [PAD_IDX] * (max_length - len(indicies))
    elif len(indicies) > max_length:
        indicies = indicies[:max_length]
    return [indicies]

def preprocess_sentence_BERT(sentence: str, tokenizer, max_length: int=128):
    encoded_input = tokenizer(
        sentence,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    return encoded_input


def output_predictions(folder_path: str, file_name: str, labeled_predictions: List[Tuple[int, str]]) -> None:
    file_path = os.path.join(folder_path, file_name)

    os.makedirs(folder_path, exist_ok=True)

    with open(file_path, 'w') as file:
        for label, sentence in labeled_predictions:
            file.write(f"{label}\t{sentence}\n")

def extract_labels(filepath: str):
    labels = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                label, _ = line.split("\t")
                label = int(label)
            labels.append(label)
    return labels