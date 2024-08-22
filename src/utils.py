from collections import Counter
from typing import List
from src.data_loader import SentimentExample

# Builds a vocab based on the min_freq
def build_vocab(examples: List[SentimentExample], min_freq: int = 2) -> List[str]:
    word_counter = Counter(word for ex in examples for word in ex.words)
    vocab = ["PAD", "UNK"] + [word for word, count in word_counter.items() if count >= min_freq]
    return vocab

# Reads a sentiment file and creates SentimentExample objects to load to DataLoader
def read_sentiment_examples(filepath: str) -> List[SentimentExample]:
    examples = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                label, sent = line.split("\t")
                label = int(label)
                sent = sent.lower().split()
                examples.append(SentimentExample(sent, label))
    return examples
