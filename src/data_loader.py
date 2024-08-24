import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from typing import List

class SentimentExample:
    """
    Data wrapper for a single example for sentiment analysis.

    Attributes:
        words (List[string]): list of words
        label (int): 0 or 1 (0 = negative, 1 = positive)
        word_indices (List[int]): list of word indices in the vocab, which will generated by the `indexing_sentiment_examples` method
    """
    def __init__(self, words: List[str], label: int = None):
        self.words = words
        self.label = label
        self.word_indices = None

    def __repr__(self):
        return repr(self.words) + "; label=" + repr(self.label)

class SentimentDataset(Dataset):
    """
    Dataset class to handle SentimentExamples for PyTorch DataLoader.
    """
    def __init__(self, examples: List[SentimentExample], vocab: List[str], PAD_idx: int, UNK_idx: int):
        self.examples = examples
        self.vocab = vocab
        self.PAD_idx = PAD_idx
        self.UNK_idx = UNK_idx

        self.indexing_sentiment_examples()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        if ex.label is not None:
            return ex.word_indices, len(ex.word_indices), ex.label
        else:
            return ex.word_indices, len(ex.word_indices)

    def indexing_sentiment_examples(self):
        for ex in self.examples:
            ex.word_indices = [self.vocab.index(word) if word in self.vocab else self.UNK_idx for word in ex.words]

def pad_collate_fn(batch):
    """
    Custom collate function for DataLoader to handle padding.
    """
    batch_inputs, batch_lengths = zip(*[(b[0], b[1]) for b in batch])
    max_length = max(batch_lengths)
    padded_inputs = [inputs + [0] * (max_length - len(inputs)) for inputs in batch_inputs]
    if len(batch[0]) == 3:
        batch_labels = [b[2] for b in batch]
        return torch.tensor(padded_inputs), torch.tensor(batch_lengths), torch.tensor(batch_labels)
    else: 
        return torch.tensor(padded_inputs), torch.tensor(batch_lengths)
