import torch
import torch.nn as nn
from typing import List

class FeedForwardNeuralNetClassifier(nn.Module):
    """
    A Feed-Forward Neural Network for sentiment classification.
    """
    def __init__(self, vocab_size: int, emb_dim: int, n_hidden_units: int):
        super(FeedForwardNeuralNetClassifier, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, emb_dim)
        self.fc1 = nn.Linear(emb_dim, n_hidden_units)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden_units, 1)
        self.sigmoid = nn.Sigmoid()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embeddings = torch.mean(self.word_embeddings(inputs), dim=1)
        hidden = self.relu(self.fc1(embeddings))
        logits = self.fc2(hidden)
        return logits.squeeze()

    def predict(self, inputs: torch.Tensor) -> List[int]:
        logits = self.forward(inputs)
        probabilities = self.sigmoid(logits)
        return (probabilities >= 0.5).int().tolist()
