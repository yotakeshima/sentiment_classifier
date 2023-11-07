import torch
from torch import nn, optim
from typing import List, Dict
import random
import numpy as np
from collections import Counter
import time
import os

print(torch.__version__)
print(torch.cuda.is_available())

seed = 12345
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

class FeedForwardNeuralNetClassifier(nn.Module):
    """
    The Feed-Forward Neural Net sentiment classifier.
    """
    def __init__(self, vocab_size, emb_dim, n_hidden_units):
        """
        In the __init__ function, you will define modules in FFNN.
        :param vocab_size: size of vocabulary
        :param emb_dim: dimension of the embedding vectors
        :param n_hidden_units: dimension of the hidden units
        """
        super(FeedForwardNeuralNetClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_hidden_units = n_hidden_units
       
        # TODO: implement a randomly initialized word embedding matrix using nn.Embedding
        # It should have a size of `(vocab_size x emb_dim)`
        self.word_embeddings = nn.Embedding(vocab_size, emb_dim)

        # TODO: implement the FFNN architecture using nn functions
        self.W = nn.Linear(emb_dim, n_hidden_units)
        self.g = nn.ReLU()
        self.V = nn.Linear(n_hidden_units, 1)
        self.sigmoid = nn.Sigmoid()

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, batch_inputs: torch.Tensor, batch_lengths: torch.Tensor) -> torch.Tensor:
        """
        The forward function, which defines how FFNN should work when given a batch of inputs and their actual sent lengths (i.e., before PAD)
        :param batch_inputs: a torch.Tensor object of size (n_examples, max_sent_length_in_this_batch), which is the *indexed* inputs
        :param batch_lengths: a torch.Tensor object of size (n_examples), which describes the actual sentence length of each example (i.e., before PAD)
        :return the logits of FFNN (i.e., the unnormalized hidden units before sigmoid) of shape (n_examples)
        """
        
        word_avg = torch.mean(self.word_embeddings(batch_inputs), 1 )
        logits = torch.squeeze(self.V(self.g(self.W(word_avg))))
        return logits
    
    def batch_predict(self, batch_inputs: torch.Tensor, batch_lengths: torch.Tensor) -> List[int]:
        """
        Make predictions for a batch of inputs. This function may directly invoke `forward` (which passes the input through FFNN and returns the output logits)

        :param batch_inputs: a torch.Tensor object of size (n_examples, max_sent_length_in_this_batch), which is the *indexed* inputs
        :param batch_lengths: a torch.Tensor object of size (n_examples), which describes the actual sentence length of each example (i.e., before PAD)
        :return: a list of predicted classes for this batch of data, either 0 for negative class or 1 for positive class
        """
        logits = self.forward(batch_inputs, batch_lengths)
        
        predicted_labels = self.sigmoid(logits)
        predicted_labels = [1 if x >= 0.5 else 0 for x in predicted_labels]
        return predicted_labels
