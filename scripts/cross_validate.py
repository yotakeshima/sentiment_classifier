from src.model import FeedForwardNeuralNetClassifier
from src.data_loader import SentimentDataset
from src.train import cross_validate_model
from src.utils import read_sentiment_examples, build_vocab
import torch

train_examples = read_sentiment_examples('data/train.txt')
vocab = build_vocab(train_examples)

PAD_IDX = vocab.index("PAD")
UNK_IDX = vocab.index("UNK")

model = FeedForwardNeuralNetClassifier(vocab_size=len(vocab), emb_dim=300, n_hidden_units=300)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = SentimentDataset(train_examples, vocab, PAD_IDX, UNK_IDX)

print("Cross Validating the model:")
cross_validate_model(model, dataset, n_splits=10, device=device)