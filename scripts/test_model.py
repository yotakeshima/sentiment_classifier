
import torch
import os
from src.model import FeedForwardNeuralNetClassifier
from src.data_loader import SentimentDataset, DataLoader, pad_collate_fn
from src.utils import  read_sentiment_examples, build_vocab, output_predictions
from src.predict import make_predictions

test_examples = read_sentiment_examples('data/test-blind.txt', False)
train_examples = read_sentiment_examples('data/train.txt')
vocab = build_vocab(train_examples)

model = FeedForwardNeuralNetClassifier(vocab_size=len(vocab), emb_dim=300, n_hidden_units=300)
model.load_state_dict(torch.load("checkpoints/best_model.pth"))
model.eval()

PAD_IDX = vocab.index("PAD")
UNK_IDX = vocab.index("UNK")

test_dataset = SentimentDataset(train_examples, vocab, PAD_IDX, UNK_IDX)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_predictions = []

predictions = make_predictions(model, test_loader, device=device)

with open('data/test-blind.txt', 'r') as file:
    sentences = [line.strip() for line in file]
labeled_predictions = [(label, sentence) for label, sentence in zip(predictions, sentences)]

output_predictions('predictions', 'output.txt', labeled_predictions)