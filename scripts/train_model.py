import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.model import FeedForwardNeuralNetClassifier
from src.data_loader import SentimentDataset, pad_collate_fn
from src.train import train_model
from src.utils import read_sentiment_examples, build_vocab

if __name__ == "__main__":
    # Load data and create DataLoader
    train_examples = read_sentiment_examples('data/train.txt')
    val_examples = read_sentiment_examples('data/dev.txt')
    vocab = build_vocab(train_examples)

    train_dataset = SentimentDataset(train_examples, vocab, PAD_idx=0, UNK_idx=1)
    val_dataset = SentimentDataset(val_examples, vocab, PAD_idx=0, UNK_idx=1)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate_fn)

    # Initialize model, optimizer
    model = FeedForwardNeuralNetClassifier(vocab_size=len(vocab), emb_dim=300, n_hidden_units=300)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_model(model, train_loader, val_loader, optimizer, n_epochs=20, device=device)

    