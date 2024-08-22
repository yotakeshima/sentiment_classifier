
import torch
from src.model import FeedForwardNeuralNetClassifier
from src.data_loader import SentimentDataset, DataLoader, pad_collate_fn
from src.utils import  read_sentiment_examples, build_vocab, make_predictions

test_examples = read_sentiment_examples('data/test-blind.txt')
train_examples = read_sentiment_examples('data/train.txt')
vocab = build_vocab(train_examples)

model = FeedForwardNeuralNetClassifier(vocab_size=len(vocab), emb_dim=300, n_hidden_units=300)
model.load_state_dict(torch.load("checkpoints/best_model.ckpt"))
model.eval()

test_dataset = SentimentDataset(train_examples, vocab, PAD_idx=0, UNK_idx=1)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_predictions = []

for batch_inputs, batch_lengths, _ in test_loader:
    # Move the batch inputs to the device (GPU or CPU)
    batch_inputs = batch_inputs.to(device)
    
    # Make predictions for the current batch
    batch_predictions = make_predictions(model, batch_inputs, device=device)
    
    # Collect all predictions
    all_predictions.extend(batch_predictions)