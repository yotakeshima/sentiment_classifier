import torch
import os
from src.model import FeedForwardNeuralNetClassifier
from src.data_loader import SentimentDataset, DataLoader, pad_collate_fn
from src.utils import  read_sentiment_examples, build_vocab, output_predictions, read_sentiment_sentence
from src.predict import make_predictions

train_examples = read_sentiment_examples('data/train.txt')
vocab = build_vocab(train_examples)

model = FeedForwardNeuralNetClassifier(vocab_size=len(vocab), emb_dim=300, n_hidden_units=300)
model.load_state_dict(torch.load("checkpoints/best_model.pth"))
model.eval()

PAD_IDX = vocab.index("PAD")
UNK_IDX = vocab.index("UNK")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

while True:
    sentence = input("Enter a sentence for sentiment predictions (or type 'quit' to exit): \n")
    if sentence.lower() == 'quit':
        break
        
    sentence = read_sentiment_sentence(sentence)
    test_sentence = SentimentDataset(sentence, vocab, PAD_IDX, UNK_IDX)
    test_loader = DataLoader(test_sentence, batch_size=32, shuffle=False, collate_fn=pad_collate_fn)
    prediction = make_predictions(model, test_loader, device=device)
    sentiment = "Positive" if prediction == 1 else "Negative"

    print(f"Predicted Sentiment: {sentiment} (Probability: {probablity:.4f})")

