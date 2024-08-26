import torch
import os
from src.model import FeedForwardNeuralNetClassifier
from src.utils import  read_sentiment_examples, build_vocab, pre_process


train_examples = read_sentiment_examples('data/train.txt')
vocab = build_vocab(train_examples)

model = FeedForwardNeuralNetClassifier(vocab_size=len(vocab), emb_dim=300, n_hidden_units=300)
model.load_state_dict(torch.load("checkpoints/best_model.pth"))
model.eval()

PAD_IDX = vocab.index("PAD")
UNK_IDX = vocab.index("UNK")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

while True:
    sentence = input("\nEnter a sentence for sentiment predictions (or type 'quit' to exit): \n")
    if sentence.lower() == 'quit':
        break
        
    processed_test = pre_process(sentence, vocab, PAD_IDX, UNK_IDX)
    test_loader = torch.tensor(processed_test, device=device)
    with torch.no_grad():
        logits = model(test_loader)
        probability = torch.sigmoid(logits).item()
    sentiment = "Positive" if probability >= 0.5 else "Negative"

    print(f"\nPredicted Sentiment: \t{sentiment} \n(Probability: \t{probability:.4f})")

