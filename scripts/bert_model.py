from src.model import BertFFNNC
from src.predict import bert_predict
from transformers import BertTokenizer
import torch

model = BertFFNNC()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

while True:
    sentence = input("\nEnter a sentence for sentiment predictions (or type 'quit' to exit): \n")
    if sentence.lower() == 'quit':
        break
    sentiment, probability = bert_predict(sentence, model, tokenizer, device)

    print(f"\nPredicted Sentiment: \t{sentiment} \n(Probability: \t{probability:.4f})")