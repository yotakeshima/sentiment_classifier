import torch
import pandas as pd
from src.utils import create_dataframe
from src.predict import bert_predict, test_on_data
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import os
import logging

# Suppress specific warnings
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    if "transformers" in logger.name.lower():
        logger.setLevel(logging.ERROR)

os.makedirs('predictions', exist_ok=True)

print("Loading model...")

file_path = 'data/dev.txt'
test_df = create_dataframe(file_path)

# Load the saved model
PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
model = BertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, num_labels=2)
model.load_state_dict(torch.load('checkpoints/best_bert_model.pth'))
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()

tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

MAX_LEN = 64  # Ensure this matches the max length used during training

def test_single_input():
    while True:
        sentence = input("Enter a sentence to analyze its sentiment (or 'q' to exit): \n")
        if sentence == 'q':
            break
        predicted_label = bert_predict(model, sentence, tokenizer, MAX_LEN)
        sentiment = "Positive" if predicted_label == 1 else "Negative"
        print(f"The predicted sentiment is: {sentiment}")

while True:
    print("Choose an option:")
    print("1. Test the model on the test data")
    print("2. Test the model with a single input from the keyboard")
    
    choice = input("Enter 1 or 2 (or 'q' to exit): ")
    
    if choice == '1':
        test_on_data(test_df)
    elif choice == '2':
        test_single_input()
    elif choice == 'q':
        break
    else:
        print("Invalid choice. Please enter 1 or 2.")
