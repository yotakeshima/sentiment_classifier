from src.model import BertFFNNC
from src.data_loader import BertDataset
from src.predict import bert_predict
from src.utils import create_dataframe
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import logging

# Suppress specific warnings
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    if "transformers" in logger.name.lower():
        logger.setLevel(logging.ERROR)

file_path = 'data/train.txt'

df = create_dataframe(file_path)

pre_trained_model = 'bert-base-uncased'

tokenizer = BertTokenizer.from_pretrained(pre_trained_model)

kf = KFold(n_splits=10, shuffle=True, random_state=42)
BATCH_SIZE = 32
MAX_LEN = 64
EPOCHS = 10

best_accuracy = 0.0
best_model_state = None

for fold, (train_index, test_index) in enumerate(kf.split(df)):
    print(f"Fold {fold + 1}/{kf.get_n_splits()}")

    train_df = df.iloc[train_index]
    test_df = df.iloc[test_index]

    train_dataset = BertDataset(train_df, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = BertDataset(test_df, tokenizer, MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')
        print("CUDA is not available. Using CPU.")


    model = BertForSequenceClassification.from_pretrained(pre_trained_model, num_labels=2)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss().to('cuda' if torch.cuda.is_available() else 'cpu')
    
   
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to('cuda' if torch.cuda.is_available() else 'cpu')
            attention_mask = batch['attention_mask'].to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = batch['label'].to('cuda' if torch.cuda.is_available() else 'cpu')

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{10}, Loss: {epoch_loss/len(train_loader):.4f}")
    
    model.eval()
    test_labels = []
    test_preds = []


    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to('cuda' if torch.cuda.is_available() else 'cpu')
            attention_mask = batch['attention_mask'].to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = batch['label'].to('cuda' if torch.cuda.is_available() else 'cpu')

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)

            test_labels.extend(labels.cpu().numpy())
            test_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(test_labels, test_preds)
    print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_state = model.state_dict()
    
    print(f"Best Cross-Validation Accuracy: {best_accuracy:.4f}")

if best_model_state is not None:
    torch.save(best_model_state, 'checkpoints/best_bert_model.pth')
    print("Best model saved")