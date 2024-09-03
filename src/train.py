import torch
import os
from torch.utils.data import DataLoader
from src.evaluate import evaluate_model
from src.model import FeedForwardNeuralNetClassifier
from src.data_loader import BertDataset, pad_collate_fn
from sklearn.model_selection import KFold
import numpy as np


def train_model(model, train_loader, val_loader, optimizer, n_epochs=20, device='cpu'):
    best_acc = 0
    best_epoch = -1
    
    # Ensure the checkpoint directory exists
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for inputs, lengths, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = model.loss_fn(logits, labels.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
        val_acc = evaluate_model(model, val_loader, device)
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1  # Record the best epoch
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            print(f"New best model saved with accuracy: {best_acc:.3f}")

    print(f"Training complete. Best accuracy: {best_acc:.3f} at epoch {best_epoch}.")

def cross_validate_model(model, dataset, n_splits=10, device='cpu'):
    kfold = KFold(n_splits=n_splits, shuffle=True)
    results = []

    for train_idx, val_idx in kfold.split(dataset):
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, collate_fn=pad_collate_fn)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, collate_fn=pad_collate_fn)

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(10):
            epoch_loss = 0
            for inputs, lengths, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(inputs)
                loss = model.loss_fn(logits, labels.float())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{10}, Loss: {epoch_loss/len(train_loader):.4f}")
            
        model.eval()
        accuracy = evaluate_model(model, val_loader, device=device)
        results.append(accuracy)
        print(f"Fold Accuracy: {accuracy:.4f}")

    print(f"Cross-Validation Accuracy: {np.mean(results):.4f}")
    return results

