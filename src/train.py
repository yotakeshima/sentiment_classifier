import torch
import os
from torch.utils.data import DataLoader
from src.evaluate import evaluate_model
from src.model import FeedForwardNeuralNetClassifier
from src.data_loader import pad_collate_fn

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
