import torch
# Evaluates the model, provides a validation accuracy.
def evaluate_model(model, val_loader, device='cpu'):
    model.eval()
    correct_preds = total_preds = 0
    with torch.no_grad():
        for inputs, lengths, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model.predict(inputs)
            correct_preds += (torch.tensor(preds) == labels).sum().item()
            total_preds += labels.size(0)
    accuracy = correct_preds / total_preds
    print(f"Validation Accuracy: {accuracy:.3f}")
    return accuracy
