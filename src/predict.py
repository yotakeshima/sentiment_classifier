import torch

def predict(model, input_tensor, device='cpu'):
    model.eval()
    model.to(device)

    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        logits = model(input_tensor)
        predictions = torch.sigmoid(logits)
        preds = (predictions >= 0.5).int()

    return preds.cpu().numpy().tolist()

def make_predictions(model, test_loader, device='cpu'):

    all_predictions = []

    for batch_inputs, batch_lengths, _ in test_loader:
        batch_inputs = batch_inputs.to(device)
        batch_predictions = predict(model, batch_inputs, device=device)
        all_predictions.extend(batch_predictions)


    return all_predictions