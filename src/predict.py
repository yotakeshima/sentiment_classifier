import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import pandas as pd
from src.utils import preprocess_sentence_BERT

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

# Function to predict sentiment for a single sentence
def bert_predict(model, sentence, tokenizer, max_len):
    encoding = tokenizer.encode_plus(
        sentence,
        max_length=max_len,
        truncation=True,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to('cuda' if torch.cuda.is_available() else 'cpu')
    attention_mask = encoding['attention_mask'].to('cuda' if torch.cuda.is_available() else 'cpu')
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits, dim=1)  # Convert logits to probabilities
        _, prediction = torch.max(probs, dim=1)   # Get the index of the max probability
        
    return prediction.item()

def test_on_data(test_df):
    predictions = []

    for _, row in test_df.iterrows():
        sentence = row['sentence']
        true_label = row['label']
        
        predicted_label = bert_predict(sentence)
        
        predictions.append({
            'sentence': sentence,
            'true_label': true_label,
            'predicted_label': predicted_label
        })

    # Convert predictions to a DataFrame
    results_df = pd.DataFrame(predictions)

    # Calculate and print accuracy
    accuracy = accuracy_score(results_df['true_label'], results_df['predicted_label'])
    print(f"Test Set Accuracy: {accuracy:.4f}")

    # Save the DataFrame to an Excel file
    results_df.to_excel('predictions/predictions.xlsx', index=False)

    print("Predictions have been saved to 'predictions/predictions.xlsx'")