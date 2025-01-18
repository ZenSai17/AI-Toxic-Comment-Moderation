from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import torch
import pandas as pd
import os

model = AutoModelForSequenceClassification.from_pretrained('./toxic_comment_model')
tokenizer = AutoTokenizer.from_pretrained('./toxic_comment_model')


dataset = load_dataset('csv', data_files='./dataset/labeled_toxic_comments.csv', split='train')


def preprocess_function(examples):
    return tokenizer(examples['comment_text'], padding="max_length", truncation=True)


encoded_dataset = dataset.map(preprocess_function, batched=True)

predictions = []
texts = []


for example in encoded_dataset:
    inputs = tokenizer(example['comment_text'], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item() 
    predictions.append('Toxic Comment' if predicted_class == 1 else 'Non-Toxic Comment')
    texts.append(example['comment_text'])


results_df = pd.DataFrame({
    'text': texts,
    'prediction': predictions
})


if not os.path.exists('results'):
    os.makedirs('results')


results_df.to_csv('results/toxic_comment_predictions.csv', index=False)

print("Results saved to:", 'results/toxic_comment_predictions.csv')
