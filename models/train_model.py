import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


dataset_path = r"C:\Users\sai\Desktop\hackifest\dataset\toxic_nontoxic_dataset_updated.csv"
data = pd.read_csv(dataset_path)


data.columns = data.columns.str.strip()  
print("Columns in dataset:", data.columns)
print(data.head())


print("Missing values:\n", data.isnull().sum())

train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
print(f"Training data size: {len(train_data)}")
print(f"Validation data size: {len(val_data)}")


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(list(train_data['Comment']), padding=True, truncation=True, max_length=512, return_attention_mask=True)
val_encodings = tokenizer(list(val_data['Comment']), padding=True, truncation=True, max_length=512, return_attention_mask=True)

class ToxicCommentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.input_ids = encodings['input_ids']
        self.attention_masks = encodings['attention_mask']  
        self.labels = labels.tolist() 

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx]),
            "attention_mask": torch.tensor(self.attention_masks[idx]), 
            "labels": torch.tensor(self.labels[idx])
        }


train_dataset = ToxicCommentDataset(train_encodings, train_data['Label'])
val_dataset = ToxicCommentDataset(val_encodings, val_data['Label'])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.to(device) 
optimizer = AdamW(model.parameters(), lr=1e-5)


def train_model(model, train_loader, optimizer, device):
    model.train()
    for epoch in range(3):
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

        
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

           
            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item())

train_model(model, train_loader, optimizer, device)


model.save_pretrained('toxic_comment_model')


def evaluate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            predicted_labels = torch.argmax(logits, dim=-1)
            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")


evaluate_model(model, val_loader, device)
model.save_pretrained('toxic_comment_model')
print("model saved successfully")
