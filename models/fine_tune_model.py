from datasets import load_dataset, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split
import pandas as pd


data = pd.read_csv("dataset/labeled_toxic_comments.csv")


data = Dataset.from_pandas(data)


data = data.train_test_split(test_size=0.2)


tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


def tokenize_function(example):
    return tokenizer(example["comment_text"], padding="max_length", truncation=True)

tokenized_data = data.map(tokenize_function, batched=True)


tokenized_data = tokenized_data.rename_column("label", "labels")
tokenized_data.set_format("torch", columns=["input_ids", "attention_mask", "labels"])


model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
)


trainer.train()


model.save_pretrained("./toxic_comment_model")
tokenizer.save_pretrained("./toxic_comment_model")

print("Model fine-tuned and saved to './toxic_comment_model'")
