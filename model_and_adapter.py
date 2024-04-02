import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
from transformers.adapters.composition import Fuse
from transformers.adapters.control import ActivationControl
from transformers.adapters.composition import Stack
from transformers import AdapterType
from clean import cleaner
from torch.utils.data import DataLoader
from model import train_and_eval



# Load the dataset
dataset = load_dataset('csv', data_files='reviews.csv')

# Assuming the dataset has three classes and the labels are in a column named 'label'

# Load the pre-trained BERT model
model_name = 'drive/MyDrive/HuggingFace/SwahBERT'
model = BertForSequenceClassification.from_pretrained(model_name)

# LOAD AND ACTIVATE ADAPTER
model.load_adapter("./sentiment_adapter")
model.activate_adapter("sentiment_adapter")

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)
def tokenize_batch(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

# Split the dataset into training and evaluation sets
train_dataset = dataset['train']
eval_dataset = dataset['test']

# Define batch size and create data loaders
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle = True)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=batch_size,
    num_train_epochs=3,
    logging_dir="./logs",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    logging_strategy="epoch"
)

# Define the evaluation function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.argmax(predictions, dim=1)
    return {"accuracy": (predictions == labels).float().mean().item()}

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_loader,
    eval_dataset=eval_loader,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate(eval_loader)
print(results)

# ALTERNATIVELY


train_and_eval(3, 3)