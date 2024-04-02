import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
from clean import cleaner
from torch.utils.data import DataLoader



data = pd.read_csv("Train.csv")
data = cleaner(data).to_csv("data.csv")

dataset = load_dataset('csv', data_files='data.csv')
model_name = 'drive/MyDrive/HuggingFace/SwahBERT'
model = BertForSequenceClassification.from_pretrained(model_name)

# LOAD AND ACTIVATE ADAPTER

qa_adapter = model.load_adapter("qa/squad1@ukp", config="houlsby")
model.set_active_adapters(qa_adapter)
model.load_adapter("./sentiment_adapter")
model.activate_adapter("sentiment_adapter")


tokenizer = BertTokenizer.from_pretrained(model_name)
def tokenize_batch(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)


train_dataset = dataset['train']
eval_dataset = dataset['test']


batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle = True)


training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=batch_size,
    num_train_epochs=3,
    logging_dir="./logs",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    logging_strategy="epoch"
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.argmax(predictions, dim=1)
    return {"accuracy": (predictions == labels).float().mean().item()}


trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_loader,
    eval_dataset=eval_loader,
    compute_metrics=compute_metrics
)


trainer.train()


results = trainer.evaluate(eval_loader)
print(results)

# ALTERNATIVELY WE CAN USE THE LOOP IN model.py

