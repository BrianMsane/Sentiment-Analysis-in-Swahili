from clean import cleaner
import os
import shutil
import torch
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split

from torch.utils.data import(
    TensorDataset,
    DataLoader,
    random_split
)
from transformers import(
    get_linear_schedule_with_warmup,
    BertTokenizer,
    BertForSequenceClassification
) 
from torch.optim import AdamW
from sklearn.metrics import(
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    classification_report

) 
import matplotlib.pyplot as plt
tf.get_logger().setlevel("ERROR")


model_name = 'drive/MyDrive/HuggingFace/SwahBERT'
batch_size = 32
max_seq_length = 128
epoch = 5
learning_rate = 2e-5 # # 5e-5
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
num_labels = 3

# DATA
data = pd.read_csv("Train.csv")
data = cleaner(data)
X = data['text'].values
y = data['labels'].values


tokenizer = BertTokenizer.from_pretrained(model_name)
tokenized_texts = tokenizer(list(X), truncation = True, padding = True,
                             return_tensors = 'pt', max_length = max_seq_length,
                             add_special_tokens = True, return_token_type_ids = False)
input_ids = tokenized_texts.input_ids
attension_mask = tokenized_texts.attention_mask

# CONVERT LABELS TO INTEGERS
labels = torch.tensor(y, dtype = torch.long)
dataset = TensorDataset(input_ids, attension_mask, labels)


num_classes = 3
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DATA LOADERS
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)

model = BertForSequenceClassification(model_name, num_labels = num_labels) 
model.to(device)

optimizer = AdamW(model.parameters(), lr = learning_rate, eps = 1e-08, weight_decay = 0.01)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, 
                                            num_training_steps = len(train_loader) * epoch)



# TRAINING LOOP
def train_and_eval(epochs = epoch, num_labels = num_labels):


    for epoch in range(epoch):
        model.train()
        for batch in train_loader:
            input_ids, attention_masks, labels = batch
            input_ids, attention_masks, labels = input_ids.to(device), attention_masks.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_masks, label=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

    # Evaluation

    model.eval()
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(val_labels, val_preds)
    balanced = balanced_accuracy_score(val_labels, val_preds)
    Rec = recall_score(val_labels, val_preds)
    f1 = f1_score(val_labels, val_preds)
    Prec = precision_score(val_labels, val_preds)
    report = classification_report(val_labels, val_preds)

    print(f"Validation Scores \nAccuracy: {accuracy}, \nBalanced accuracy score : {balanced}, \nRecall : {Rec}, \nF1_score :  {F1},\n Precision : {Prec}.")
    print(f"Classification Report: \n {report}")



# save the models
def saving():
    save_directory = 'drive/MyDrive/newModel'
    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)