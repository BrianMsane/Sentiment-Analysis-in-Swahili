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
import adapters

# LOAD PD, CLEAN AND TRANSFORM TO DATASET
data = pd.read_csv("Train.csv")
data = cleaner().to_csv("data.csv")
dataset = load_dataset('csv', data_files = 'data.csv')

# Define the adapter type and configuration
adapter_type = AdapterType.text_task
adapter_config = {"adapter_name": "sentiment_adapter", "non_linearity": "gelu"}


model_name = 'drive/MyDrive/HuggingFace/SwahBERT'
model = BertForSequenceClassification.from_pretrained(model_name)
adapters.init(model)
tokenizer = BertTokenizer.from_pretrained(model_name)



training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    logging_strategy="epoch"
)

# Define the training function
def train_adapter():
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = torch.argmax(predictions, dim=1)
        return {"accuracy": (predictions == labels).float().mean().item()}

    # Define the adapter setup
    adapter_setup = Stack([ActivationControl.setup_adapter_adapter_composition(),
                           Fuse("adapters")])

    # Train the adapter
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        adapter_setup=adapter_setup
    )
    trainer.train()
    

# Run the training function
train_adapter()



# SAVE AND PUSH TO HUB
adapter_setup = Stack([ActivationControl.setup_adapter_adapter_composition(),
                        Fuse("adapters")])
model.save_adapter("./sentiment_adapter", config=adapter_config)
model.push_to_hub("sentiment_adapter", adapter_setup=adapter_setup)
