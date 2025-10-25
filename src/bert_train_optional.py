# src/bert_train_optional.py
"""
Contoh pipeline singkat menggunakan Hugging Face Transformers (IndoBERT).
Butuh paket: transformers, datasets, torch.
Butuh GPU untuk training wajar.

Ini hanya contoh minimal — untuk tugas kamu cukup TF-IDF+NB.
"""
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import pandas as pd

MODEL_NAME = "indobenchmark/indobert-base-p1"  # contoh; cek HF untuk model IndoBERT yang tersedia
df = pd.read_csv("../data/reviews.csv")
ds = Dataset.from_pandas(df)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def preprocess(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
ds = ds.map(preprocess, batched=True)

label2id = {lab:i for i,lab in enumerate(df['label'].unique())}
id2label = {v:k for k,v in label2id.items()}
ds = ds.map(lambda x: {'labels':[label2id[l] for l in x['label']]}, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(label2id), id2label=id2label, label2id=label2id)

training_args = TrainingArguments(
    output_dir="./bert_out",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10
)

trainer = Trainer(model=model, args=training_args, train_dataset=ds.train_test_split(test_size=0.1)['train'], eval_dataset=ds.train_test_split(test_size=0.1)['test'])
trainer.train()
trainer.save_model("./bert_out")
