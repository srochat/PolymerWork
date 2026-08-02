#!/usr/bin/env python
# coding: utf-8

# --- Imports and Initial Setup ---
import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from collections import Counter, OrderedDict
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig, AutoModelForTokenClassification,
    get_scheduler
)
token = "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"  # your Hugging Face token

from transformers.modeling_outputs import TokenClassifierOutput
from torch.optim import AdamW

import datasets
import evaluate
from tqdm.auto import tqdm
import uuid

first_run = True

if first_run:
    os.system('pip install datasets evaluate')

# --- Load and Prepare Dataset ---
def load_json(file):
    with open(file, 'r') as f:
        lines = f.read().split('\n')
    return [json.loads(line) for line in lines if line]

def transform_to_dataset_dict(data):
    return {"tokens": [x["words"] for x in data], "ner_tags": [x["ner"] for x in data]}

path = '../Data/PolymerAbstracts'
train_data = load_json(f'{path}/train.json')
test_data = load_json(f'{path}/test.json')
val_data = load_json(f'{path}/dev.json')

dataset_dict = datasets.DatasetDict({
    "train": datasets.Dataset.from_dict(transform_to_dataset_dict(train_data)),
    "test": datasets.Dataset.from_dict(transform_to_dataset_dict(test_data)),
    "val": datasets.Dataset.from_dict(transform_to_dataset_dict(val_data))
})

# --- Label Mapping ---
ner_tag_mapping = {
    0: 'INORGANIC', 1: 'MATERIAL_AMOUNT', 2: 'MONOMER', 3: 'O',
    4: 'ORGANIC', 5: 'POLYMER', 6: 'POLYMER_FAMILY', 7: 'PROP_NAME', 8: 'PROP_VALUE'
}
ner_tag_mapping_flipped = {v: k for k, v in ner_tag_mapping.items()}

def map_labels_to_tags(example):
    example['tags'] = [ner_tag_mapping_flipped[tag] for tag in example['ner_tags']]
    return example

# --- Tokenization ---
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_auth_token=token)
label_all_tokens = False

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], max_length=512, truncation=True,
        padding=True, is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = dataset_dict.map(map_labels_to_tags).map(tokenize_and_align_labels, batched=True)

# --- Transition Matrix ---
def find_transition_matrix(data):
    matrix = np.zeros((len(ner_tag_mapping), len(ner_tag_mapping)))
    for example in data:
        for i in range(len(example['tags']) - 1):
            matrix[example['tags'][i], example['tags'][i + 1]] += 1
    return matrix

matrix = find_transition_matrix(tokenized_dataset['train'])
matrix = np.delete(matrix, 3, axis=0)
matrix = np.delete(matrix, 3, axis=1)
labels = [ner_tag_mapping[i] for i in range(len(ner_tag_mapping)) if i != 3]
sns.heatmap(matrix, annot=True, xticklabels=labels, yticklabels=labels)
plt.show()

# --- Label Distribution ---
def count_labels(dataset_partition):
    return Counter(label for row in dataset_partition['labels'] for label in row if label != -100)

def get_label_counts_dataframe(dsets):
    data, label_set = OrderedDict(), set()
    for name, part in dsets.items():
        counts = count_labels(part)
        total = sum(counts.values())
        for label, count in counts.items():
            label_set.add(label)
            if label not in data:
                data[label] = {}
            data[label][f'{name} Abs'] = count
            data[label][f'{name} %'] = count / total * 100
    df = pd.DataFrame.from_dict(data, orient='index').fillna(0)
    columns = [f'{name} {metric}' for metric in ['Abs', '%'] for name in dsets]
    return df[columns]

datasets_split = {"Train": tokenized_dataset['train'], "Val": tokenized_dataset['val'], "Test": tokenized_dataset['test']}
df = get_label_counts_dataframe(datasets_split)
print(df)

# --- DataLoaders ---
def convert_to_data_loader(dataset, batch_size=16):
    return DataLoader(
        TensorDataset(
            torch.tensor(dataset['input_ids']),
            torch.tensor(dataset['labels']),
            torch.tensor(dataset['token_type_ids']),
            torch.tensor(dataset['attention_mask'])
        ), batch_size=batch_size, shuffle=False
    )

train_loader = convert_to_data_loader(tokenized_dataset["train"])
dev_loader = convert_to_data_loader(tokenized_dataset["val"])
test_loader = convert_to_data_loader(tokenized_dataset["test"])

bert_data = tokenized_dataset.remove_columns(['tokens', 'ner_tags', 'tags'])
bert_data.set_format("torch")

train_dataloader = DataLoader(bert_data["train"], shuffle=True, batch_size=16)
eval_dataloader = DataLoader(bert_data["val"], batch_size=16)
test_dataloader = DataLoader(bert_data["test"], batch_size=16)

# --- Model Setup ---
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class MyTaskSpecificCustomModel(nn.Module):
    def __init__(self, checkpoint, num_labels, hidden_size, freeze_bert=False):
        super().__init__()
        self.num_labels = num_labels
        config = AutoConfig.from_pretrained(checkpoint, output_hidden_states=True, output_attentions=True)
        self.bert = AutoModel.from_pretrained(checkpoint, config=config)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.dropout = nn.Dropout(0.1)
        self.hidden_layer = nn.LSTM(config.hidden_size, hidden_size, bidirectional=True, batch_first=True)
        self.output_layer = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        x = self.dropout(outputs[0])
        x, _ = self.hidden_layer(x)
        logits = self.output_layer(x)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
        return TokenClassifierOutput(logits=logits)

model = MyTaskSpecificCustomModel("bert-base-uncased", num_labels=9, hidden_size=64).to(device)

# --- Optimizer and Scheduler ---
bert_params = list(model.bert.parameters())
other_params = list(model.dropout.parameters()) + list(model.hidden_layer.parameters()) + list(model.output_layer.parameters())

optimizer = AdamW([
    {"params": bert_params, "lr": 5e-5},
    {"params": other_params, "lr": 5e-3},
])

num_epochs = 10
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# --- Training Loop ---
progress_bar = tqdm(range(num_training_steps))
for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        labels = batch['labels'].detach().cpu().numpy()
        valid = labels != -100
        metric_acc.add_batch(predictions=preds[valid], references=labels[valid])
        metric_f1.add_batch(predictions=preds[valid], references=labels[valid])

    #print(f"Epoch {epoch+1} Acc: {metric_acc.compute()['accuracy']:.4f}, F1: {metric_f1.compute(average='macro')['f1']:.4f}")

exp_id_acc = str(uuid.uuid4())
exp_id_f1 = str(uuid.uuid4())

acc = metric_acc.compute()['accuracy']
f1 = metric_f1.compute(average='macro')['f1']

print(f"Epoch {epoch+1} Acc: {acc:.4f}, F1: {f1:.4f}")


# --- Evaluation ---
def predict(loader):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        gold = batch['labels'].detach().cpu().numpy()
        mask = gold != -100
        preds.extend(pred[mask])
        labels.extend(gold[mask])
    return preds, labels

pred, gold = predict(eval_dataloader)
print(classification_report(gold, pred, target_names=ner_tag_mapping.values(), digits=3))


# Final test report
model.eval()
pred_all, gold_all = [], []

for batch in test_loader:
    # batch = [input_ids, labels, token_type_ids, attention_mask]
    input_ids, labels, token_type_ids, attention_mask = [b.to(device) for b in batch]
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
    labels = labels.cpu().numpy()

    # flatten and filter out padding (-100)
    for p_seq, g_seq in zip(preds, labels):
        for p_tok, g_tok in zip(p_seq, g_seq):
            if g_tok != -100:
                pred_all.append(p_tok)
                gold_all.append(g_tok)

print(classification_report(
    gold_all,
    pred_all,
    target_names=list(ner_tag_mapping.values()),
    digits=3
))
