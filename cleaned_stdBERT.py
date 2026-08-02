#!/usr/bin/env python
# coding: utf-8

import os
import json
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter, OrderedDict
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig, AutoModelForTokenClassification,
    get_scheduler
)
from transformers.modeling_outputs import TokenClassifierOutput
from torch.optim import AdamW

import datasets
import evaluate
from tqdm.auto import tqdm
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

# Install required packages if first run
first_run = True
if first_run:
    os.system('pip install datasets evaluate')

# Define paths
DATA_PATH = '../Data/PolymerAbstracts'

# Load JSON lines into structured list of dicts
def load_json(file):
    with open(file, 'r') as f:
        lines = f.read().split('\n')
    return [json.loads(line) for line in lines if line.strip()]

# Convert to huggingface dataset format
def transform_to_dataset_dict(data):
    return {
        "tokens": [obj["words"] for obj in data],
        "ner_tags": [obj["ner"] for obj in data]
    }

# Load datasets
train_data = load_json(os.path.join(DATA_PATH, 'train.json'))
test_data = load_json(os.path.join(DATA_PATH, 'test.json'))
val_data = load_json(os.path.join(DATA_PATH, 'dev.json'))

# Build DatasetDict
raw_dataset = datasets.DatasetDict({
    "train": datasets.Dataset.from_dict(transform_to_dataset_dict(train_data)),
    "test": datasets.Dataset.from_dict(transform_to_dataset_dict(test_data)),
    "val": datasets.Dataset.from_dict(transform_to_dataset_dict(val_data))
})

# NER tag mappings
ner_tag_mapping = {
    0: 'INORGANIC', 1: 'MATERIAL_AMOUNT', 2: 'MONOMER', 3: 'O',
    4: 'ORGANIC', 5: 'POLYMER', 6: 'POLYMER_FAMILY', 7: 'PROP_NAME', 8: 'PROP_VALUE'
}
#reverse_tag_mapping = {v: k for k, v in ner_tag_mapping.items()}
label_to_id = {v: k for k, v in ner_tag_mapping.items()}
id_to_label = {i: label for label, i in label_to_id.items()}

# Convert numeric tags to labels
def map_labels_to_tags(example):
    #print(example['ner_tags'])  # <--- Add this for debugging
    #example['tags'] = [ner_tag_mapping[tag] for tag in example['ner_tags']]
    example['tags'] = [label_to_id[tag] for tag in example['ner_tags']]
    return example

# Tokenize and align NER labels
def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples["tokens"], max_length=512, truncation=True,
        padding=True, is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                #label_ids.append(reverse_tag_mapping[label[word_idx]])
                #label_ids.append(label_to_id[label[word_idx]])
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized["labels"] = labels
    return tokenized

# Tokenization
#bert_pretrained = "m3rg-iitd/matscibert"
bert_pretrained = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(bert_pretrained)
tokenized_dataset = raw_dataset.map(map_labels_to_tags).map(tokenize_and_align_labels, batched=True)

# Transition matrix analysis
reverse_tag_mapping = {v: k for k, v in label_to_id.items()}
def find_transition_matrix(data):
    matrix = np.zeros((len(ner_tag_mapping), len(ner_tag_mapping)))
    for example in data:
        for i in range(len(example['tags']) - 1):
            #matrix[reverse_tag_mapping[example['tags'][i]], reverse_tag_mapping[example['tags'][i + 1]]] += 1
            matrix[example['tags'][i], example['tags'][i + 1]] += 1
    return matrix

transition_matrix = find_transition_matrix(tokenized_dataset['train'])
transition_matrix = np.delete(transition_matrix, 3, axis=0)
transition_matrix = np.delete(transition_matrix, 3, axis=1)

# Plot transition heatmap
sns.heatmap(transition_matrix, annot=True,
            xticklabels=[ner_tag_mapping[i] for i in ner_tag_mapping if i != 3],
            yticklabels=[ner_tag_mapping[i] for i in ner_tag_mapping if i != 3])
plt.show()

# Label distribution summary
def count_labels(partition, pad_val=-100):
    return Counter(label for row in partition['labels'] for label in row if label != pad_val)

def get_label_counts_dataframe(datasets):
    data = OrderedDict()
    labels = set()
    for name, partition in datasets.items():
        counts = count_labels(partition)
        total = sum(counts.values())
        for label, count in counts.items():
            labels.add(label)
            if label not in data:
                data[label] = {}
            data[label][f'{name} Abs'] = count
            data[label][f'{name} %'] = 100 * count / total

    df = pd.DataFrame(data).T.fillna(0)
    cols = [f'{n} {m}' for m in ['Abs', '%'] for n in datasets.keys()]
    return df[cols]

print(get_label_counts_dataframe({
    'Train': tokenized_dataset['train'],
    'Val': tokenized_dataset['val'],
    'Test': tokenized_dataset['test']
}))

# Convert to tensor loaders
def convert_to_data_loader(dataset):
    tensors = [torch.tensor(dataset[col]) for col in ["input_ids", "labels", "token_type_ids", "attention_mask"]]
    return DataLoader(TensorDataset(*tensors), batch_size=16)

train_loader = convert_to_data_loader(tokenized_dataset["train"])
dev_loader = convert_to_data_loader(tokenized_dataset["val"])
test_loader = convert_to_data_loader(tokenized_dataset["test"])



# Define model
class MyTaskSpecificCustomModel(nn.Module):
    def __init__(self, checkpoint, num_labels):
        super().__init__()
        config = AutoConfig.from_pretrained(checkpoint, output_hidden_states=True, output_attentions=True)
        self.bert = AutoModel.from_pretrained(checkpoint, config=config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        logits = self.classifier(self.dropout(outputs.last_hidden_state))
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            return TokenClassifierOutput(loss=loss, logits=logits)
        return TokenClassifierOutput(logits=logits)

# Initialize model and training setup
num_labels = len(ner_tag_mapping)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyTaskSpecificCustomModel(bert_pretrained, num_labels).to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0,
    num_training_steps=10 * len(train_loader)
)

# Training loop
metric_acc = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")
progress_bar = tqdm(range(10 * len(train_loader)))

for epoch in range(10):
    model.train()
    for batch in train_loader:
        batch = [b.to(device) for b in batch]
        inputs = {"input_ids": batch[0], "labels": batch[1], "token_type_ids": batch[2], "attention_mask": batch[3]}
        outputs = model(**inputs)
        outputs.loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    pred_all, gold_all = [], []
    for batch in dev_loader:
        batch = [b.to(device) for b in batch]
        with torch.no_grad():
            outputs = model(input_ids=batch[0], token_type_ids=batch[2], attention_mask=batch[3], labels=batch[1])
        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        labels = batch[1].cpu().numpy()
        for p, g in zip(preds, labels):
            for pred_token, gold_token in zip(p, g):
                if gold_token != -100:
                    pred_all.append(pred_token)
                    gold_all.append(gold_token)

    print(f"Epoch {epoch + 1} - Acc: {metric_acc.compute(predictions=pred_all, references=gold_all)['accuracy']:.4f}, "
          f"F1: {metric_f1.compute(predictions=pred_all, references=gold_all, average='macro')['f1']:.4f}")

# Final test report
model.eval()
pred_all, gold_all = [], []
for batch in test_loader:
    batch = [b.to(device) for b in batch]
    with torch.no_grad():
        outputs = model(input_ids=batch[0], token_type_ids=batch[2], attention_mask=batch[3], labels=batch[1])
    preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
    labels = batch[1].cpu().numpy()
    for p, g in zip(preds, labels):
        for pred_token, gold_token in zip(p, g):
            if gold_token != -100:
                pred_all.append(pred_token)
                gold_all.append(gold_token)

print(classification_report(gold_all, pred_all, target_names=ner_tag_mapping.values(), digits=3))

from sklearn.metrics import classification_report

# Final validation report
model.eval()
pred_all, gold_all = [], []

for batch in dev_loader:
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
