#!/usr/bin/env python
# coding: utf-8

import json
import numpy as np
import datasets
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
from torchcrf import CRF
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from gensim.models import word2vec
from gensim.corpora import Dictionary

first_run = False

# Load and parse JSON

def load_json(file):
    with open(file, 'r') as f:
        data = f.read()
    objects = [json.loads(obj) for obj in data.split('\n') if obj]
    return objects

def transform_to_dataset_dict(data):
    return {"tokens": [obj["words"] for obj in data], "ner_tags": [obj["ner"] for obj in data]}

# Load datasets
train_data = load_json('../Data/PolymerAbstracts/train.json')
test_data = load_json('../Data/PolymerAbstracts/test.json')
val_data = load_json('../Data/PolymerAbstracts/dev.json')

dataset_dict = datasets.DatasetDict({
    "train": datasets.Dataset.from_dict(transform_to_dataset_dict(train_data)),
    "test": datasets.Dataset.from_dict(transform_to_dataset_dict(test_data)),
    "val": datasets.Dataset.from_dict(transform_to_dataset_dict(val_data))
})

# Label mappings
ner_tag_mapping = {
    0: 'INORGANIC', 1: 'MATERIAL_AMOUNT', 2: 'MONOMER', 3: 'O',
    4: 'ORGANIC', 5: 'POLYMER', 6: 'POLYMER_FAMILY',
    7: 'PROP_NAME', 8: 'PROP_VALUE'
}
ner_tag_mapping_flipped = {v: k for k, v in ner_tag_mapping.items()}

def map_labels_to_tags(example):
    example['tags'] = [ner_tag_mapping_flipped[tag] for tag in example['ner_tags']]
    return example

# Token preprocessing

tokenized_dataset = dataset_dict.map(map_labels_to_tags)
dictionary = Dictionary(tokenized_dataset["train"]["tokens"])

# Encoding

def encode_text(sample):
    sample['input_ids'] = [dictionary.token2id.get(token, 0) + 1 for token in sample["tokens"]]
    return sample

pre_train_dataset = tokenized_dataset["train"].map(encode_text)

# Padding and truncating
sequence_length = 450

def pad_text(sample):
    length = len(sample['input_ids'])
    sample['lengths'] = min(length, sequence_length)
    sample['input_ids'] = sample['input_ids'][:sequence_length] + [0]*(sequence_length - length)
    sample['tags'] = sample['tags'][:sequence_length] + [-1]*(sequence_length - length)
    sample['attention_mask'] = [1]*sample['lengths'] + [0]*(sequence_length - sample['lengths'])
    return sample

pad_train_dataset = pre_train_dataset.map(pad_text)

# DataLoader conversion

def convert_to_data_loader(dataset):
    inputs = torch.tensor(dataset['input_ids'])
    labels = torch.tensor(dataset['tags']).long()
    masks = torch.tensor(dataset['attention_mask']).bool()
    return DataLoader(TensorDataset(inputs, labels, masks), batch_size=32, shuffle=False)

num_classes = len(ner_tag_mapping)
train_loader = convert_to_data_loader(pad_train_dataset)

# Validation and test sets
pre_dev_dataset = tokenized_dataset["val"].map(encode_text)
pad_dev_dataset = pre_dev_dataset.map(pad_text)
dev_loader = convert_to_data_loader(pad_dev_dataset)

pre_test_dataset = tokenized_dataset["test"].map(encode_text)
pad_test_dataset = pre_test_dataset.map(pad_text)
test_loader = convert_to_data_loader(pad_test_dataset)

# Train Word2Vec model
word2vec_model = word2vec.Word2Vec(pad_train_dataset['tokens'], vector_size=100, window=5, min_count=1, workers=4)

vocab_size = len(dictionary) + 1
embedding_matrix = torch.zeros((vocab_size, word2vec_model.vector_size))
for i, word in dictionary.items():
    if word in word2vec_model.wv:
        embedding_matrix[i+1] = torch.tensor(word2vec_model.wv[word])

# Define the LSTM+CRF model
class RegularizedLSTMTextClassifierWithEmbeddings(nn.Module):
    def __init__(self, hidden_size, num_classes, embedding_matrix, dropout_prob=0.25):
        super().__init__()
        self.embedding_layer = nn.Embedding.from_pretrained(embedding_matrix, freeze=False, padding_idx=0)
        self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_size, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.Tanh()
        self.output_layer = nn.Linear(hidden_size * 2, num_classes)
        self.crf = CRF(num_classes, batch_first=True)

    def forward(self, input_ids, labels=None, mask=None):
        x = self.embedding_layer(input_ids)
        x = self.dropout(x)
        lstm_out, _ = self.lstm(x)
        x = self.dropout(self.activation(lstm_out))
        emissions = self.output_layer(x)
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=mask)
            return loss, None
        else:
            predictions = self.crf.decode(emissions, mask=mask)
            return None, predictions

# Instantiate model
hidden_size = 32
model = RegularizedLSTMTextClassifierWithEmbeddings(hidden_size, num_classes, embedding_matrix)

# Training loop

def train_nn(num_epochs, model, train_loader, dev_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_epochs)
    train_losses, dev_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        for inputs, labels, masks in train_loader:
            optimizer.zero_grad()
            loss, _ = model(inputs, labels, masks)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        scheduler.step()
        train_losses.append(epoch_train_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            epoch_dev_loss = 0
            for inputs, labels, masks in dev_loader:
                loss, _ = model(inputs, labels, masks)
                epoch_dev_loss += loss.item()
            dev_losses.append(epoch_dev_loss / len(dev_loader))

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_losses[-1]:.4f}, Dev Loss: {dev_losses[-1]:.4f}")

    return model, train_losses, dev_losses

# Class weights (optional for use)
train_labels = [label for sublist in pad_train_dataset['tags'] for label in sublist if label != -1]
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
weight_tensor = torch.tensor(class_weights, dtype=torch.float)

# Train model
trained_model, train_losses, dev_losses = train_nn(20, model, train_loader, dev_loader)

# Plotting
plt.plot(train_losses, label='Train')
plt.plot(dev_losses, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Predict

def predict_nn(model, loader):
    model.eval()
    gold, pred = [], []
    with torch.no_grad():
        for inputs, labels, masks in loader:
            _, predictions = model(inputs, None, masks)
            gold.extend(labels[masks].tolist())
            pred.extend([p for batch in predictions for p in batch])
    return gold, pred

gold_labels, pred_labels = predict_nn(trained_model, dev_loader)
print(classification_report(gold_labels, pred_labels, target_names=ner_tag_mapping.values(), digits=3))

# --- Report as DataFrame --- #
# Generate classification report as a dictionary
report_dict = classification_report(gold_labels, pred_labels, target_names=ner_tag_mapping.values(), digits=3, output_dict=True)
# Convert to DataFrame
df_classification_report = pd.DataFrame(report_dict).transpose()
# Save to CSV
df_classification_report.to_csv("classification_report.csv", index=True)
