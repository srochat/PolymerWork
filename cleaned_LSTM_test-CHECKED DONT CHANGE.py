# python
# coding: utf-8

import json
import numpy as np
import datasets
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from gensim.corpora import Dictionary
from gensim.models import Word2Vec
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score
import pandas as pd

# --- Utility Functions --- #

def load_json(file):
    with open(file, 'r') as f:
        lines = f.read().split('\n')
    return [json.loads(obj) for obj in lines if obj]

def transform_to_dataset_dict(data):
    return {"tokens": [obj["words"] for obj in data], "ner_tags": [obj["ner"] for obj in data]}

# Load data
train_data = load_json('../Data/PolymerAbstracts/train.json')
test_data = load_json('../Data/PolymerAbstracts/test.json')
val_data = load_json('../Data/PolymerAbstracts/dev.json')

dataset_dict = datasets.DatasetDict({
    "train": datasets.Dataset.from_dict(transform_to_dataset_dict(train_data)),
    "test": datasets.Dataset.from_dict(transform_to_dataset_dict(test_data)),
    "val": datasets.Dataset.from_dict(transform_to_dataset_dict(val_data))
})

# --- Label Mapping --- #
ner_tag_mapping = {
    0: 'INORGANIC', 1: 'MATERIAL_AMOUNT', 2: 'MONOMER', 3: 'O', 4: 'ORGANIC',
    5: 'POLYMER', 6: 'POLYMER_FAMILY', 7: 'PROP_NAME', 8: 'PROP_VALUE'
}
ner_tag_mapping_flipped = {v: k for k, v in ner_tag_mapping.items()}

def map_labels_to_tags(example):
    example['tags'] = [ner_tag_mapping_flipped[tag] for tag in example['ner_tags']]
    return example

tokenized_dataset = dataset_dict.map(map_labels_to_tags)

# --- Dictionary & Encoding --- #
dictionary = Dictionary(tokenized_dataset["train"]["tokens"])

def encode_text(sample):
    sample['input_ids'] = [dictionary.token2id.get(token, 0) + 1 for token in sample["tokens"]]
    return sample

pre_train_dataset = tokenized_dataset["train"].map(encode_text)

# --- Padding --- #
sequence_length = 450

def pad_text(sample):
    pad_len = sequence_length - len(sample['input_ids'])
    if pad_len > 0:
        sample['input_ids'] = [0]*pad_len + sample['input_ids']
        sample['tags'] = [-1]*pad_len + sample['tags']
    else:
        sample['input_ids'] = sample['input_ids'][:sequence_length]
        sample['tags'] = sample['tags'][:sequence_length]
    return sample

pad_train_dataset = pre_train_dataset.map(pad_text)

# --- DataLoader --- #
batch_size = 32

def convert_to_data_loader(dataset):
    inputs = torch.tensor(dataset['input_ids'])
    labels = torch.tensor(dataset['tags']).long()
    return DataLoader(TensorDataset(inputs, labels), batch_size=batch_size, shuffle=False)

num_classes = len(ner_tag_mapping)
train_loader = convert_to_data_loader(pad_train_dataset)

# Validation/Test Loaders
pad_dev_dataset = tokenized_dataset["val"].map(encode_text).map(pad_text)
dev_loader = convert_to_data_loader(pad_dev_dataset)

pad_test_dataset = tokenized_dataset["test"].map(encode_text).map(pad_text)
test_loader = convert_to_data_loader(pad_test_dataset)

# --- Word2Vec Embedding --- #
word2vec_model = Word2Vec(pad_train_dataset['tokens'], vector_size=100, window=5, min_count=1, workers=4)
embedding_matrix = torch.zeros((len(dictionary) + 1, word2vec_model.vector_size))
for i, word in dictionary.items():
    if word in word2vec_model.wv:
        embedding_matrix[i+1] = torch.tensor(word2vec_model.wv[word])

##TSNE##

from sklearn.manifold import TSNE

# Step 1: Convert token IDs to Word2Vec embeddings and average over each sequence
def embed_sentences_with_word2vec(pad_dataset):
    sentence_embeddings = []
    sentence_labels = []

    for tokens, tags in zip(pad_dataset['input_ids'], pad_dataset['tags']):
        vectors = []
        labels = []

        for token_id, label in zip(tokens, tags):
            if token_id != 0:  # skip padding
                word = dictionary[token_id - 1]  # reverse lookup
                if word in word2vec_model.wv:
                    vectors.append(word2vec_model.wv[word])
                    labels.append(label)

        if vectors:
            sentence_embeddings.append(np.mean(vectors, axis=0))  # average over tokens
            sentence_labels.append(max(set(labels), key=labels.count))  # majority label as proxy

    return np.array(sentence_embeddings), np.array(sentence_labels)

# Step 2: Generate embeddings for validation set
val_embeddings, val_labels = embed_sentences_with_word2vec(pad_dev_dataset)

# Step 3: t-SNE projection
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
val_emb_2d = tsne.fit_transform(val_embeddings)

# Step 4: Plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(val_emb_2d[:, 0], val_emb_2d[:, 1], c=val_labels, cmap='tab10', alpha=0.7)
plt.legend(*scatter.legend_elements(), title="NER Tags", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('t-SNE of Validation Set (Word2Vec Avg Embeddings Before Training)')
plt.xlabel('t-SNE-1')
plt.ylabel('t-SNE-2')
plt.grid(True)
plt.tight_layout()
plt.show()


# --- Model --- #
class LSTMTextClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes, embedding_matrix, dropout_prob=0.25): #replace 0 by 0.25; dropout prevents overfitting
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False, padding_idx=0)
        self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_size, bidirectional=False, batch_first=True)
        #modify bidirectional argument if needed between true and false
        self.dropout = nn.Dropout(dropout_prob)
        self.output = nn.Linear(hidden_size, num_classes) # *2 only for biLSTM

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(self.dropout(x))
        x = self.dropout(torch.tanh(x))
        return self.output(x).transpose(1, 2)

model = LSTMTextClassifier(hidden_size=32, num_classes=num_classes, embedding_matrix=embedding_matrix) # replace 32 by 64 for biLSTM

# --- Training --- #
def train_model(model, train_loader, dev_loader, num_epochs, weight_tensor, patience=3):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_epochs)
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=-1)  # loss weighting active; loss weighting gives more importance to minority classes
    #loss_fn = nn.CrossEntropyLoss(ignore_index=-1) # no loss weighting active

    train_losses, dev_losses = [], []

    best_val_loss = float('inf')
    counter = 0  # counts epochs with no improvement

    for epoch in range(num_epochs):
        model.train()
        total_correct, total_seen, epoch_loss = 0, 0, []

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(1)
            mask = labels != -1
            total_correct += (preds[mask] == labels[mask]).sum().item()
            total_seen += mask.sum().item()
            epoch_loss.append(loss.item())

        scheduler.step()
        train_acc = total_correct / total_seen * 100
        train_losses.append(np.mean(epoch_loss))

        model.eval()
        val_correct, val_seen, val_loss = 0, 0, []
        with torch.no_grad():
            for inputs, labels in dev_loader:
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                preds = outputs.argmax(1)
                mask = labels != -1
                val_correct += (preds[mask] == labels[mask]).sum().item()
                val_seen += mask.sum().item()
                val_loss.append(loss.item())

        avg_val_loss = np.mean(val_loss)
        dev_losses.append(avg_val_loss)
        val_acc = val_correct / val_seen * 100

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_losses[-1]:.4f}, Accuracy: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

        # Early stopping check
        #if avg_val_loss < best_val_loss:
         #   best_val_loss = avg_val_loss
          #  counter = 0
            # Optionally save your best model here if you want
            # torch.save(model.state_dict(), 'best_model.pt')
        #else:
         #   counter += 1
          #  if counter >= patience:
           #     print(f"Early stopping triggered at epoch {epoch+1}")
            #    break

    return model, train_losses, dev_losses


# --- Weights --- #
all_labels = [label for doc in pad_train_dataset['tags'] for label in doc if label != -1]
class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
weight_tensor = torch.tensor(class_weights, dtype=torch.float)

trained_model, train_losses, dev_losses = train_model(model, train_loader, dev_loader, num_epochs=20, weight_tensor=weight_tensor) #num_epochs originally 20

# --- Evaluation --- #
def predict(model, data_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs).argmax(1)
            mask = labels != -1
            all_preds.extend(outputs[mask].tolist())
            all_labels.extend(labels[mask].tolist())
    return all_labels, all_preds

gold, pred = predict(trained_model, dev_loader)
print(classification_report(gold, pred, target_names=ner_tag_mapping.values(), digits=3))

# --- Report as DataFrame --- #
df_report = pd.DataFrame(classification_report(gold, pred, target_names=ner_tag_mapping.values(), output_dict=True)).transpose()
print(df_report)
df_report.to_csv("classification_report.csv", index=True)

# --- Loss Plot --- #
plt.plot(train_losses, label='Train')
plt.plot(dev_losses, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.title('Training vs Validation Loss')
plt.show()