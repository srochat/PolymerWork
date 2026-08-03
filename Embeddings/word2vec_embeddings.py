#!/usr/bin/env python
# coding: utf-8

import json
import numpy as np
import datasets
import torch
import matplotlib.pyplot as plt
from gensim.corpora import Dictionary
from gensim.models import word2vec
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
import torch.nn as nn
from torchcrf import CRF
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap, BoundaryNorm

# ========== Configurations ==========
first_run = False
sequence_length = 450
batch_size = 32

# ========== Utility Functions ==========
def load_json(file):
    with open(file, 'r') as f:
        data = f.read()
    return [json.loads(obj) for obj in data.split('\n') if obj]

def transform_to_dataset_dict(data):
    return {
        "tokens": [obj["words"] for obj in data],
        "ner_tags": [obj["ner"] for obj in data]
    }

def map_labels_to_tags(example):
    example['tags'] = [ner_tag_mapping_flipped[tag] for tag in example['ner_tags']]
    return example

def encode_text(sample):
    sample['input_ids'] = [dictionary.token2id.get(token, 0) + 1 for token in sample["tokens"]]
    return sample

def pad_text(sample):
    length = len(sample['input_ids'])
    if length >= sequence_length:
        sample['input_ids'] = sample['input_ids'][:sequence_length]
        sample['tags'] = sample['tags'][:sequence_length]
        sample['lengths'] = sequence_length
    else:
        pad_len = sequence_length - length
        sample['input_ids'] += [0] * pad_len
        sample['tags'] += [-1] * pad_len
        sample['lengths'] = length
    sample['attention_mask'] = [1] * sample['lengths'] + [0] * (sequence_length - sample['lengths'])
    return sample

def convert_to_data_loader(dataset, num_classes):
    input_tensor = torch.tensor(dataset['input_ids'])
    label_tensor = torch.tensor(dataset['tags']).long()
    mask_tensor = torch.tensor(dataset['attention_mask']).bool()
    tensor_dataset = TensorDataset(input_tensor, label_tensor, mask_tensor)
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)

def get_tokens_and_tags(dataset):
    ids, tokens, tags = [], [], []
    for item in dataset:
        ids.extend(item["input_ids"])
        tokens.extend(item["tokens"])
        tags.extend(item["tags"])
    return ids, tokens, tags

# ========== Load and Prepare Dataset ==========
train_data = load_json('../Data/PolymerAbstracts/train.json')
test_data = load_json('../Data/PolymerAbstracts/test.json')
val_data = load_json('../Data/PolymerAbstracts/dev.json')

dataset_dict = datasets.DatasetDict({
    "train": datasets.Dataset.from_dict(transform_to_dataset_dict(train_data)),
    "test": datasets.Dataset.from_dict(transform_to_dataset_dict(test_data)),
    "val": datasets.Dataset.from_dict(transform_to_dataset_dict(val_data))
})

# ========== Tag Mapping ==========
ner_tag_mapping = {
    0: 'INORGANIC', 1: 'MATERIAL_AMOUNT', 2: 'MONOMER', 3: 'O',
    4: 'ORGANIC', 5: 'POLYMER', 6: 'POLYMER_FAMILY', 7: 'PROP_NAME', 8: 'PROP_VALUE'
}



ner_tag_mapping_flipped = {v: k for k, v in ner_tag_mapping.items()}

# ========== Preprocessing ==========
tokenized_dataset = dataset_dict.map(map_labels_to_tags)
dictionary = Dictionary(tokenized_dataset["train"]["tokens"])

pre_train_dataset = tokenized_dataset["train"].map(encode_text)
rv_l = [len(doc) for doc in pre_train_dataset['input_ids']]

plt.figure(figsize=(6, 4))
plt.hist(rv_l, bins=33, color='skyblue', edgecolor='black')
plt.title('Distribution of Document Lengths', fontsize=16, fontweight='bold')
plt.xlabel('Document Length (tokens)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
#plt.savefig("document_length_distribution.png", dpi=300)
plt.show()

pad_train_dataset = pre_train_dataset.map(pad_text)
train_loader = convert_to_data_loader(pad_train_dataset, len(ner_tag_mapping))

pre_dev_dataset = tokenized_dataset["val"].map(encode_text).map(pad_text)
dev_loader = convert_to_data_loader(pre_dev_dataset, len(ner_tag_mapping))

pre_test_dataset = tokenized_dataset["test"].map(encode_text).map(pad_text)
test_loader = convert_to_data_loader(pre_test_dataset, len(ner_tag_mapping))

# ========== Word2Vec Embedding ==========
word2vec_model = word2vec.Word2Vec(pad_train_dataset['tokens'], vector_size=100, window=5, min_count=1, workers=4)

vocab_size = len(dictionary) + 1
embedding_matrix = torch.zeros((vocab_size, word2vec_model.vector_size))
for i, word in dictionary.items():
    if word in word2vec_model.wv:
        embedding_matrix[i+1] = torch.tensor(word2vec_model.wv[word])


# ========== t-SNE Visualization (remove padding and 'O', single scatter, centered colorbar) ==========
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
from sklearn.manifold import TSNE

NER_COLOR_MAP = {
    "INORGANIC": "#1f77b4",
    "MATERIAL_AMOUNT": "#ff7f0e",
    "MONOMER": "#2ca02c",
    "ORGANIC": "#d62728",
    "POLYMER": "#9467bd",
    "POLYMER_FAMILY": "#8c564b",
    "PROP_NAME": "#e377c2",
    "PROP_VALUE": "#7f7f7f"
}
#tag_names = [ner_tag_mapping[tag] for tag in tsne_tags]


tag_order = [
    0,  # INORGANIC
    1,  # MATERIAL_AMOUNT
    2,  # MONOMER
    3, # O
    4,  # ORGANIC
    5,  # POLYMER
    6,  # POLYMER_FAMILY
    7,  # PROP_NAME
    8,   # PROP_VALUE
]


# Extract tokens and tags
tsne_ids, tsne_tokens, tsne_tags = get_tokens_and_tags(pre_dev_dataset)

# Filter out padding (-1) and "O" tokens
filtered_tokens = []
filtered_tags = []

for token, tag in zip(tsne_tokens, tsne_tags):
    if tag != -1 and tag != 3 and token in word2vec_model.wv:  # 3 is 'O'
        filtered_tokens.append(token)
        filtered_tags.append(tag)

# Word2Vec embeddings for filtered tokens
tsne_embeddings = np.array([word2vec_model.wv[token] for token in filtered_tokens])

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_embeddings_2d = tsne.fit_transform(tsne_embeddings)

# Map tags to string labels
tag_names = [ner_tag_mapping[tag] for tag in filtered_tags]

# color array
colors = np.array([NER_COLOR_MAP[tag] for tag in tag_names])

# Plot
plt.figure(figsize=(12,10))
sc = plt.scatter(
    tsne_embeddings_2d[:,0],
    tsne_embeddings_2d[:,1],
    c=colors,
    s=125,
    alpha=1,
    edgecolor='none'
)

# legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=NER_COLOR_MAP[tag], markersize=10, label=tag)
                   for tag in NER_COLOR_MAP.keys() if tag != "O"]
plt.legend(handles=legend_elements, title="NER Tag", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, title_fontsize=16)

plt.title("Word2Vec Embeddings Projected with t-SNE (All Non-'O' Points)", fontsize=20)
plt.xlabel("t-SNE Dimension 1", fontsize=20)
plt.ylabel("t-SNE Dimension 2", fontsize=20)
plt.tick_params(axis='both', labelsize=18)
plt.grid(True)
plt.tight_layout()

# save
plt.savefig("cleaned_word2vec_embeddings_all_points.png", dpi=300, bbox_inches='tight', pad_inches=0.1)


plt.show()

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math

# Exclude 'O'
highlight_tags = [tag for tag in NER_COLOR_MAP.keys() if tag != "O"]
n_tags = len(highlight_tags)

# Determine grid size (e.g., 3 columns)
n_cols = 3
n_rows = math.ceil(n_tags / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*6, n_rows*6), squeeze=False)

grey_color = "#D3D3D3"

for i, highlight_tag in enumerate(highlight_tags):
    row = i // n_cols
    col = i % n_cols
    ax = axes[row, col]

    # Color array: highlighted tag in color, rest in grey
    colors = [NER_COLOR_MAP[tag] if tag == highlight_tag else grey_color for tag in tag_names]

    ax.scatter(
        tsne_embeddings_2d[:,0],
        tsne_embeddings_2d[:,1],
        c=colors,
        s=80,
        alpha=1,
        edgecolor='none'
    )

    # Add legend only for highlighted tag
    legend_elements = [Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=NER_COLOR_MAP[highlight_tag],
                              markersize=10, label=highlight_tag)]
    ax.legend(handles=legend_elements, title="Highlighted NER Tag", fontsize=10, title_fontsize=12)
    ax.set_title(highlight_tag, fontsize=14)
    ax.set_xlabel("t-SNE Dim 1", fontsize=12)
    ax.set_ylabel("t-SNE Dim 2", fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(True)

# Hide empty subplots if any
for j in range(i+1, n_rows*n_cols):
    row = j // n_cols
    col = j % n_cols
    axes[row, col].axis('off')

plt.tight_layout()
plt.suptitle("Word2Vec t-SNE Embeddings: Highlighting Each NER Tag", fontsize=18, y=1.02)
plt.savefig("tsne_word2vec_highlighted_multi_panel.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
