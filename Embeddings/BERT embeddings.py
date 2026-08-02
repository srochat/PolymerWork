
import json
import torch
from transformers import BertTokenizerFast, BertModel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load tokenizer and BERT model
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
assert hasattr(tokenizer, "is_fast") and tokenizer.is_fast

model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

#Load Datasets from JSON

def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

train_data = load_dataset('../Data/PolymerAbstracts/train.json')
val_data = load_dataset('../Data/PolymerAbstracts/test.json')
test_data = load_dataset('../Data/PolymerAbstracts/dev.json')

print(f"Train size: {len(train_data)} | Val size: {len(val_data)} | Test size: {len(test_data)}")




from tqdm import tqdm
import torch
from collections import defaultdict

token_embeddings = []
token_labels = []

for sample in tqdm(val_data):
    tokens = sample["words"]
    ner_tags = sample["ner"]

    inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
    word_ids = inputs.word_ids(0)

    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state.squeeze(0)  # shape [seq_len, 768]

    previous_word_idx = None
    for i, word_idx in enumerate(word_ids):
        input_id = inputs["input_ids"][0][i]

        if word_idx is None:
            continue  # skip special tokens
        if word_idx == previous_word_idx:
            continue  # skip non-first subwords

        tag = ner_tags[word_idx]
        if tag != "O":
            token_embeddings.append(last_hidden_state[i].cpu().numpy())
            token_labels.append(tag)

        previous_word_idx = word_idx

print(f"Number of token-level embeddings collected: {len(token_embeddings)}")



import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# -----------------------------
# Fixed color mapping for NER tags
# -----------------------------
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


# Convert and filter
token_embeddings = np.array(token_embeddings)
token_labels = np.array(token_labels)

print(f"Number of token-level embeddings (excluding 'O'): {len(token_embeddings)}")


# Remove "O" class
mask = token_labels != "O"
token_embeddings = token_embeddings[mask]
token_labels = token_labels[mask]

# Run t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=None)
embeddings_2d = tsne.fit_transform(token_embeddings)

#Plot
plt.figure(figsize=(12, 10))
unique_labels = sorted(set(token_labels))

for label in unique_labels:
    idx = token_labels == label

    plt.scatter(
        embeddings_2d[idx,0],
        embeddings_2d[idx,1],
        label=label,
        color=NER_COLOR_MAP[label],
        s=125,
        alpha=1
    )


plt.title("Token-level BERT Embeddings Projected with t-SNE (No 'O' Class)")
plt.xlabel("t-SNE Dimension 1", fontsize=20)
plt.ylabel("t-SNE Dimension 2", fontsize=20)
plt.tick_params(axis='both', labelsize=18)
plt.legend(title="NER Tag", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig("myBERTembeddings.png", dpi=300, bbox_inches='tight')
plt.show()

import pandas as pd

# Assuming:
# embeddings_2d is a NumPy array of shape (num_points, 2)
# token_labels is a NumPy array or list of labels with length num_points

# Create a DataFrame
df = pd.DataFrame({
    'x': embeddings_2d[:, 0],
    'y': embeddings_2d[:, 1],
    'label': token_labels
})

# Save to CSV
df.to_csv('embeddings_2d_labels.csv', index=False)



import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math

# Exclude 'O'
highlight_tags = list(NER_COLOR_MAP.keys())
n_tags = len(highlight_tags)

# Grid layout
n_cols = 3
n_rows = math.ceil(n_tags / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*6, n_rows*6), squeeze=False)

grey_color = "#D3D3D3"

for i, highlight_tag in enumerate(highlight_tags):
    row = i // n_cols
    col = i % n_cols
    ax = axes[row, col]

    # Highlight one tag, grey others
    colors = [NER_COLOR_MAP[tag] if tag == highlight_tag else grey_color for tag in token_labels]

    ax.scatter(
        embeddings_2d[:,0],
        embeddings_2d[:,1],
        c=colors,
        s=80,
        alpha=1,
        edgecolor='none'
    )

    legend_elements = [Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=NER_COLOR_MAP[highlight_tag],
                              markersize=10,
                              label=highlight_tag)]

    ax.legend(handles=legend_elements, title="Highlighted Tag")
    ax.set_title(highlight_tag)
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.grid(True)

# Hide unused subplots
for j in range(i+1, n_rows*n_cols):
    row = j // n_cols
    col = j % n_cols
    axes[row, col].axis('off')

plt.tight_layout()
plt.suptitle("BERT t-SNE Embeddings: Highlighting Each NER Tag", fontsize=18)

plt.savefig("tsne_BERT_highlighted_multi_panel.png",
            dpi=300,
            bbox_inches='tight')

plt.show()
