#!/usr/bin/env python
# coding: utf-8

import os
import json
import numpy as np
import datasets
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter, OrderedDict
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_scheduler
)
import datasets

from torch import nn
from torch.optim import AdamW
from tqdm.auto import tqdm
import evaluate
from sklearn.metrics import confusion_matrix
from seqeval.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report as seqeval_classification_report
)
from datetime import datetime

MAX_LEN = 512

# ================= DEBUG CONFIG =================
class DebugConfig:
    ENABLED = False
    TRAIN_SAMPLES = 20
    VAL_SAMPLES = 10
    TEST_SAMPLES = 10
    BATCH_SIZE = 2
    MAX_TRAIN_BATCHES = 2
    MAX_EVAL_BATCHES = 1
    EPOCHS = 1


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(timestamp)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------- DATA LOADING -----------------
#def load_json(file):
#    with open(file, 'r') as f:
#        data = f.read()
#    objects = data.split('\n')
#    json_objects = []
#    for obj in objects:
#        if obj:
#            json_objects.append(json.loads(obj))
#    return json_objects


#def transform_to_dataset_dict(data):
#    return {
#        "tokens": [obj["words"] for obj in data],
#        "ner_tags": [obj["ner"] for obj in data]
#    }


#train_data = load_json('../Data/PolymerAbstracts/train.json')
#test_data = load_json('../Data/PolymerAbstracts/test.json')
#val_data = load_json('../Data/PolymerAbstracts/dev.json')

#dataset_dict = datasets.DatasetDict({
#    "train": datasets.Dataset.from_dict(transform_to_dataset_dict(train_data)),
#    "test": datasets.Dataset.from_dict(transform_to_dataset_dict(test_data)),
#    "val": datasets.Dataset.from_dict(transform_to_dataset_dict(val_data))
#})

def read_dataset(dataset_path):
    with open(dataset_path, 'r') as dataset_file:
        dataset_json = json.load(dataset_file)
    
    return dataset_json

def merge_es_to_o(data):
    for i in range(len(data["label"])):
        for j in range(len(data["label"][i])):
            tag = data["label"][i][j]

            # Case 1: tag is a string
            if isinstance(tag, str):
                if tag == "ES" or tag.endswith("-ES"):
                    data["label"][i][j] = "O"

            # Case 2: tag is a list (your case)
            elif isinstance(tag, list):
                new_tag = []
                for t in tag:
                    if t == "ES" or t.endswith("-ES"):
                        new_tag.append("O")
                    else:
                        new_tag.append(t)
                data["label"][i][j] = new_tag

    return data


val_json = merge_es_to_o(read_dataset('../../Data/Cleaned_data/Final_v2/validation_split.txt'))
train_json = merge_es_to_o(read_dataset('../../Data/Cleaned_data/Final_v2/train_split.txt'))
test_json = merge_es_to_o(read_dataset('../../Data/Cleaned_data/Final_v2/test_split.txt'))
#val_json = read_dataset('../../Data/Cleaned_data/Final_v2/validation_split.txt')
#train_json = read_dataset('../../Data/Cleaned_data/Final_v2/train_split.txt')
#test_json = read_dataset('../../Data/Cleaned_data/Final_v2/test_split.txt')

#def transform_to_dataset_dict(data):
#    dataset = {
#        "tokens": [],
#        "ner_tags": []
#    }
#    text = data['text']
#    labels = data['label']
#    for i,section in enumerate(text):
#        for j in range(len(section)):
#            dataset['tokens'].append(section[j])
#            dataset['ner_tags'].append(labels[i][j])
#    return dataset

def transform_to_dataset_dict(data):
    tokens = []
    ner_tags = []

    for i in range(len(data["text"])):
        for j in range(len(data["text"][i])):
            tokens.append(data["text"][i][j])
            ner_tags.append(data["label"][i][j])

    #for i in range(len(data["text"])):
    #    flat_tokens = [tok for sent in data["text"][i] for tok in sent]
    #    flat_labels = [lab for sent in data["label"][i] for lab in sent]

    #    tokens.append(flat_tokens)
    #    ner_tags.append(flat_labels)

    return {
        "tokens": tokens,
        "ner_tags": ner_tags
    }

dataset_dict = datasets.DatasetDict({
    'train': datasets.Dataset.from_dict(transform_to_dataset_dict(train_json)),
    'test': datasets.Dataset.from_dict(transform_to_dataset_dict(test_json)),
    'val': datasets.Dataset.from_dict(transform_to_dataset_dict(val_json))
})

# ----------------- LABEL MAPPING -----------------
#ner_tag_mapping = {
#    0: 'MATERIAL', 1: 'CONDITION', 2: 'PROPERTY NAME', 3: 'PROPERTY VALUE',
#    4: 'O'
#}
#ner_tag_mapping_flipped = {v: k for k, v in ner_tag_mapping.items()}


#def convert_to_bio(tags):
#    bio_tags = []
#    prev_tag = "O"

#    for tag in tags:
#        if tag == "O":
#            bio_tags.append("O")
#            prev_tag = "O"
#        else:
#            if tag != prev_tag:
#                bio_tags.append(f"B-{tag}")
#            else:
#                bio_tags.append(f"I-{tag}")
#            prev_tag = tag

#    return bio_tags


#def map_labels_to_tags(example):
#    raw_tags = [ner_tag_mapping_flipped[t] for t in example['ner_tags']]
#    example['tags'] = convert_to_bio(raw_tags)
#    return example


# build label set
#unique_labels = set()
#for split in ["train", "val", "test"]:
#    for tags in dataset_dict[split]["ner_tags"]:
#        raw = [ner_tag_mapping_flipped[t] for t in tags]
#        bio = convert_to_bio(raw)
#        unique_labels.update(bio)

#labels = set()
#for label in dataset_dict['train']['ner_tags']:
#    labels.update(label)

labels = set()

for split in ["train", "val", "test"]:
    for seq in dataset_dict[split]["ner_tags"]:   # seq should be a list
        if isinstance(seq, list):
            labels.update(seq)
        else:
            raise ValueError(f"Unexpected type: {type(seq)}")

print(type(dataset_dict['train']['ner_tags'][0]))
print(dataset_dict['train']['ner_tags'][0][:5])
#ner_tag_mapping = {i: label for i, label in enumerate(labels)}
#ner_tag_mapping_flipped = dict(map(reversed, ner_tag_mapping.items()))

# map all labels to their real tags
#def map_labels_to_tags(example):
#    example['tags'] = [ner_tag_mapping_flipped[tag] for tag in example['ner_tags']]
#    return example

#label_list = sorted(list(unique_labels))
#label2id = {l: i for i, l in enumerate(label_list)}
#id2label = {i: l for l, i in label2id.items()}
label_list = sorted(labels)
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

print('B-ES' in label2id)
print(label2id)

# ----------------- TOKENIZATION -----------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
label_all_tokens = True


def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        is_split_into_words=True
    )

    labels = []

    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous:
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(
                    label2id[label[word_idx]] if label_all_tokens else -100
                )
            previous = word_idx

        labels.append(label_ids)

    tokenized["labels"] = labels
    return tokenized


#tokenized_dataset = dataset_dict.map(map_labels_to_tags).map(
#    tokenize_and_align_labels,
#    batched=True
#)
tokenized_dataset = dataset_dict.map(
    tokenize_and_align_labels,
    batched=True
)


bert_data = tokenized_dataset.remove_columns(['tokens', 'ner_tags'])
bert_data.set_format("torch")


# ----------------- DATALOADERS -----------------
def apply_debug_dataset(data):
    if not DebugConfig.ENABLED:
        return data["train"], data["val"], data["test"]

    return (
        data["train"].select(range(DebugConfig.TRAIN_SAMPLES)),
        data["val"].select(range(DebugConfig.VAL_SAMPLES)),
        data["test"].select(range(DebugConfig.TEST_SAMPLES)),
    )


train_ds, val_ds, test_ds = apply_debug_dataset(bert_data)

batch_size = DebugConfig.BATCH_SIZE if DebugConfig.ENABLED else 16

train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
eval_dataloader = DataLoader(val_ds, batch_size=batch_size)
test_dataloader = DataLoader(test_ds, batch_size=batch_size)


# ----------------- MODEL (BERT ONLY) -----------------
class BertTokenClassifier(nn.Module):
    def __init__(self, checkpoint, num_labels):
        super().__init__()
        self.config = AutoConfig.from_pretrained(checkpoint)
        self.bert = AutoModel.from_pretrained(checkpoint, config=self.config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_fn(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1)
            )

        return {"loss": loss, "logits": logits}


model = BertTokenClassifier(
    checkpoint="bert-base-uncased",
    num_labels=len(label_list)
).to(device)


# ----------------- OPTIMIZER -----------------
optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = DebugConfig.EPOCHS if DebugConfig.ENABLED else 20
num_training_steps = num_epochs * len(train_dataloader)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)


# ----------------- TRAINING -----------------
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    train_bar = tqdm(
        train_dataloader,
        desc=f"Epoch {epoch+1}/{num_epochs} [Training]",
        leave=False
    )

    for step, batch in enumerate(train_bar):
        if DebugConfig.ENABLED and step >= DebugConfig.MAX_TRAIN_BATCHES:
            break

        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        train_bar.set_postfix(loss=loss.item())

    #print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss:.4f}")


# ----------------- EVALUATION (SEQEVAL) -----------------
def collect_seqeval(dataloader):
    model.eval()
    y_true, y_pred = [], []

    ignored_count = 0
    total_count = 0

    with torch.no_grad():
        eval_bar = tqdm(dataloader, desc=f"Evaluating", leave=False)

        for batch in eval_bar:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
            )

            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=-1)
             
            labels = batch["labels"].cpu().numpy()

            # ---- DEBUG BLOCK ----
            ignored_count += np.sum(labels == -100)
            total_count += labels.size
            # ---------------------

            print("Sample predictions:", preds[0][:20])
            print("Sample gold:", batch["labels"][0][:20])

            golds = batch["labels"].cpu().tolist()
            preds = preds.cpu().tolist()
            masks = batch["attention_mask"].cpu().tolist()

            for p_seq, g_seq, m_seq in zip(preds, golds, masks):
                true_seq, pred_seq = [], []

                for p, g, m in zip(p_seq, g_seq, m_seq):
                    if m == 1 and g != -100:
                        true_seq.append(id2label[g])
                        pred_seq.append(id2label[p])

                if true_seq:
                    y_true.append(true_seq)
                    y_pred.append(pred_seq)

    print("Ignored tokens (-100):", ignored_count)
    print("Total tokens:", total_count)
    print("Ignored %:", ignored_count / total_count * 100)

    return y_true, y_pred


def print_seqeval(dataloader, name):
    y_true, y_pred = collect_seqeval(dataloader)

    print(f"\n===== {name} =====")
    print(seqeval_classification_report(y_true, y_pred))
    print("F1:", f1_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))


print_seqeval(test_dataloader, "TEST")
print_seqeval(eval_dataloader, "VALIDATION")


def flatten_for_confusion(y_true, y_pred, label2id):
    y_true_flat = []
    y_pred_flat = []

    for true_seq, pred_seq in zip(y_true, y_pred):
        for t, p in zip(true_seq, pred_seq):
            y_true_flat.append(label2id[t])
            y_pred_flat.append(label2id[p])

    return y_true_flat, y_pred_flat


y_true, y_pred = collect_seqeval(test_dataloader)

y_true_flat, y_pred_flat = flatten_for_confusion(
    y_true, y_pred, label2id
)




def plot_and_save_confusion_matrices(y_true, y_pred, label_mapping, prefix="test"):
    labels = list(range(len(label_mapping)))
    label_names = [label_mapping[i] for i in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_true_norm = np.nan_to_num(cm.astype(float) / cm.sum(axis=1, keepdims=True))
    cm_pred_norm = np.nan_to_num(cm.astype(float) / cm.sum(axis=0, keepdims=True))

    df_true = pd.DataFrame(cm_true_norm, index=label_names, columns=label_names)
    df_pred = pd.DataFrame(cm_pred_norm, index=label_names, columns=label_names)

    #df_true.to_csv(f"{prefix}_confusion_matrix_true_normalized.csv")
    #df_pred.to_csv(f"{prefix}_confusion_matrix_pred_normalized.csv")
    df_true.to_csv(f"{prefix}_{timestamp}_confusion_matrix_true_normalized.csv")
    df_pred.to_csv(f"{prefix}_{timestamp}_confusion_matrix_pred_normalized.csv")

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_true, annot=True, fmt=".2f", cmap="Blues")
    plt.title("Confusion Matrix (Normalized by TRUE labels)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(f"{prefix}_{timestamp}_confusion_matrix_true_normalized.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_pred, annot=True, fmt=".2f", cmap="Greens")
    plt.title("Confusion Matrix (Normalized by PREDICTED labels)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(f"{prefix}_{timestamp}_confusion_matrix_pred_normalized.png", dpi=300)
    plt.close()

    print(f"Saved confusion matrices with prefix '{prefix}'")

plot_and_save_confusion_matrices(
    y_true_flat,
    y_pred_flat,
    id2label,
    prefix="test"
)
