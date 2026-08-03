#!/usr/bin/env python
# coding: utf-8

first_run = True
USE_CRF = False  #  toggle here

# ================= DEBUG CONFIG =================
class DebugConfig:
    ENABLED = False          #  toggle ON/OFF here
    TRAIN_SAMPLES = 20      # smaller dataset size
    VAL_SAMPLES = 10
    TEST_SAMPLES = 10
    BATCH_SIZE = 2
    MAX_TRAIN_BATCHES = 2   # stop early per epoch
    MAX_EVAL_BATCHES = 1
    EPOCHS = 1
if DebugConfig.ENABLED:
    print("Running quick sanity check (NOT full training)")
# ===============================================

import os
import json
import numpy as np
import datasets
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter, OrderedDict
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput
from torchcrf import CRF
from torch import nn
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import evaluate
import itertools
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from seqeval.metrics import f1_score, precision_score, recall_score
from seqeval.metrics import classification_report as seqeval_classification_report
from datetime import datetime


# ----------------- GPU CHECK -----------------
print("CUDA available:", torch.cuda.is_available())
print("Default device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
if torch.cuda.is_available():
    print("Current GPU index:", torch.cuda.current_device())
    print("GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))
# ---------------------------------------------

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


# Load file list
os.listdir('../Data/PolymerAbstracts')

# Helper functions
def load_json(file):
    with open(file, 'r') as f:
        data = f.read()
    objects = data.split('\n')
    json_objects = []
    for obj in objects:
        if obj:
            try:
                json_objects.append(json.loads(obj))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON object: {e}")
    return json_objects

def transform_to_dataset_dict(data):
    dataset = {"tokens": [], "ner_tags": []}
    for obj in data:
        dataset["tokens"].append(obj["words"])
        dataset["ner_tags"].append(obj["ner"])
    return dataset

train_data = load_json('../Data/PolymerAbstracts/train.json')
test_data = load_json('../Data/PolymerAbstracts/test.json')
val_data = load_json('../Data/PolymerAbstracts/dev.json')

dataset_dict = datasets.DatasetDict({
    "train": datasets.Dataset.from_dict(transform_to_dataset_dict(train_data)),
    "test": datasets.Dataset.from_dict(transform_to_dataset_dict(test_data)),
    "val": datasets.Dataset.from_dict(transform_to_dataset_dict(val_data))
})


ner_tag_mapping = {
    0: 'INORGANIC', 1: 'MATERIAL_AMOUNT', 2: 'MONOMER', 3: 'O',
    4: 'ORGANIC', 5: 'POLYMER', 6: 'POLYMER_FAMILY',
    7: 'PROP_NAME', 8: 'PROP_VALUE'
}
ner_tag_mapping_flipped = dict(map(reversed, ner_tag_mapping.items()))


def convert_to_bio(tags):
    bio_tags = []
    prev_tag = "O"

    for tag in tags:
        if tag == "O":
            bio_tags.append("O")
            prev_tag = "O"
        else:
            if tag != prev_tag:
                bio_tags.append(f"B-{tag}")
            else:
                bio_tags.append(f"I-{tag}")
            prev_tag = tag

    return bio_tags


def map_labels_to_tags(example):
    raw_tags = [ner_tag_mapping_flipped[tag] for tag in example['ner_tags']]
    example['tags'] = convert_to_bio(raw_tags)
    return example

unique_labels = set()

for split in ["train", "val", "test"]:
    for tags in dataset_dict[split]["ner_tags"]:
        raw = [ner_tag_mapping_flipped[t] for t in tags]
        bio = convert_to_bio(raw)
        unique_labels.update(bio)

label_list = sorted(list(unique_labels))
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}


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
                #label_ids.append(label[word_idx])
                label_ids.append(label2id[label[word_idx]])
            else:
                #label_ids.append(label[word_idx] if label_all_tokens else -100)
                label_ids.append(label2id[label[word_idx]] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
label_all_tokens = False

tokenized_dataset = dataset_dict.map(map_labels_to_tags).map(tokenize_and_align_labels, batched=True)

bert_data = tokenized_dataset.remove_columns(['tokens', 'ner_tags', 'tags'])
bert_data.set_format("torch")

def find_transition_matrix(data):
    transition_matrix = np.zeros((len(label_list), len(label_list)))
    for example in data:
        tags = example['tags']
        for i in range(len(tags) - 1):
            transition_matrix[label2id[tags[i]], label2id[tags[i+1]]] += 1
    return transition_matrix

print("Transition matrix for training data")
transition_matrix = find_transition_matrix(tokenized_dataset['train'])
#transition_matrix = np.delete(transition_matrix, 3, 0)
#transition_matrix = np.delete(transition_matrix, 3, 1)
#xticklabels = [ner_tag_mapping[i] for i in range(len(ner_tag_mapping)) if i != 3]
#yticklabels = [ner_tag_mapping[i] for i in range(len(ner_tag_mapping)) if i != 3]
xticklabels = label_list
yticklabels = label_list

sns.heatmap(transition_matrix, annot=True, xticklabels=xticklabels, yticklabels=yticklabels)

def count_labels(tokenized_dataset_partition, padding_label=-100):
    flat_labels = [label for sublist in tokenized_dataset_partition['labels'] for label in sublist if label != padding_label]
    return Counter(flat_labels)

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
            data[label][f'{name} %'] = count / total * 100
    sorted_labels = sorted(labels)
    df_data = {label: data[label] for label in sorted_labels}
    df = pd.DataFrame(df_data).T.fillna(0)
    columns = []
    for metric in ['Abs', '%']:
        for name in datasets.keys():
            columns.append(f'{name} {metric}')
    df = df[columns]
    return df

datasets_split = {'Train': tokenized_dataset['train'], 'Val': tokenized_dataset['val'], 'Test': tokenized_dataset['test']}
df = get_label_counts_dataframe(datasets_split)
print(df)

batch_size = 16

def convert_to_data_loader(dataset, num_classes):
    input_tensor = torch.from_numpy(np.array(dataset['input_ids']))
    tags_tensor = torch.from_numpy(np.array(dataset['labels'])).long()
    token_type_tensor = torch.from_numpy(np.array(dataset['token_type_ids'])).long()
    attention_mask_tensor = torch.from_numpy(np.array(dataset['attention_mask'])).bool()
    tensor_dataset = TensorDataset(input_tensor, tags_tensor, token_type_tensor, attention_mask_tensor)
    loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)
    return loader

num_classes = len(ner_tag_mapping)

def apply_debug_dataset(bert_data):
    if not DebugConfig.ENABLED:
        return bert_data["train"], bert_data["val"], bert_data["test"]

    print("⚡ DEBUG MODE ENABLED")

    train = bert_data["train"].select(range(DebugConfig.TRAIN_SAMPLES))
    val   = bert_data["val"].select(range(DebugConfig.VAL_SAMPLES))
    test  = bert_data["test"].select(range(DebugConfig.TEST_SAMPLES))

    return train, val, test


train_ds, val_ds, test_ds = apply_debug_dataset(bert_data)

train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=DebugConfig.BATCH_SIZE if DebugConfig.ENABLED else 16)
eval_dataloader  = DataLoader(val_ds, batch_size=DebugConfig.BATCH_SIZE if DebugConfig.ENABLED else 16)
test_dataloader  = DataLoader(test_ds, batch_size=DebugConfig.BATCH_SIZE if DebugConfig.ENABLED else 16)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

train_labels = tokenized_dataset['train']['labels']
flat_train_labels = [item for sublist in train_labels for item in sublist]
padding_label = -100
non_padded_labels = [label for label in flat_train_labels if label != padding_label]
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(non_padded_labels), y=non_padded_labels)
weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
print(weight_tensor)

class MyTaskSpecificCustomModel(nn.Module):
    def __init__(self, checkpoint, num_labels, hidden_size, freeze_bert=False, use_crf=True):
        super(MyTaskSpecificCustomModel, self).__init__()
        self.num_labels = num_labels
        self.use_crf = use_crf
        self.config = AutoConfig.from_pretrained(checkpoint, output_attentions=False, output_hidden_state=False) # s added to attentions on 26 March
        self.bert = AutoModel.from_pretrained(checkpoint, config=self.config)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(0.1)
        self.hidden_layer = nn.LSTM(self.config.hidden_size, hidden_size, bidirectional=True, batch_first=True, dropout=0.1)
        self.output_layer = nn.Linear(hidden_size * 2, num_labels)
        #self.crf = CRF(num_tags=num_labels, batch_first=True)
        if self.use_crf:
            self.crf = CRF(num_tags=num_labels, batch_first=True)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None, weight_tensor=None):
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        sequence_outputs = self.dropout(last_hidden_state)
        lstm_output, _ = self.hidden_layer(sequence_outputs)
        logits = self.output_layer(lstm_output)

        loss = None
        predictions = None

        mask = attention_mask == 1

        if self.use_crf:
            if labels is not None:
                labels_fixed = labels.clone()
                labels_fixed[labels_fixed == -100] = 0
                loss = -self.crf(logits, labels_fixed, mask=mask, reduction='mean')
            else:
                predictions = self.crf.decode(logits, mask=mask)

        else:
            # standard token classification
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

            if labels is not None:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                predictions = torch.argmax(logits, dim=-1)
        


        
      
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        ), predictions 

#model = MyTaskSpecificCustomModel(checkpoint="bert-base-uncased", num_labels=9, hidden_size=64, freeze_bert=False)
model = MyTaskSpecificCustomModel(
    checkpoint="bert-base-uncased",
    num_labels=len(label_list),   
    hidden_size=64,
    freeze_bert=False,
    use_crf=USE_CRF
)
model.to(device)

bert_params = list(model.bert.parameters())
#other_params = list(model.dropout.parameters()) + list(model.hidden_layer.parameters()) + list(model.output_layer.parameters()) + list(model.crf.parameters())
other_params = (
    list(model.dropout.parameters()) +
    list(model.hidden_layer.parameters()) +
    list(model.output_layer.parameters())
)

if USE_CRF:
    other_params += list(model.crf.parameters())
    
optimizer = AdamW(
    [
        {"params": bert_params, "lr": 5e-5},
        {"params": other_params, "lr": 5e-3},
    ],
    lr=5e-3,
)

#num_epochs = 10
num_epochs = DebugConfig.EPOCHS if DebugConfig.ENABLED else 20

num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

progress_bar = tqdm(range(num_training_steps))



# --- Metrics ---
metric_acc = evaluate.load("accuracy", experiment_id=str(os.getpid()))
metric_f1  = evaluate.load("f1", experiment_id=str(os.getpid()))

progress_bar = tqdm(total=num_epochs * len(train_dataloader), desc="Training")

for epoch in range(num_epochs):
    # ----- TRAINING -----
    model.train()
    for step, batch in enumerate(train_dataloader):
        # Debug mode early stopping
        if DebugConfig.ENABLED and step >= DebugConfig.MAX_TRAIN_BATCHES:
            break

        # Move batch to device
        if isinstance(batch, dict):
            batch_device = {k: v.to(device) for k, v in batch.items()}
        else:
            batch_device = {
                "input_ids": batch[0].to(device),
                "labels": batch[1].to(device),
                "token_type_ids": batch[2].to(device),
                "attention_mask": batch[3].to(device)
            }

        outputs, _ = model(**batch_device)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        progress_bar.update(1)

    print(f"Epoch {epoch+1}/{num_epochs} training complete. Loss: {loss.item():.4f}")

    # ----- EVALUATION -----
    model.eval()
    has_data = False
    metric_acc = evaluate.load("accuracy", experiment_id=str(os.getpid()))
    metric_f1  = evaluate.load("f1", experiment_id=str(os.getpid()))

    for step, batch in enumerate(eval_dataloader):
        if DebugConfig.ENABLED and step >= DebugConfig.MAX_EVAL_BATCHES:
            break

        # Move batch to device
        if isinstance(batch, dict):
            batch_device = {k: v.to(device) for k, v in batch.items()}
        else:
            batch_device = {
                "input_ids": batch[0].to(device),
                "labels": batch[1].to(device),
                "token_type_ids": batch[2].to(device),
                "attention_mask": batch[3].to(device)
            }

        with torch.no_grad():
            # Exclude labels for CRF decoding
            batch_input = {k: v for k, v in batch_device.items() if k != "labels"}
            _, preds = model(**batch_input)

        if not USE_CRF:
           preds = preds.cpu().tolist()

        if preds is not None:
            golds = batch_device["labels"].cpu().tolist()
            masks = batch_device["attention_mask"].cpu().tolist()
            filtered_predictions = []
            filtered_references  = []

            # Align predictions with labels and attention mask
            for seq_pred, seq_gold, mask_seq in zip(preds, golds, masks):
                valid_positions = [i for i, m in enumerate(mask_seq) if m == 1]

                for pred_idx, pos in enumerate(valid_positions):
                    g = seq_gold[pos]
                    p = seq_pred[pred_idx]

                    if g != -100:
                        filtered_predictions.append(p)
                        filtered_references.append(g)

            if filtered_predictions:
                metric_acc.add_batch(predictions=filtered_predictions, references=filtered_references)
                metric_f1.add_batch(predictions=filtered_predictions, references=filtered_references)
                has_data = True

    if has_data:
        print(f"Epoch {epoch+1} - Val Acc: {metric_acc.compute()['accuracy']:.4f}")
        print(f"Epoch {epoch+1} - Val F1 : {metric_f1.compute(average='macro')['f1']:.4f}")
    else:
        print(f"Epoch {epoch+1} - Val Acc: N/A (no valid tokens)")
        print(f"Epoch {epoch+1} - Val F1 : N/A (no valid tokens)")

    # ----- Checkpoint saving -----
    if (epoch + 1) % 10 == 0:
        print(f"Saving model checkpoint for epoch {epoch+1}...")
        torch.save(model.state_dict(), f'model_checkpoint_epoch_{epoch+1}.pth')
        torch.save(optimizer.state_dict(), f'optimizer_checkpoint_epoch_{epoch+1}.pth')

progress_bar.close()


# ----- HELPER FUNCTIONS FOR TEST -----


def collect_predictions_seqeval(dataloader):
    model.eval()
    y_true, y_pred = [], []

    for batch in dataloader:
        if isinstance(batch, dict):
            batch_device = {k: v.to(device) for k, v in batch.items()}
        else:
            batch_device = {
                "input_ids": batch[0].to(device),
                "labels": batch[1].to(device),
                "token_type_ids": batch[2].to(device),
                "attention_mask": batch[3].to(device)
            }

        with torch.no_grad():
            batch_input = {k: v for k, v in batch_device.items() if k != "labels"}
            _, preds = model(**batch_input)

        if not USE_CRF:
           preds = preds.cpu().tolist()

        golds = batch_device["labels"].cpu().tolist()
        masks = batch_device["attention_mask"].cpu().tolist()

        for seq_pred, seq_gold, mask_seq in zip(preds, golds, masks):
            true_seq = []
            pred_seq = []

            valid_positions = [i for i, m in enumerate(mask_seq) if m == 1]

            for pred_idx, pos in enumerate(valid_positions):
                g = seq_gold[pos]
                p = seq_pred[pred_idx]

                if g != -100:
                    true_seq.append(id2label[g])
                    pred_seq.append(id2label[p])

            if true_seq:
                y_true.append(true_seq)
                y_pred.append(pred_seq)

    return y_true, y_pred



def collect_predictions_flat(dataloader):
    model.eval()
    y_true, y_pred = [], []

    for batch in dataloader:
        if isinstance(batch, dict):
            batch_device = {k: v.to(device) for k, v in batch.items()}
        else:
            batch_device = {
                "input_ids": batch[0].to(device),
                "labels": batch[1].to(device),
                "token_type_ids": batch[2].to(device),
                "attention_mask": batch[3].to(device)
            }

        with torch.no_grad():
            batch_input = {k: v for k, v in batch_device.items() if k != "labels"}
            _, preds = model(**batch_input)

        if not USE_CRF:
            preds = preds.cpu().tolist()

        golds = batch_device["labels"].cpu().tolist()
        masks = batch_device["attention_mask"].cpu().tolist()

        for seq_pred, seq_gold, mask_seq in zip(preds, golds, masks):
            valid_positions = [i for i, m in enumerate(mask_seq) if m == 1]

            for pred_idx, pos in enumerate(valid_positions):
                g = seq_gold[pos]
                p = seq_pred[pred_idx]

                if g != -100:
                    y_true.append(g)
                    y_pred.append(p)

    return y_true, y_pred

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


#------seqeval report------
def print_seqeval_report(dataloader, name):
    y_true, y_pred = collect_predictions_seqeval(dataloader)

    print(f"\n===== {name} SET (SEQEVAL) =====")
    print(seqeval_classification_report(y_true, y_pred))
    print("F1:", f1_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))

# ----- FINAL TEST REPORT -----
#y_true_test, y_pred_test = collect_predictions(test_dataloader)
#y_true_test, y_pred_test = collect_predictions_seqeval(test_dataloader)
y_true_test, y_pred_test = collect_predictions_flat(test_dataloader)
print_seqeval_report(test_dataloader, "TEST")
print_seqeval_report(eval_dataloader, "VALIDATION")
plot_and_save_confusion_matrices(y_true_test, y_pred_test, id2label, prefix="test")





