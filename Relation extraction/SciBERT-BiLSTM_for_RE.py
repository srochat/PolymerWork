

# ================= DEBUG CONFIG =================
class DebugConfig:
    ENABLED = False          # 🔁 toggle ON/OFF here
    TRAIN_SAMPLES = 20      # small dataset size
    VAL_SAMPLES = 10
    TEST_SAMPLES = 10
    BATCH_SIZE = 2
    MAX_TRAIN_BATCHES = 2   # stop early per epoch
    MAX_EVAL_BATCHES = 1
    EPOCHS = 1
if DebugConfig.ENABLED:
    print("Running quick sanity check (NOT full training)")
# ===============================================

# --- Imports and Initial Setup ---
import os
import json
import numpy as np
import pandas as pd
import torch
import re
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

from transformers.modeling_outputs import TokenClassifierOutput
from torch.optim import AdamW

import datasets
import evaluate
from tqdm.auto import tqdm
import uuid

first_run = True

if first_run:
    os.system('pip install datasets evaluate')


# ----------------- GPU CHECK -----------------
print("CUDA available:", torch.cuda.is_available())
print("Default device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
if torch.cuda.is_available():
    print("Current GPU index:", torch.cuda.current_device())
    print("GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))
# ---------------------------------------------


def load_json(file):
    with open(file, 'r') as f:
        lines = f.read().split('\n')
    return [json.loads(line) for line in lines if line]

def transform_to_dataset_dict(data):
    return {"tokens": [x["words"] for x in data], "ner_tags": [x["ner"] for x in data]}

path = '../../Data/PolymerAbstracts'
train_data = load_json(f'{path}/train.json')
test_data = load_json(f'{path}/test.json')
val_data = load_json(f'{path}/dev.json')

if DebugConfig.ENABLED:
    train_data = train_data[:DebugConfig.TRAIN_SAMPLES]
    val_data = val_data[:DebugConfig.VAL_SAMPLES]
    test_data = test_data[:DebugConfig.TEST_SAMPLES]

dataset_dict = datasets.DatasetDict({
    "train": datasets.Dataset.from_dict(transform_to_dataset_dict(train_data)),
    "test": datasets.Dataset.from_dict(transform_to_dataset_dict(test_data)),
    "val": datasets.Dataset.from_dict(transform_to_dataset_dict(val_data))
})


def clean_material(x):
    if x is None:
        return []

    if isinstance(x, str):
        x = x.strip()

        if x.startswith("{") and x.endswith("}"):
            x = x[1:-1]

        parts = [p.strip().strip("'") for p in x.split(",")]
        return [p for p in parts if p]

    return x


# --- Label Mapping ---
ner_tag_mapping = {
    0: 'INORGANIC', 1: 'MATERIAL_AMOUNT', 2: 'MONOMER', 3: 'O',
    4: 'ORGANIC', 5: 'POLYMER', 6: 'POLYMER_FAMILY', 7: 'PROP_NAME', 8: 'PROP_VALUE'
}
ner_tag_mapping_flipped = {v: k for k, v in ner_tag_mapping.items()}

label2id = {v: k for k, v in ner_tag_mapping.items()}
id2label = ner_tag_mapping

def map_labels_to_ids(example):
    example["tags"] = [label2id[tag] for tag in example["ner_tags"]]
    return example

dataset_dict = dataset_dict.map(map_labels_to_ids)


print(dataset_dict["train"][0]["tags"][:20])
print(type(dataset_dict["train"][0]["tags"][0]))


# --- Tokenization ---
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
label_all_tokens = False

def tokenize_and_align_labels(examples):

    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding=True,
        max_length=512,
        is_split_into_words=True
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
                label_ids.append(-100)  # important for subwords

            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = dataset_dict.map(
    tokenize_and_align_labels,
    batched=True
)



# --- Transition Matrix ---
def find_transition_matrix(data):
    matrix = np.zeros((len(ner_tag_mapping), len(ner_tag_mapping)))
    for example in data:
        for i in range(len(example['tags']) - 1):
            matrix[example['tags'][i], example['tags'][i + 1]] += 1
    return matrix

matrix = find_transition_matrix(dataset_dict['train'])
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
bert_data.set_format(
    type="torch",
    columns=["input_ids", "token_type_ids", "attention_mask", "labels"]
)

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

model = MyTaskSpecificCustomModel("allenai/scibert_scivocab_uncased", num_labels=9, hidden_size=64).to(device)


# --- Optimizer and Scheduler ---
bert_params = list(model.bert.parameters())
other_params = list(model.dropout.parameters()) + list(model.hidden_layer.parameters()) + list(model.output_layer.parameters())

optimizer = AdamW([
    {"params": bert_params, "lr": 5e-5},
    {"params": other_params, "lr": 5e-3},
])

num_epochs = 20
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


def predictions_to_spans(pred_sequences, tokens_list):

    docs = []

    for pred_seq, tokens in zip(pred_sequences, tokens_list):

        spans = []

        current_label = None
        start = None

        for i, label_id in enumerate(pred_seq):

            if label_id == -100:
                continue

            label_name = ner_tag_mapping[label_id]

            if label_name != "O":

                if current_label is None:
                    current_label = label_id
                    start = i

                elif current_label != label_id or (i - start) > 15:

                    spans.append([current_label, start, i - 1])
                    current_label = label_id
                    start = i

            else:

                if current_label is not None:
                    spans.append([current_label, start, i - 1])
                    current_label = None

        if current_label is not None:
            spans.append([current_label, start, len(pred_seq) - 1])

        docs.append(spans)

    return docs


def build_docs_dict(docs):
    docs_dict = {}

    for i, doc in enumerate(docs):
        entity_dict = {
            'INORGANIC': [],
            'MATERIAL_AMOUNT': [],
            'MONOMER': [],
            'ORGANIC': [],
            'POLYMER': [],
            'POLYMER_FAMILY': [],
            'PROP_NAME': [],
            'PROP_VALUE': []
        }

        for span in doc:
            entity_dict[ner_tag_mapping[span[0]]].append(span)

        docs_dict[i] = entity_dict

    return docs_dict


print(tokenized_dataset["train"].column_names)


def subtoken_to_word_predictions(pred_seq, word_ids):

    word_predictions = []
    seen_words = set()

    for pred, word_id in zip(pred_seq, word_ids):

        if word_id is None:
            continue

        if word_id == -1:
            continue

        if word_id not in seen_words:
            word_predictions.append(pred)
            seen_words.add(word_id)

    return word_predictions


# Get predictions aligned per sequence
pred_sequences = []
gold_sequences = []

for batch in eval_dataloader:

    batch = {k: v.to(device) for k, v in batch.items() if k != "tokens"}

    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    preds = torch.argmax(logits, dim=-1).cpu().numpy()
    labels = batch["labels"].cpu().numpy()

    tokens_batch = dataset_dict["val"]["tokens"]  # aligned externally

    for i, (p_seq, l_seq) in enumerate(zip(preds, labels)):

        tokens = tokens_batch[i]

        # reconstruct word_ids via tokenizer again
        encoding = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=512
        )

        word_ids = encoding.word_ids()

        word_preds = subtoken_to_word_predictions(p_seq, word_ids)
        word_gold = subtoken_to_word_predictions(l_seq, word_ids)

        pred_sequences.append(word_preds)
        gold_sequences.append(word_gold)

# Convert predictions to spans
tokens_list = dataset_dict["val"]["tokens"]
docs = predictions_to_spans(pred_sequences, tokens_list)

# Build docs_dict
docs_dict = build_docs_dict(docs)


def sentence_token_spans(tokens):

    spans = []

    start = 0

    for i, tok in enumerate(tokens):

        if re.match(r'[.!?]', tok):
            spans.append((start, i))
            start = i + 1

    if start < len(tokens):
        spans.append((start, len(tokens) - 1))

    return spans


def is_valid_entity(span, tokens):

    start, end = span[1], span[2]

    if start is None or end is None:
        return False

    if start < 0 or end >= len(tokens):
        return False

    text = " ".join(tokens[start:end+1]).strip()

    # empty
    if len(text) == 0:
        return False

    # junk only
    bad_exact = {
        "{", "}", ".", ",", "-", "(", ")", "/", "=",
        "was",
        "approximately",
        "respectively",
        "appeared",
        "even"
    }

    if text.lower() in bad_exact:
        return False

    # reject punctuation
    if not any(c.isalnum() for c in text):
        return False

    # reject extremely long spans
    if len(text.split()) > 12:
        return False

    return True

def safe_text(tokens, span):

    start, end = span[1], span[2]

    if start is None or end is None:
        return None

    if start < 0 or end >= len(tokens):
        return None

    text = " ".join(tokens[start:end+1]).strip()

    # remove junk
    if len(text) < 2:
        return None

    # remove punctuation
    bad_texts = {
        "", ".", ",", ";", ":", "(", ")",
        "{", "}", "[", "]",
        "°", "-", "/", "=",
        "and", "or", "with",
        "respectively",
        "of", "to", "the"
    }

    if text.lower() in bad_texts:
        return None

    # too many punctuation chars
    punct_ratio = sum(
        1 for c in text if not c.isalnum()
    ) / max(len(text), 1)

    if punct_ratio > 0.7:
        return None

    return text

def clean_entities(entities, tokens):

    cleaned = []

    for ent in entities:

        start, end = ent[1], ent[2]

        if start is None or end is None:
            continue

        if start < 0 or end >= len(tokens):
            continue

        text = " ".join(tokens[start:end+1])

        if text.strip() in {"", "}", "{", ","}:
            continue

        cleaned.append(ent)

    return cleaned

def link_prop_name_value(doc):
    relations = []

    prop_names = doc['PROP_NAME']
    prop_values = doc['PROP_VALUE']

    for pv in prop_values:
        best_name = None
        best_dist = float('inf')

        for pn in prop_names:
            dist = abs(pv[1] - pn[1])

            if dist < best_dist:
                best_dist = dist
                best_name = pn

        if best_name:
            relations.append({
                "prop_value": pv,
                "prop_name": best_name
            })

    return relations

def link_entities(doc, tokens):

    relations = []

    entities = [
        e for e in (
            doc['POLYMER']
            + doc['POLYMER_FAMILY']
            + doc['MONOMER']
            + doc['ORGANIC']
            + doc['INORGANIC']
        )
        if is_valid_entity(e, tokens)
    ]

    prop_values = doc['PROP_VALUE']

    for pv in prop_values:

        best_entity = None
        best_score = float("inf")

        for ent in entities:

            # distance between entity and property value
            dist = abs(pv[1] - ent[1])

            # allow larger window
            if dist > 80:
                continue

            ent_text = safe_text(tokens, ent)

            if ent_text is None:
                continue

            # penalize junk-like entities
            penalty = 0

            if len(ent_text.split()) > 6:
                penalty += 20

            if ent_text.lower() in {
                "approximately",
                "respectively",
                "appeared",
                "worth noting"
            }:
                penalty += 100

            score = dist + penalty

            if score < best_score:
                best_score = score
                best_entity = ent

        if best_entity is not None:

            relations.append({
                "prop_value": pv,
                "entity": best_entity
            })

    return relations


def extract_relations_simple(docs_dict):

    relations_output = {}

    for i, doc in docs_dict.items():

        pv_name = link_prop_name_value(doc)

        pv_entity = link_entities(doc, tokens_list[i])

        relations_output[i] = {
            "name_links": pv_name,
            "entity_links": pv_entity
        }

    return relations_output

def create_table(docs_dict, relations_output, tokens_list):
    table = []

    for doc_id, doc in docs_dict.items():

        name_links = relations_output[doc_id]["name_links"]
        entity_links = relations_output[doc_id]["entity_links"]

        for nl in name_links:
            pv = nl["prop_value"]
            pn = nl["prop_name"]

            # find matching entity
            entity_texts = []

            for el in entity_links:
                if el["prop_value"] == pv:
                    ent = el["entity"]
                    ent_text = safe_text(tokens_list[doc_id], ent)

                    if ent_text is not None:
                        entity_texts.append(ent_text)

            pv_text = safe_text(tokens_list[doc_id], pv)
            pn_text = safe_text(tokens_list[doc_id], pn)

            if pv_text is None or pn_text is None:
               continue

            table.append([
                doc_id,
                pv_text,
                pn_text,
                set(entity_texts) if entity_texts else None
            ])

    return pd.DataFrame(
        table,
        columns=['Document', 'Property Value', 'Property Name', 'Material']
    )


relations_output = extract_relations_simple(docs_dict)

df_table = create_table(docs_dict, relations_output, tokens_list)

df_table = df_table[
    df_table["Property Value"].notna()
]

df_table = df_table[
    df_table["Property Name"].notna()
]

df_table = df_table[
    df_table["Property Name"].str.len() > 2
]

df_table = df_table[
    ~df_table["Property Name"].str.fullmatch(
        r"[^\w]+",
        na=False
    )
]

df_table["Material"] = df_table["Material"].apply(clean_material)

print(df_table.head())


df_table


df_table.to_csv("relation_table.csv", index=False)

