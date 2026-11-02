#!/usr/bin/env python3
"""Standalone inference/evaluation for ORIGINAL PURE-SUM.

Based on train_pure_sum_original_materialsbert.py. Requires its trained RE
checkpoint; this is a learned classifier, not a training-free heuristic.

Input POLYIE split format: JSON object with nested text/label/relation arrays:
  text[doc][segment] = word tokens
  label[doc][segment] = BIO tags
  relation[doc][segment] = list of relations, each a list of [start,end] spans.
Predictions: nested [doc][segment][word] or flat [segment][word] BIO tags.

pip install torch transformers tqdm
python demo_pure_sum_original_re.py --test test_split.txt \
  --checkpoint pure_sum_original_materialsbert_checkpoints/run_1/best_model.pth \
  --predictions test_predictions.json --output-dir pure_sum_demo_results
"""
import argparse
import csv
import itertools
import json
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = 'pranav-s/MaterialsBERT'
VALID = {('CN','PN','PV'), ('CN','PN','PV','Condition')}
TYPES = {'CN','PN','PV','Condition'}
MAX_WIDTH_WORDS = 100
MAX_CONTENT = 510
MARKER_START, MARKER_END = '<<m>>', '<</m>>'


def entities_from_bio(labels):
    result, start, current = [], None, None
    for i, tag in enumerate(list(labels) + ['O']):
        if tag == 'O':
            if current in TYPES:
                result.append((current, start, i))
            current, start = None, None
            continue
        if '-' not in tag:
            raise ValueError(f'Invalid BIO tag {tag!r}')
        prefix, typ = tag.split('-', 1)
        if prefix not in ('B','I'):
            raise ValueError(f'Invalid BIO tag {tag!r}')
        if prefix == 'B' or typ != current:
            if current in TYPES:
                result.append((current, start, i))
            current, start = typ, i
    return result


def relation_key(rel):
    return tuple(tuple(int(v) for v in span) for span in rel)


def load_test(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, dict) or not all(k in data for k in ('text','label','relation')):
        raise ValueError('Expected original PURE-SUM nested split format: JSON object with text, label, relation. Convert other formats first.')
    for key in ('label','relation'):
        if len(data[key]) != len(data['text']):
            raise ValueError(f'Document count mismatch for {key}')
    for doc_id, doc in enumerate(data['text']):
        for seg_id, words in enumerate(doc):
            if len(words) != len(data['label'][doc_id][seg_id]):
                raise ValueError(f'Gold BIO length mismatch at {doc_id}/{seg_id}')
    return data


def load_predictions(path, data):
    predictions = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(predictions, dict):
        predictions = predictions.get('predictions', predictions.get('labels'))
    if not isinstance(predictions, list):
        raise ValueError('Predictions must be JSON arrays of BIO tag sequences')
    # Accept either document/segment nested or flattened segment sequences.
    expected_docs = len(data['text'])
    expected_segments = sum(len(doc) for doc in data['text'])
    if len(predictions) == expected_docs and all(
        isinstance(p, list) and len(p) == len(data['text'][i])
        and (not p or isinstance(p[0], list)) for i, p in enumerate(predictions)
    ):
        nested = predictions
    elif len(predictions) == expected_segments:
        iterator = iter(predictions)
        nested = [[next(iterator) for _ in doc] for doc in data['text']]
    else:
        raise ValueError(f'Prediction shape mismatch: {len(predictions)} outer items; expected {expected_docs} documents or {expected_segments} segments')
    for d, doc in enumerate(data['text']):
        for s, words in enumerate(doc):
            if len(nested[d][s]) != len(words):
                raise ValueError(f'Predicted BIO length mismatch at document {d}, segment {s}: {len(nested[d][s])} != {len(words)}')
    return nested


def candidates_from_labels(labels):
    by_type = {typ: [] for typ in TYPES}
    for typ, start, end in entities_from_bio(labels):
        by_type[typ].append((start, end))
    for cn, pn, pv in itertools.product(by_type['CN'], by_type['PN'], by_type['PV']):
        core = (cn, pn, pv)
        for spans in (core, *(core + (condition,) for condition in by_type['Condition'])):
            if max(end for _, end in spans) - min(start for start, _ in spans) <= MAX_WIDTH_WORDS:
                yield spans


def encode_candidate(words, spans, tokenizer):
    """Faithful marker insertion and candidate-centred 510-subword crop."""
    encoding = tokenizer(words, is_split_into_words=True, add_special_tokens=False, truncation=False)
    ids = encoding['input_ids']
    word_map = {}
    for i, w in enumerate(encoding.word_ids()):
        if w is not None:
            if w not in word_map:
                word_map[w] = [i, i + 1]
            else:
                word_map[w][1] = i + 1
    try:
        subspans = [(word_map[a][0], word_map[b-1][1]) for a, b in spans]
    except (KeyError, IndexError):
        return None
    ordered = sorted(subspans)
    if any(a_end > b_start for (_, a_end), (b_start, _) in zip(ordered, ordered[1:])):
        return None
    starts = {a for a, _ in subspans}
    ends = {b for _, b in subspans}
    marked, positions = [], []
    for i in range(len(ids) + 1):
        if i in ends:
            marked.append(tokenizer.convert_tokens_to_ids(MARKER_END))
        if i in starts:
            positions.append(len(marked))
            marked.append(tokenizer.convert_tokens_to_ids(MARKER_START))
        if i < len(ids):
            marked.append(ids[i])
    if len(positions) != len(subspans):
        return None
    first = min(positions)
    last_end = max(ends)
    closing = last_end + sum(st <= last_end for st in starts) + sum(en < last_end for en in ends)
    required_end = closing + 1
    if required_end - first > MAX_CONTENT:
        return None
    if len(marked) <= MAX_CONTENT:
        left, right = 0, len(marked)
    else:
        available = MAX_CONTENT - (required_end - first)
        left = max(0, first - available // 2)
        left = min(left, len(marked) - MAX_CONTENT)
        right = left + MAX_CONTENT
        if not (left <= first and right >= required_end):
            return None
    input_ids = [tokenizer.cls_token_id] + marked[left:right] + [tokenizer.sep_token_id]
    marker_positions = [p - left + 1 for p in positions]
    if not all(input_ids[p] == tokenizer.convert_tokens_to_ids(MARKER_START) for p in marker_positions):
        return None
    return input_ids, marker_positions


class OriginalPureSum(nn.Module):
    """Original MaterialsBERT marker-average, LayerNorm, dropout, linear head."""
    def __init__(self, tokenizer):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        self.encoder.resize_token_embeddings(len(tokenizer))
        h = self.encoder.config.hidden_size
        self.layer_norm = nn.LayerNorm(h)
        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)
        self.classifier = nn.Linear(h, 1)

    def forward(self, ids, mask, positions):
        hidden = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        pooled = torch.stack([hidden[i, pos, :].mean(dim=0) for i, pos in enumerate(positions)])
        return self.classifier(self.dropout(self.layer_norm(pooled))).squeeze(-1)


def score(gold, predicted):
    tp, fp, fn = len(gold & predicted), len(predicted - gold), len(gold - predicted)
    return tp, fp, fn


def metrics(counts):
    tp, fp, fn = counts
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    return {'TP': tp, 'FP': fp, 'FN': fn, 'precision': precision,
            'recall': recall, 'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.}


def evaluate_source(name, data, labels, model, tokenizer, device, threshold, batch_size):
    rows, totals = [], [0, 0, 0]
    missing_gold = 0
    for d, document in enumerate(data['text']):
        for s, words in enumerate(document):
            gold = {relation_key(rel) for rel in data['relation'][d][s]
                    if len(rel) in (3, 4) and all(isinstance(span, (list, tuple)) and len(span) == 2 for span in rel)}
            # Evaluate only supported ordered CN-PN-PV[Condition] signatures.
            gold_labels = data['label'][d][s]
            gold = {rel for rel in gold if all(0 <= a < len(gold_labels) and b <= len(gold_labels)
                    for a, b in rel) and tuple(gold_labels[a].split('-', 1)[-1] for a, _ in rel) in VALID}
            candidate_list = list(candidates_from_labels(labels[d][s]))
            encoded = []
            for spans in candidate_list:
                item = encode_candidate(words, spans, tokenizer)
                if item is not None:
                    encoded.append((spans, *item))
            predicted = set()
            for offset in range(0, len(encoded), batch_size):
                batch = encoded[offset:offset + batch_size]
                max_len = max(len(x[1]) for x in batch)
                ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
                mask = torch.zeros_like(ids)
                for i, (_, input_ids, _) in enumerate(batch):
                    ids[i, :len(input_ids)] = torch.tensor(input_ids, device=device)
                    mask[i, :len(input_ids)] = 1
                with torch.inference_mode():
                    probs = torch.sigmoid(model(ids, mask, [x[2] for x in batch])).cpu().tolist()
                if isinstance(probs, float):
                    probs = [probs]
                for (spans, _, _), prob in zip(batch, probs):
                    if prob >= threshold:
                        predicted.add(spans)
                    rows.append({'source': name, 'doc_id': d, 'seg_id': s,
                                 'spans': json.dumps(spans),
                                 'entities': json.dumps([' '.join(words[a:b]) for a, b in spans], ensure_ascii=False),
                                 'probability': prob, 'predicted': int(prob >= threshold),
                                 'gold': int(spans in gold)})
            counts = score(gold, predicted)
            totals = [a + b for a, b in zip(totals, counts)]
            missing_gold += len(gold - set(candidate_list))
    return metrics(totals), rows, missing_gold


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--test', required=True, help='Original nested POLYIE test_split.txt')
    parser.add_argument('--checkpoint', required=True, help='Trained ORIGINAL PURE-SUM best_model.pth')
    parser.add_argument('--predictions', help='Optional NER predicted BIO JSON')
    parser.add_argument('--output-dir', default='pure_sum_demo_results')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--threshold', type=float, help='Override checkpoint validation threshold')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    data = load_test(args.test)
    if not Path(args.checkpoint).is_file():
        parser.error(f'RE checkpoint not found: {args.checkpoint}. This learned model cannot be evaluated without its trained weights.')
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tokenizer.add_special_tokens({'additional_special_tokens': [MARKER_START, MARKER_END]})
    model = OriginalPureSum(tokenizer)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint.get('architecture') != 'MaterialsBERTPureSumOriginal':
        raise ValueError(f"Checkpoint architecture {checkpoint.get('architecture')!r} is not original PURE-SUM; refusing role-aware/incompatible model")
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.to(device).eval()
    threshold = args.threshold if args.threshold is not None else checkpoint['best_threshold']
    if not 0 <= threshold <= 1:
        raise ValueError('Threshold must be between 0 and 1')
    sources = [('gold_ner', data['label'])]
    if args.predictions:
        sources.append(('predicted_ner', load_predictions(args.predictions, data)))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_rows, summary = [], []
    for name, labels in sources:
        result, rows, missing = evaluate_source(name, data, labels, model, tokenizer, device, threshold, args.batch_size)
        print(f'{name}: {result} | gold relations absent from candidates: {missing}')
        summary.append({'source': name, **result, 'gold_relations_absent_from_candidates': missing})
        all_rows.extend(rows)
    with (out / 'pure_sum_original_metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    with (out / 'pure_sum_original_candidates.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['source','doc_id','seg_id','spans','entities','probability','predicted','gold']); writer.writeheader(); writer.writerows(all_rows)
    (out / 'pure_sum_original_config.json').write_text(json.dumps({'model': MODEL_NAME, 'architecture': 'MaterialsBERTPureSumOriginal', 'checkpoint': args.checkpoint, 'threshold': threshold, 'device': device, 'max_relation_width_words': MAX_WIDTH_WORDS, 'sources': [name for name, _ in sources]}, indent=2), encoding='utf-8')
    print(f'Saved results to {out.resolve()}')


if __name__ == '__main__':
    main()
