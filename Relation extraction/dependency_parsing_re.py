#!/usr/bin/env python3
"""Standalone POLYIE dependency-based RE demonstration.

Reproduces the held-out dependency heuristic in `dependency parse.ipynb`
(cell `predict_dependency_cn_strategy`, strategy='combined', penalty=2,
pn_pv_max=3, cn_max=20). Uses gold BIO tags and, optionally, saved NER
predictions on identical test segments. No PURE-SUM imports or Jupyter needed.

Install: python -m pip install 'spacy>=3.7,<4'
         python -m spacy download en_core_web_sm
Run:     python demo_dependency_re.py --test test_split.txt \
             --predictions test_predictions.json --output-dir dependency_results

The original dependency heuristic predicts CN-PN-PV triples, not Conditions.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ENTITY_TYPES = {'CN', 'PN', 'PV', 'Condition'}
VALID_SIGNATURES = {('CN', 'PN', 'PV'), ('CN', 'PN', 'PV', 'Condition')}


def spans_from_bio(tags):
    """Follow the BIO span reconstruction in train_pure_sum_original_materialsbert.py."""
    spans, start, typ = [], None, None
    for i, tag in enumerate(list(tags) + ['O']):
        if tag == 'O':
            if typ in ENTITY_TYPES:
                spans.append((typ, start, i))
            start, typ = None, None
            continue
        if '-' not in tag:
            raise ValueError(f'Invalid BIO tag {tag!r} at token {i}')
        prefix, new_typ = tag.split('-', 1)
        if prefix not in ('B', 'I'):
            raise ValueError(f'Invalid BIO prefix {tag!r}')
        if prefix == 'B' or new_typ != typ:
            if typ in ENTITY_TYPES:
                spans.append((typ, start, i))
            start, typ = i, new_typ
    return spans


def entities_from_tags(tags):
    return [{'id': (start, end), 'label': typ,
             'start_offset': start, 'end_offset': end}
            for typ, start, end in spans_from_bio(tags)]


def entity_head(doc, start, end):
    candidates = [token for token in doc[start:end]
                  if token.head.i < start or token.head.i >= end
                  or token.head.i == token.i]
    if not candidates:
        candidates = list(doc[start:end])
    candidates = [token for token in candidates if not token.is_punct] or candidates
    return candidates[0].i


def dependency_distance(doc, index_a, index_b):
    a, b = doc[index_a], doc[index_b]
    if a.sent.start != b.sent.start:
        return None
    ancestors = {}
    current, depth = a, 0
    while True:
        ancestors[current.i] = depth
        if current.head == current:
            break
        current, depth = current.head, depth + 1
    current, depth_b = b, 0
    while True:
        if current.i in ancestors:
            return ancestors[current.i] + depth_b
        if current.head == current:
            break
        current, depth_b = current.head, depth_b + 1
    return None


def predict_dependency(doc_record, nlp, strategy='combined', penalty=2,
                       pn_pv_max=3, cn_max=20):
    """Frozen dependency heuristic from the original notebook (core triples)."""
    from spacy.tokens import Doc
    tokens, entities = doc_record['text'], doc_record['entities']
    parsed = Doc(nlp.vocab, words=tokens)
    for _, component in nlp.pipeline:
        parsed = component(parsed)
    by_type = {typ: [e for e in entities if e['label'] == typ]
               for typ in ('CN', 'PN', 'PV')}
    heads = {e['id']: entity_head(parsed, e['start_offset'], e['end_offset'])
             for e in entities}

    def dep(a, b):
        return dependency_distance(parsed, heads[a['id']], heads[b['id']])

    def linear(a, b):
        return max(0, a['start_offset'] - b['end_offset'],
                   b['start_offset'] - a['end_offset'])

    predicted = set()
    for pv in by_type['PV']:
        pn_candidates = []
        for pn in by_type['PN']:
            d = dep(pn, pv)
            if d is not None and d <= pn_pv_max:
                pn_candidates.append((d, linear(pn, pv), pn))
        if not pn_candidates:
            continue
        pn = min(pn_candidates, key=lambda x: (x[0], x[1]))[2]
        expression_end = max(pn['end_offset'], pv['end_offset'])
        cn_candidates = []
        for cn in by_type['CN']:
            d_cn_pn, d_cn_pv = dep(cn, pn), dep(cn, pv)
            if d_cn_pn is None or d_cn_pv is None:
                continue
            combined = d_cn_pn + d_cn_pv
            if combined > cn_max:
                continue
            after = int(cn['start_offset'] >= expression_end)
            if strategy == 'combined':
                score = combined
            elif strategy == 'pn_focused':
                score = d_cn_pn
            elif strategy == 'pv_focused':
                score = d_cn_pv
            else:
                raise ValueError(f'Unknown strategy: {strategy}')
            score += penalty * after
            cn_candidates.append((score, linear(cn, pn) + linear(cn, pv), cn))
        if not cn_candidates:
            continue
        cn = min(cn_candidates, key=lambda x: (x[0], x[1]))[2]
        predicted.add((tuple(cn['id']), tuple(pn['id']), tuple(pv['id'])))
    return predicted


def load_test(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not all(k in data for k in ('text', 'label', 'relation')):
        raise ValueError('Expected JSON keys: text, label, relation')
    docs, raw, excluded = [], 0, 0
    if not (len(data['text']) == len(data['label']) == len(data['relation'])):
        raise ValueError('Document-level text/label/relation counts differ')
    for doc_id, (segments, labels, relations) in enumerate(zip(
            data['text'], data['label'], data['relation'])):
        if not (len(segments) == len(labels) == len(relations)):
            raise ValueError(f'Segment counts differ in document {doc_id}')
        for seg_id, (tokens, tags, gold_relations) in enumerate(zip(
                segments, labels, relations)):
            if len(tokens) != len(tags):
                raise ValueError(f'Gold BIO alignment mismatch: {doc_id}/{seg_id}')
            entities = entities_from_tags(tags)
            span_to_type = {e['id']: e['label'] for e in entities}
            gold = set()
            for rel in gold_relations:
                raw += 1
                key = tuple(tuple(span) for span in rel)
                signature = tuple(span_to_type.get(span, 'UNKNOWN') for span in key)
                if signature in VALID_SIGNATURES:
                    gold.add(key)
                else:
                    excluded += 1
            docs.append({'doc_id': doc_id, 'seg_id': seg_id,
                         'text': tokens, 'entities': entities, 'relations': gold})
    return docs, raw, excluded


def evaluate(docs, nlp, source, example_rows):
    counts = Counter()
    for doc in docs:
        predicted = predict_dependency(doc, nlp)
        # Match the original notebook: exact full-relation matching. The
        # optional Condition in a gold quadruple is not silently discarded.
        gold = doc['relations']
        counts['TP'] += len(predicted & gold)
        counts['FP'] += len(predicted - gold)
        counts['FN'] += len(gold - predicted)
        for rel in sorted(predicted):
            cn, pn, pv = rel
            tokens = doc['text']
            example_rows.append({
                'entity_source': source, 'doc_id': doc['doc_id'],
                'seg_id': doc['seg_id'],
                'chemical_name': ' '.join(tokens[cn[0]:cn[1]]),
                'property_name': ' '.join(tokens[pn[0]:pn[1]]),
                'property_value': ' '.join(tokens[pv[0]:pv[1]]),
                'cn_span': str(cn), 'pn_span': str(pn), 'pv_span': str(pv),
                'exact_gold_match': rel in gold,
            })
    tp, fp, fn = (counts[k] for k in ('TP', 'FP', 'FN'))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {'entity_source': source, 'TP': tp, 'FP': fp, 'FN': fn,
            'precision': p, 'recall': r, 'f1': f1}


def write_csv(path, rows, fields):
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--test', required=True, help='POLYIE held-out test_split.txt')
    ap.add_argument('--predictions', help='Optional NER test_predictions.json')
    ap.add_argument('--output-dir', default='dependency_results')
    ap.add_argument('--spacy-model', default='en_core_web_sm')
    args = ap.parse_args()
    try:
        import spacy
    except ImportError as exc:
        raise SystemExit('Install spaCy: python -m pip install "spacy>=3.7,<4"') from exc
    try:
        nlp = spacy.load(args.spacy_model)
    except OSError as exc:
        raise SystemExit(f'Missing spaCy model {args.spacy_model!r}; run: '
                         f'python -m spacy download {args.spacy_model}') from exc
    if not nlp.has_pipe('parser'):
        raise SystemExit('The spaCy model must include a dependency parser.')
    gold_docs, raw, excluded = load_test(args.test)
    print(f'Test segments: {len(gold_docs)}; raw gold relations: {raw}; '
          f'excluded invalid-signature relations: {excluded}')
    if excluded:
        print('NOTE: Excluded relations are not evaluated, as in the original notebook.')
    datasets = [('Gold NER', gold_docs)]
    if args.predictions:
        predictions = json.loads(Path(args.predictions).read_text(encoding='utf-8'))
        if not isinstance(predictions, list) or len(predictions) != len(gold_docs):
            raise ValueError('Predictions must be one BIO-label list per test segment '
                             f'({len(gold_docs)} expected).')
        predicted_docs = []
        for i, (doc, tags) in enumerate(zip(gold_docs, predictions)):
            if not isinstance(tags, list) or len(tags) != len(doc['text']):
                raise ValueError(f'Prediction/token alignment mismatch in segment {i}')
            predicted_docs.append({**doc, 'entities': entities_from_tags(tags)})
        datasets.append(('Predicted NER', predicted_docs))
    metrics, examples = [], []
    for source, docs in datasets:
        row = evaluate(docs, nlp, source, examples)
        metrics.append(row)
        print(f"{source}: TP={row['TP']} FP={row['FP']} FN={row['FN']} "
              f"P={row['precision']:.4f} R={row['recall']:.4f} F1={row['f1']:.4f}")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'dependency_gold_vs_predicted_metrics.csv', metrics,
              ['entity_source', 'TP', 'FP', 'FN', 'precision', 'recall', 'f1'])
    write_csv(out / 'dependency_extracted_relations.csv', examples,
              ['entity_source', 'doc_id', 'seg_id', 'chemical_name',
               'property_name', 'property_value', 'cn_span', 'pn_span',
               'pv_span', 'exact_gold_match'])
    config = {'test': args.test, 'predictions': args.predictions,
              'spacy_model': args.spacy_model, 'strategy': 'combined',
              'penalty': 2, 'pn_pv_max': 3, 'cn_max': 20,
              'conditions_predicted': False, 'gold_relations_raw': raw,
              'gold_relations_excluded': excluded,
              'evaluation': 'exact full-relation span tuples'}
    (out / 'dependency_run_config.json').write_text(
        json.dumps(config, indent=2), encoding='utf-8')
    print(f'Results saved to: {out.resolve()}')


if __name__ == '__main__':
    main()
