#!/usr/bin/env python3
"""Standalone V3 proximity RE demonstration.

Uses the original V3 proximity algorithm, with gold and/or predicted BIO NER
labels from the held-out POLYIE split.

Example:
  python proximity_re.py --test test_split.txt \
      --predictions checkpoints_windowed_full/test_predictions.json \
      --output-dir v3_demo_results

Gold-only:
  python proximity_re.py --test test_split.txt --output-dir v3_demo_results

Gold relation annotations must contain ordered CN, PN, PV spans and optionally
Condition spans, represented as [start, end] word offsets (end exclusive).
"""
import argparse
import csv
import json
from pathlib import Path

MAX_PN_PV_DISTANCE = 30
MAX_CN_DISTANCE = 80
MAX_CONDITION_DISTANCE = 20


def distance(a, b):
    """Number of tokens between two entity spans (original V3)."""
    if a["end_offset"] <= b["start_offset"]:
        return b["start_offset"] - a["end_offset"]
    if b["end_offset"] <= a["start_offset"]:
        return a["start_offset"] - b["end_offset"]
    return 0


def sentence_ids(tokens):
    """Original V3 punctuation-based approximate sentence segmentation."""
    result = []
    sid = 0
    for token in tokens:
        result.append(sid)
        if token in {".", "!", "?"} or token.endswith((".", "!", "?")):
            sid += 1
    return result


def predict_relations(doc, use_conditions=False):
    """Original V3: nearest PN per PV; prefer CN preceding PN/PV pair.

    Returns ordered entity-ID tuples (CN, PN, PV[, Condition]).
    """
    entities = doc["entities"]
    tokens = doc["text"]
    sids = sentence_ids(tokens)

    def sid(entity):
        return sids[entity["start_offset"]]

    def candidates(label, anchor, max_distance):
        return [
            e for e in entities
            if e["label"] == label
            and sid(e) == sid(anchor)
            and distance(e, anchor) <= max_distance
        ]

    predicted = set()
    for pv in entities:
        if pv["label"] != "PV":
            continue
        pn_candidates = candidates("PN", pv, MAX_PN_PV_DISTANCE)
        if not pn_candidates:
            continue
        pn = min(pn_candidates, key=lambda e: (
            distance(e, pv), abs(e["start_offset"] - pv["start_offset"])
        ))
        cn_candidates = [
            e for e in entities
            if e["label"] == "CN"
            and sid(e) == sid(pn)
            and min(distance(e, pn), distance(e, pv)) <= MAX_CN_DISTANCE
        ]
        if not cn_candidates:
            continue
        pair_start = min(pn["start_offset"], pv["start_offset"])
        preceding_cn = [e for e in cn_candidates if e["end_offset"] <= pair_start]
        if preceding_cn:
            cn = min(preceding_cn, key=lambda e: (
                pair_start - e["end_offset"], -e["start_offset"]
            ))
        else:
            cn = min(cn_candidates, key=lambda e: (
                min(distance(e, pn), distance(e, pv)), distance(e, pn)
            ))
        relation = (cn["id"], pn["id"], pv["id"])
        if use_conditions:
            condition_candidates = [
                e for e in entities
                if e["label"] == "Condition"
                and sid(e) == sid(pv)
                and distance(e, pv) <= MAX_CONDITION_DISTANCE
            ]
            if condition_candidates:
                condition = min(condition_candidates, key=lambda e: distance(e, pv))
                relation += (condition["id"],)
        predicted.add(relation)
    return predicted


def extract_bio_spans(tags):
    """Convert BIO tags to (type, start, end) word spans (end exclusive)."""
    spans = []
    start = None
    current_type = None
    for i, tag in enumerate(list(tags) + ["O"]):
        if tag == "ES" or tag.endswith("-ES"):
            tag = "O"  # same ES handling as the NER training loader
        if tag == "O":
            if current_type is not None:
                spans.append((current_type, start, i))
            start, current_type = None, None
            continue
        if "-" not in tag:
            raise ValueError(f"Invalid BIO tag {tag!r} at word {i}")
        prefix, entity_type = tag.split("-", 1)
        if prefix not in ("B", "I"):
            raise ValueError(f"Invalid BIO prefix {prefix!r} at word {i}")
        if prefix == "B" or entity_type != current_type:
            if current_type is not None:
                spans.append((current_type, start, i))
            start, current_type = i, entity_type
    return spans


def entities_to_doc(words, spans):
    return {"text": words, "entities": [
        {"id": i, "label": label, "start_offset": start, "end_offset": end}
        for i, (label, start, end) in enumerate(spans)
    ]}


def predicted_span_relations(words, spans, use_conditions):
    doc = entities_to_doc(words, spans)
    predicted_ids = predict_relations(doc, use_conditions)
    id_to_span = {
        e["id"]: (e["start_offset"], e["end_offset"])
        for e in doc["entities"]
    }
    return {
        tuple(id_to_span[entity_id] for entity_id in relation)
        for relation in predicted_ids
    }


def gold_span_relations(relations, tags, full):
    result = set()
    for relation in relations:
        if not isinstance(relation, (list, tuple)) or len(relation) not in (3, 4):
            continue
        chosen = relation if full else relation[:3]
        if not all(
            isinstance(span, (list, tuple)) and len(span) == 2
            and all(isinstance(v, int) for v in span)
            and 0 <= span[0] < span[1] <= len(tags)
            for span in chosen
        ):
            continue
        expected = ("CN", "PN", "PV", "Condition") if len(chosen) == 4 else ("CN", "PN", "PV")
        actual = tuple(tags[start].split("-", 1)[-1] for start, end in chosen)
        if actual == expected:
            result.add(tuple(tuple(span) for span in chosen))
    return result


def load_test(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("text", "label", "relation"):
        if key not in data:
            raise ValueError(f"{path}: missing required key {key!r}")
    if not (len(data["text"]) == len(data["label"]) == len(data["relation"])):
        raise ValueError("Document counts differ among text, label and relation")
    segments = []
    for doc_idx, (doc_words, doc_labels, doc_relations) in enumerate(zip(
        data["text"], data["label"], data["relation"]
    )):
        if not (len(doc_words) == len(doc_labels) == len(doc_relations)):
            raise ValueError(f"Document {doc_idx}: segment counts do not match")
        for seg_idx, (words, tags, relations) in enumerate(zip(doc_words, doc_labels, doc_relations)):
            if len(words) != len(tags):
                raise ValueError(f"Document {doc_idx}, segment {seg_idx}: words/tags length mismatch")
            tags = ["O" if t == "ES" or t.endswith("-ES") else t for t in tags]
            segments.append({"doc": doc_idx, "segment": seg_idx,
                             "words": words, "tags": tags, "relations": relations})
    return segments


def evaluate(segments, ner_tags, full, label):
    tp = fp = fn = 0
    examples = []
    for item, tags in zip(segments, ner_tags):
        words = item["words"]
        pred = predicted_span_relations(words, extract_bio_spans(tags), full)
        gold = gold_span_relations(item["relations"], item["tags"], full)
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
        for relation in sorted(pred):
            examples.append({
                "ner_source": label,
                "evaluation": "full" if full else "core",
                "document": item["doc"], "segment": item["segment"],
                "chemical": " ".join(words[relation[0][0]:relation[0][1]]),
                "property": " ".join(words[relation[1][0]:relation[1][1]]),
                "value": " ".join(words[relation[2][0]:relation[2][1]]),
                "condition": " ".join(words[relation[3][0]:relation[3][1]]) if len(relation) == 4 else "",
                "word_offsets": json.dumps(relation),
                "exact_gold_match": relation in gold,
            })
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2*p*r / (p+r) if p+r else 0.0
    return {"NER input": label, "RE evaluation": "full" if full else "core",
            "TP": tp, "FP": fp, "FN": fn,
            "Precision": p, "Recall": r, "F1": f1}, examples


def save_csv(path, rows, fieldnames):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", default="test_split.txt", help="POLYIE held-out test JSON split")
    parser.add_argument("--predictions", default=None, help="NER test_predictions.json (optional)")
    parser.add_argument("--output-dir", default="v3_demo_results")
    args = parser.parse_args()

    segments = load_test(args.test)
    sources = [("Gold NER", [s["tags"] for s in segments])]
    if args.predictions:
        predicted_tags = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
        if len(predicted_tags) != len(segments):
            raise ValueError(f"Predicted sequence count {len(predicted_tags)} != test segments {len(segments)}")
        for i, (tags, seg) in enumerate(zip(predicted_tags, segments)):
            if len(tags) != len(seg["words"]):
                raise ValueError(f"Prediction segment {i}: {len(tags)} labels != {len(seg['words'])} words")
        sources.append(("Predicted NER", predicted_tags))

    rows, examples = [], []
    for full in (False, True):
        for name, tags in sources:
            metrics, extracted = evaluate(segments, tags, full, name)
            rows.append(metrics)
            examples.extend(extracted)
            print(f"{name:14s} | {'Full' if full else 'Core':4s} | "
                  f"TP={metrics['TP']:4d} FP={metrics['FP']:4d} FN={metrics['FN']:4d} | "
                  f"P={metrics['Precision']:.4f} R={metrics['Recall']:.4f} F1={metrics['F1']:.4f}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_csv(out / "v3_gold_vs_predicted_metrics.csv", rows,
             ["NER input", "RE evaluation", "TP", "FP", "FN", "Precision", "Recall", "F1"])
    save_csv(out / "v3_extracted_relations.csv", examples,
             ["ner_source", "evaluation", "document", "segment", "chemical", "property",
              "value", "condition", "word_offsets", "exact_gold_match"])
    (out / "v3_run_config.json").write_text(json.dumps({
        "test": str(args.test), "predictions": str(args.predictions) if args.predictions else None,
        "segments": len(segments), "MAX_PN_PV_DISTANCE": MAX_PN_PV_DISTANCE,
        "MAX_CN_DISTANCE": MAX_CN_DISTANCE, "MAX_CONDITION_DISTANCE": MAX_CONDITION_DISTANCE,
        "core": "CN-PN-PV; Condition ignored", "full": "CN-PN-PV with optional Condition",
        "evaluation": "exact ordered word-offset tuples; gold RE annotations unchanged"
    }, indent=2), encoding="utf-8")
    print(f"\nProcessed {len(segments)} test segments. Outputs: {out.resolve()}")


if __name__ == "__main__":
    main()
