
#!/usr/bin/env python
"""Standalone three-shot GPT relation extraction for POLYIE."""

import argparse
import hashlib
import json
import os
from pathlib import Path

RE_SHOTS = [(0, 2), (1, 4), (50, 6)]
RE_DEMO_HASH_EXPECTED = (
    "7093a0747683f0bf0cb5aa2e80d9f05eb3b57976524"
    "eeeaf6e870e2b390ee5ad"
)
ENTITY_TYPES = {"CN", "PN", "PV", "Condition"}

RE_SYSTEM_PROMPT = """
You are an expert information extraction system for polymer science literature.

Your task is relation extraction (RE) using the POLYIE annotation scheme.

You will receive:
1. Scientific text represented as numbered tokens.
2. A list of previously identified entities, each with a unique ID,
   entity type, token span and text.

Identify relationships connecting:
- CN: Chemical Name
- PN: Property Name
- PV: Property Value
- Condition: Experimental or measurement condition (optional)

Each relation must follow one of these structures:

[CN_ID, PN_ID, PV_ID]

or

[CN_ID, PN_ID, PV_ID, Condition_ID]

Rules:
1. Use only entity IDs provided in the input.
2. Do not invent entities or modify their boundaries.
3. Preserve the component order CN, PN, PV, Condition.
4. Extract relationships supported by the scientific text.
5. A relation must associate a chemical with its property and
   the corresponding property value.
6. Include a Condition only when it belongs to that relationship.
7. Multiple relations may share the same chemical or property.
8. Do not infer unsupported relationships.
9. Follow the annotation conventions illustrated by the examples.
10. Return valid JSON only, using:
    {"relations": [["E0", "E1", "E2"], ...]}
11. Return {"relations": []} if no supported relations exist.
""".strip()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bio_to_spans(labels):
    """Original notebook BIO-to-span convention."""
    spans = []
    start = None
    current_type = None

    def close(end):
        if start is not None:
            spans.append([start, end, current_type])

    for i, tag in enumerate(labels):
        if tag == "O":
            close(i)
            start = None
            current_type = None
            continue

        prefix, entity_type = tag.split("-", 1)
        if prefix not in {"B", "I"}:
            raise ValueError(f"Invalid BIO prefix: {tag}")
        if entity_type not in ENTITY_TYPES:
            raise ValueError(f"Unknown entity type: {tag}")

        if prefix == "B" or current_type != entity_type or start is None:
            close(i)
            start = i
            current_type = entity_type

    close(len(labels))
    return spans


def numbered_tokens(tokens):
    return "\n".join(
        f"{i}: {token}" for i, token in enumerate(tokens)
    )


def make_entity_inventory(tokens, entities):
    inventory = []
    id_to_span = {}

    for i, (start, end, entity_type) in enumerate(entities):
        entity_id = f"E{i}"
        inventory.append({
            "id": entity_id,
            "type": entity_type,
            "start": start,
            "end": end,
            "text": " ".join(tokens[start:end]),
        })
        id_to_span[entity_id] = (start, end, entity_type)

    return inventory, id_to_span


def build_re_demonstration(train, d, s):
    tokens = train["text"][d][s]
    labels = train["label"][d][s]
    gold_relations = train["relation"][d][s]

    entities = bio_to_spans(labels)
    inventory, id_to_span = make_entity_inventory(tokens, entities)

    span_to_id = {
        (start, end): entity_id
        for entity_id, (start, end, _) in id_to_span.items()
    }

    converted_relations = []
    for relation in gold_relations:
        converted = []
        for span in relation:
            key = tuple(span)
            if key not in span_to_id:
                raise ValueError(
                    f"Missing entity in training example ({d}, {s}): {key}"
                )
            converted.append(span_to_id[key])
        converted_relations.append(converted)

    return {
        "document": d,
        "segment": s,
        "tokens": tokens,
        "entities": inventory,
        "answer": {"relations": converted_relations},
    }


def prepare_demonstrations(train):
    demos = [
        build_re_demonstration(train, d, s)
        for d, s in RE_SHOTS
    ]

    digest = hashlib.sha256(
        json.dumps(
            demos, ensure_ascii=False, sort_keys=True
        ).encode("utf-8")
    ).hexdigest()

    if digest != RE_DEMO_HASH_EXPECTED:
        raise ValueError(
            "Three-shot demonstration hash mismatch.\n"
            f"Expected: {RE_DEMO_HASH_EXPECTED}\n"
            f"Obtained: {digest}\n"
            "Check that the original training split is being used."
        )

    return demos, digest


def build_re_prompt(tokens, entities, demonstrations):
    inventory, _ = make_entity_inventory(tokens, entities)
    sections = []

    for i, demo in enumerate(demonstrations, 1):
        sections.append(
            f"TRAINING EXAMPLE {i}\n\n"
            f"TOKENS:\n"
            f"{numbered_tokens(demo['tokens'])}\n\n"
            f"ENTITIES:\n"
            f"{json.dumps(demo['entities'], ensure_ascii=False)}\n\n"
            f"ANSWER:\n"
            f"{json.dumps(demo['answer'], ensure_ascii=False)}"
        )

    sections.append(
        "NOW EXTRACT RELATIONS FROM THE FOLLOWING TEXT.\n\n"
        f"TOKENS:\n{numbered_tokens(tokens)}\n\n"
        f"ENTITIES:\n"
        f"{json.dumps(inventory, ensure_ascii=False)}\n\n"
        "ANSWER:"
    )

    return "\n\n".join(sections)


def parse_gpt_re_response(raw_response, tokens, entities):
    """Strict original notebook output validation."""
    data = json.loads(raw_response)

    if not isinstance(data, dict) or set(data) != {"relations"}:
        raise ValueError("Expected JSON object with only 'relations'.")

    relations = data["relations"]
    if not isinstance(relations, list):
        raise ValueError("'relations' must be a list.")

    _, id_to_span = make_entity_inventory(tokens, entities)
    expected_types = ["CN", "PN", "PV", "Condition"]

    converted = []
    seen = set()

    for relation in relations:
        if not isinstance(relation, list) or len(relation) not in (3, 4):
            raise ValueError("Each relation requires 3 or 4 entity IDs.")

        spans = []
        for position, entity_id in enumerate(relation):
            if not isinstance(entity_id, str):
                raise ValueError("Entity IDs must be strings.")
            if entity_id not in id_to_span:
                raise ValueError(f"Unknown entity ID: {entity_id}")

            start, end, entity_type = id_to_span[entity_id]
            if entity_type != expected_types[position]:
                raise ValueError(
                    f"Incorrect type for {entity_id}: "
                    f"expected {expected_types[position]}, got {entity_type}"
                )
            spans.append([start, end])

        key = tuple(tuple(span) for span in spans)
        if key in seen:
            raise ValueError("Duplicate relation detected.")

        seen.add(key)
        converted.append(spans)

    return {
        "relations": relations,
        "polyie_relations": converted,
    }


def flatten_segments(data):
    return [
        (d, s, tokens)
        for d, document in enumerate(data["text"])
        for s, tokens in enumerate(document)
    ]


def get_prediction_labels(predictions, test, d, s):
    """Accept a flat list or a nested document/segment list."""
    segments = flatten_segments(test)
    index = next(
        i for i, (di, si, _) in enumerate(segments)
        if (di, si) == (d, s)
    )

    if isinstance(predictions, dict):
        for key in ("predictions", "labels", "predicted_labels"):
            if key in predictions:
                predictions = predictions[key]
                break
        else:
            raise ValueError("Unrecognised prediction JSON object.")

    if not isinstance(predictions, list):
        raise ValueError("Prediction file must contain a list.")

    # Flat segment-level BIO sequences.
    if len(predictions) == len(segments):
        candidate = predictions[index]
        if isinstance(candidate, list) and all(
            isinstance(tag, str) for tag in candidate
        ):
            return candidate

    # Nested document -> segment -> BIO labels.
    if len(predictions) == len(test["text"]):
        candidate = predictions[d][s]
        if isinstance(candidate, list) and all(
            isinstance(tag, str) for tag in candidate
        ):
            return candidate

    raise ValueError(
        "Prediction structure not recognised. "
        "Inspect the original MaterialsBERT export format."
    )


def relation_metrics(gold_relations, predicted_relations):
    def normalise(relations):
        return {
            tuple(tuple(int(x) for x in span) for span in relation)
            for relation in relations
        }

    gold = normalise(gold_relations)
    predicted = normalise(predicted_relations)

    tp = len(gold & predicted)
    fp = len(predicted - gold)
    fn = len(gold - predicted)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Standalone POLYIE GPT RE demonstration"
    )
    parser.add_argument("--train", default="train_split.txt")
    parser.add_argument("--test", default="test_split.txt")
    parser.add_argument("--predictions", default="test_predictions.json")
    parser.add_argument(
        "--ner-source", choices=["gold", "predicted"], default="gold"
    )
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--document", type=int, default=0)
    parser.add_argument("--segment", type=int, default=0)
    parser.add_argument("--output-dir", default="gpt_re_results")
    parser.add_argument(
        "--execute", action="store_true",
        help="Submit a real OpenAI API request"
    )
    args = parser.parse_args()

    train = read_json(args.train)
    test = read_json(args.test)
    demonstrations, demo_hash = prepare_demonstrations(train)

    d, s = args.document, args.segment
    tokens = test["text"][d][s]

    if args.ner_source == "gold":
        labels = test["label"][d][s]
    else:
        labels = get_prediction_labels(
            read_json(args.predictions), test, d, s
        )

    if len(tokens) != len(labels):
        raise ValueError(
            f"Token/label length mismatch: {len(tokens)} vs {len(labels)}"
        )

    entities = bio_to_spans(labels)
    prompt = build_re_prompt(tokens, entities, demonstrations)

    prompt_hash = hashlib.sha256(
        (RE_SYSTEM_PROMPT + "\n" + prompt).encode("utf-8")
    ).hexdigest()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "document": d,
        "segment": s,
        "ner_source": args.ner_source,
        "model": args.model,
        "re_shots": RE_SHOTS,
        "re_demo_hash": demo_hash,
        "prompt_hash": prompt_hash,
        "tokens": tokens,
        "entities": entities,
        "gold_relations": test["relation"][d][s],
    }

    filename = (
        f"gpt_re_{args.ner_source}_doc{d}_seg{s}.json"
    )
    destination = output_dir / filename

    if not args.execute:
        result["status"] = "dry_run"
        result["system_prompt"] = RE_SYSTEM_PROMPT
        result["user_prompt"] = prompt
        print("DRY RUN: No API call made.")
        print(f"Demonstration hash: {demo_hash}")
        print(f"Entities: {len(entities)}")
        print(f"Prompt characters: {len(prompt)}")
    else:
        from openai import OpenAI

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        client = OpenAI()
        response = client.responses.create(
            model=args.model,
            instructions=RE_SYSTEM_PROMPT,
            input=prompt,
        )

        raw_response = response.output_text
        result["raw_output"] = raw_response
        result["response_id"] = response.id
        result["usage"] = (
            response.usage.model_dump() if response.usage else None
        )

        try:
            parsed = parse_gpt_re_response(
                raw_response, tokens, entities
            )
        except (ValueError, json.JSONDecodeError) as exc:
            result["status"] = "validation_failed"
            result["validation_error"] = str(exc)
            destination.write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            raise

        result["status"] = "completed"
        result["predicted_relations"] = parsed["polyie_relations"]
        result["predicted_entity_id_relations"] = parsed["relations"]
        result["metrics"] = relation_metrics(
            result["gold_relations"],
            result["predicted_relations"],
        )

        print(json.dumps(result["metrics"], indent=2))

    destination.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Results saved to: {destination}")


if __name__ == "__main__":
    main()
