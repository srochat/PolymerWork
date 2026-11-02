
import json
from collections import Counter

DATA_PATH = "../Data/Cleaned_data/Final/all.txt"

# Proximity parameters (token distances)
MAX_PN_PV_DISTANCE = 30
MAX_CN_DISTANCE = 80
MAX_CONDITION_DISTANCE = 20

# Start with False to evaluate CN-PN-PV only.
USE_CONDITIONS = False

def distance(a, b):
    """Number of tokens between two entity spans."""
    if a["end_offset"] <= b["start_offset"]:
        return b["start_offset"] - a["end_offset"]
    if b["end_offset"] <= a["start_offset"]:
        return a["start_offset"] - b["end_offset"]
    return 0

def sentence_ids(tokens):
    """Approximate sentence segmentation."""
    result = []
    sid = 0
    for token in tokens:
        result.append(sid)
        if token in {".", "!", "?"} or token.endswith((".", "!", "?")):
            sid += 1
    return result


def predict_relations(doc):
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

        # ----------------------------------------
        # Step 1: Find nearest PN (same as V1)
        # ----------------------------------------
        pn_candidates = candidates(
            "PN", pv, MAX_PN_PV_DISTANCE
        )

        if not pn_candidates:
            continue

        pn = min(
            pn_candidates,
            key=lambda e: (
                distance(e, pv),
                abs(
                    e["start_offset"] -
                    pv["start_offset"]
                )
            )
        )

        # ----------------------------------------
        # Step 2: Find candidate CNs
        # ----------------------------------------
        cn_candidates = [
            e for e in entities
            if e["label"] == "CN"
            and sid(e) == sid(pn)
            and min(
                distance(e, pn),
                distance(e, pv)
            ) <= MAX_CN_DISTANCE
        ]

        if not cn_candidates:
            continue

        # ----------------------------------------
        # V2: Prefer CN preceding PN-PV
        # ----------------------------------------

        # Start of the PN-PV pair
        pair_start = min(
            pn["start_offset"],
            pv["start_offset"]
        )

        # CN must finish before the pair begins
        preceding_cn = [
            e for e in cn_candidates
            if e["end_offset"] <= pair_start
        ]

        if preceding_cn:
            # Select closest preceding CN
            cn = min(
                preceding_cn,
                key=lambda e: (
                    pair_start - e["end_offset"],
                    -e["start_offset"]
                )
            )
        else:
            # Fallback to original nearest-CN rule
            cn = min(
                cn_candidates,
                key=lambda e: (
                    min(
                        distance(e, pn),
                        distance(e, pv)
                    ),
                    distance(e, pn)
                )
            )

        relation = (
            cn["id"],
            pn["id"],
            pv["id"]
        )

        # ----------------------------------------
        # Optional condition matching (same as V1)
        # ----------------------------------------
        if USE_CONDITIONS:
            condition_candidates = [
                e for e in entities
                if e["label"] == "Condition"
                and sid(e) == sid(pv)
                and distance(e, pv)
                    <= MAX_CONDITION_DISTANCE
            ]

            if condition_candidates:
                condition = min(
                    condition_candidates,
                    key=lambda e: distance(e, pv)
                )
                relation += (condition["id"],)

        predicted.add(relation)

    return predicted



def score(gold, predicted):
    tp = len(gold & predicted)
    fp = len(predicted - gold)
    fn = len(gold - predicted)

    return Counter(TP=tp, FP=fp, FN=fn)


def print_metrics(name, counts):
    tp = counts["TP"]
    fp = counts["FP"]
    fn = counts["FN"]

    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    f1 = 2*p*r/(p+r) if p+r else 0

    print(f"\n{name}")
    print(f"TP: {tp} | FP: {fp} | FN: {fn}")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1:        {f1:.4f}")


def evaluate(path):
    with open(path, encoding="utf-8") as f:
        documents = json.load(f)

    overall = Counter()
    triples = Counter()
    quadruples = Counter()

    for doc in documents:
        gold = {tuple(r) for r in doc["relations"]}
        predicted = predict_relations(doc)

        overall.update(score(gold, predicted))

        # Project both 3- and 4-entity relations
        # onto their CN-PN-PV components.
        gold3 = {r[:3] for r in gold if len(r) >= 3}
        pred3 = {r[:3] for r in predicted if len(r) >= 3}

        triples.update(score(gold3, pred3))

        gold4 = {r for r in gold if len(r) == 4}
        pred4 = {r for r in predicted if len(r) == 4}

        quadruples.update(score(gold4, pred4))

    print(f"Documents evaluated: {len(documents)}")
    print_metrics("Overall exact relation matching", overall)
    print_metrics("CN-PN-PV (condition ignored)", triples)
    print_metrics("Full four-entity relations", quadruples)


if __name__ == "__main__":
    evaluate(DATA_PATH)
