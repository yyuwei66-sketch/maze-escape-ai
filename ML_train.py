from __future__ import annotations

import json
import os
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

REPLAY_LOG = "replay_log.jsonl"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "direction_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = [
    "rel_row",
    "rel_col",
    "wall_up",
    "wall_down",
    "wall_left",
    "wall_right",
    "prev1",
    "prev2",
]

DIRECTIONS = ["up", "down", "left", "right"]


def load_records():
    """Reads game log records from a JSONL file."""
    records = []

    if not os.path.exists(REPLAY_LOG):
        return records

    with open(REPLAY_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "features" in rec and "label" in rec:
                    records.append(rec)
            except (json.JSONDecodeError, KeyError):
                pass

    return records


def records_to_xy(records):
    """Converts the raw records into structured features (X) and labels (y)."""
    X = []
    y = []

    for rec in records:
        feat = rec["features"]
        X.append(
            [
                float(feat["rel_row"]),
                float(feat["rel_col"]),
                float(feat["wall_up"]),
                float(feat["wall_down"]),
                float(feat["wall_left"]),
                float(feat["wall_right"]),
                float(feat["prev1"]),
                float(feat["prev2"]),
            ]
        )
        y.append(rec["label"])

    return X, y


def train():
    """Trains a RandomForestClassifier using the collected game log records."""
    records = load_records()
    print(f"Loaded {len(records)} samples")

    if len(records) < 100:
        print("Insufficient data: At least 100 samples are required.")
        return

    X, y = records_to_xy(records)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)

    print("\n========== Result ==========")
    print(f"Accuracy = {acc:.4f}\n")
    print(classification_report(y_test, pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)

    print(f"\nModel saved -> {MODEL_PATH}")


def predict_direction(feature_dict):
    """Predicts the next direction based on a single real-time feature dictionary."""
    if not os.path.exists(MODEL_PATH):
        print("Model file does not exist.")
        return None

    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)

    x = [
        [
            feature_dict["rel_row"],
            feature_dict["rel_col"],
            feature_dict["wall_up"],
            feature_dict["wall_down"],
            feature_dict["wall_left"],
            feature_dict["wall_right"],
            feature_dict["prev1"],
            feature_dict["prev2"],
        ]
    ]

    pred = clf.predict(x)[0]
    prob = clf.predict_proba(x)[0]

    print(f"\nPredicted Direction = {pred}\n")
    for d, p in zip(clf.classes_, prob):
        print(f"{d:<8} {p:.3f}")

    return pred


if __name__ == "__main__":
    train()
