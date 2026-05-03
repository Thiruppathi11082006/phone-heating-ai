import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


DATASET_PATH = Path("phone_heating_dataset.csv")
MODEL_PATH = Path("heating_model.pkl")
SUMMARY_PATH = Path("training_summary.json")
FEATURE_COLUMNS = ["cpu", "temp", "charging", "brightness", "apps", "network"]


def clamp(values):
    bounds = {
        "cpu": (10, 100),
        "temp": (28.0, 52.0),
        "charging": (0, 1),
        "brightness": (20, 100),
        "apps": (1, 15),
        "network": (5, 100),
    }
    cleaned = {}
    for column, value in values.items():
        low, high = bounds[column]
        if column == "charging":
            cleaned[column] = int(np.clip(round(value), low, high))
        elif column == "temp":
            cleaned[column] = round(float(np.clip(value, low, high)), 2)
        else:
            cleaned[column] = int(np.clip(round(value), low, high))
    return cleaned


def augment_critical_cases(df, target_count=120, seed=42):
    rng = np.random.default_rng(seed)
    critical_df = df[df["risk"] == 2].copy()
    if critical_df.empty:
        raise ValueError("Dataset has no critical samples to augment.")

    synthetic_rows = []
    while len(synthetic_rows) + len(critical_df) < target_count:
        base_row = critical_df.sample(n=1, replace=True, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        synthetic = {
            "cpu": base_row["cpu"] + rng.normal(8, 6),
            "temp": base_row["temp"] + rng.normal(1.8, 0.9),
            "charging": 1 if rng.random() < 0.7 else int(base_row["charging"]),
            "brightness": base_row["brightness"] + rng.normal(6, 10),
            "apps": base_row["apps"] + rng.normal(1.5, 1.2),
            "network": base_row["network"] + rng.normal(8, 10),
            "risk": 2,
        }
        synthetic = clamp({key: value for key, value in synthetic.items() if key != "risk"})
        synthetic["risk"] = 2
        synthetic_rows.append(synthetic)

    heuristic_rows = []
    while len(heuristic_rows) < 60:
        row = {
            "cpu": rng.integers(78, 101),
            "temp": rng.uniform(44.0, 50.5),
            "charging": int(rng.random() < 0.65),
            "brightness": rng.integers(65, 101),
            "apps": rng.integers(6, 13),
            "network": rng.integers(55, 101),
            "risk": 2,
        }
        row = clamp({key: value for key, value in row.items() if key != "risk"})
        row["risk"] = 2
        heuristic_rows.append(row)

    augmented = pd.concat(
        [df, pd.DataFrame(synthetic_rows), pd.DataFrame(heuristic_rows)],
        ignore_index=True,
    )
    return augmented


def train_model(df):
    x_train, x_test, y_train, y_test = train_test_split(
        df[FEATURE_COLUMNS],
        df["risk"],
        test_size=0.25,
        random_state=42,
        stratify=df["risk"],
    )
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=1,
        random_state=42,
        class_weight={0: 1.0, 1: 1.2, 2: 3.8},
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2],
        target_names=["Safe", "Warning", "Critical"],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1, 2]).tolist()
    return model, report, matrix


def main():
    df = pd.read_csv(DATASET_PATH)
    original_counts = df["risk"].value_counts().sort_index().to_dict()
    augmented_df = augment_critical_cases(df)
    augmented_counts = augmented_df["risk"].value_counts().sort_index().to_dict()

    model, report, matrix = train_model(augmented_df)
    joblib.dump(model, MODEL_PATH)

    summary = {
        "model_name": "Critical-aware RandomForestClassifier",
        "feature_columns": FEATURE_COLUMNS,
        "original_class_counts": {str(key): int(value) for key, value in original_counts.items()},
        "augmented_class_counts": {str(key): int(value) for key, value in augmented_counts.items()},
        "test_accuracy": round(float(report["accuracy"]), 4),
        "critical_precision": round(float(report["Critical"]["precision"]), 4),
        "critical_recall": round(float(report["Critical"]["recall"]), 4),
        "critical_f1": round(float(report["Critical"]["f1-score"]), 4),
        "confusion_matrix": matrix,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
