"""Train a candidate and tune its decision threshold on validation data only."""


from __future__ import annotations


import argparse
from datetime import datetime, timezone
from uuid import uuid4


import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


from pipeline_utils import (
    CANDIDATE_METADATA_PATH, CANDIDATE_PATH, CAT_COLS, FEATURES,
    MODELS_DIR, NUM_COLS, REGISTRY_DIR, ROOT, SPLITS_DIR, TARGET,
    sha256, write_json,
)



def find_best_threshold(y_true, y_proba) -> tuple[float, float]:
    """Select among 81 thresholds; break ties by proximity to 0.5."""
    thresholds = np.linspace(0.1, 0.9, 81)
    scores = [
        float(f1_score(y_true, y_proba >= t, zero_division=0))
        for t in thresholds
    ]
    best = max(
        range(len(thresholds)),
        key=lambda i: (scores[i], -abs(float(thresholds[i]) - 0.5)),
    )
    return float(thresholds[best]), scores[best]



def main(version: str = "v1", seed: int = 42) -> None:
    data_path = ROOT / "data" / "processed.csv"
    if not data_path.exists():
        raise FileNotFoundError("Exécutez d'abord prepare_data.py.")


    # "NA" is a valid region, not a missing-value marker.
    df = pd.read_csv(data_path, keep_default_na=False, na_values=[""])
    df = df[FEATURES + [TARGET]].copy()
    if df.isna().any().any():
        raise ValueError("Valeurs manquantes : nettoyer ou imputer avant fit.")
    if not set(df[TARGET].unique()) == {0, 1}:
        raise ValueError("churn doit contenir les deux classes 0 et 1.")


    # 60% training, 15% validation, 25% held-out test (up to rounding).
    development, test = train_test_split(
        df, test_size=0.25, random_state=seed, stratify=df[TARGET],
    )
    train, validation = train_test_split(
        development, test_size=0.20, random_state=seed,
        stratify=development[TARGET],
    )


    pipe = Pipeline([
        ("prep", ColumnTransformer([
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ])),
        ("clf", LogisticRegression(max_iter=200, random_state=seed)),
    ])
    # The scaler, encoder and classifier learn from training rows only.
    pipe.fit(train[FEATURES], train[TARGET])
    validation_proba = pipe.predict_proba(validation[FEATURES])[:, 1]
    threshold, validation_f1 = find_best_threshold(
        validation[TARGET], validation_proba,
    )


    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train), ("validation", validation), ("test", test)]:
        split.to_csv(SPLITS_DIR / f"{name}.csv", index=False)


    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, CANDIDATE_PATH)


    # These are the actual fitted StandardScaler parameters (ddof=0).
    scaler = pipe.named_steps["prep"].named_transformers_["num"]
    stats = {
        col: {"mean": float(mean), "std": float(np.sqrt(var)), "scale": float(scale)}
        for col, mean, var, scale in zip(
            NUM_COLS, scaler.mean_, scaler.var_, scaler.scale_,
        )
    }
    write_json(REGISTRY_DIR / "train_stats.json", stats)


    metadata = {
        "run_id": uuid4().hex,
        "version": version,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "sklearn_version": sklearn.__version__,
        "model_file": CANDIDATE_PATH.name,
        "model_sha256": sha256(CANDIDATE_PATH),
        "data_sha256": sha256(data_path),
        "test_sha256": sha256(SPLITS_DIR / "test.csv"),
        "features": FEATURES,
        "target": TARGET,
        "threshold": threshold,
        "threshold_selected_on": "validation",
        "validation_f1": validation_f1,
        # Choose the majority-class baseline using training labels only.
        "baseline_class": int(train[TARGET].mode().iloc[0]),
        "split_rows": {
            "train": len(train), "validation": len(validation), "test": len(test),
        },
    }
    write_json(CANDIDATE_METADATA_PATH, metadata)
    print(f"[OK] Candidat entraîné : {CANDIDATE_PATH}")
    print(f"[VALIDATION] threshold={threshold:.2f}, F1={validation_f1:.4f}")
    print(f"[OK] Paramètres du scaler : {REGISTRY_DIR / 'train_stats.json'}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(version=args.version, seed=args.seed)