"""Evaluate the saved candidate on the held-out test; never train or tune."""


from __future__ import annotations


from datetime import datetime, timezone


import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


from pipeline_utils import (
    CANDIDATE_METADATA_PATH, CANDIDATE_PATH, METRICS_PATH, TEST_PATH,
    read_json, sha256, write_json,
)



def main() -> None:
    for path in [CANDIDATE_PATH, CANDIDATE_METADATA_PATH, TEST_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Fichier absent : {path}. Exécutez train.py.")


    metadata = read_json(CANDIDATE_METADATA_PATH)
    model_hash = sha256(CANDIDATE_PATH)
    test_hash = sha256(TEST_PATH)
    if model_hash != metadata["model_sha256"]:
        raise ValueError("Le modèle ne correspond pas aux métadonnées du candidat.")
    if test_hash != metadata["test_sha256"]:
        raise ValueError("Le test ne correspond pas au découpage enregistré.")


    pipe = joblib.load(CANDIDATE_PATH)
    test = pd.read_csv(TEST_PATH, keep_default_na=False, na_values=[""])
    X_test = test[metadata["features"]]
    y_test = test[metadata["target"]].astype(int)
    threshold = float(metadata["threshold"])


    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    baseline_pred = np.full(len(y_test), metadata["baseline_class"], dtype=int)


    # All main metrics use the same fixed validation-selected threshold.
    metrics = {
        "run_id": metadata["run_id"],
        "model_sha256": model_hash,
        "test_sha256": test_hash,
        "candidate_metadata_sha256": sha256(CANDIDATE_METADATA_PATH),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_rows": len(test),
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "f1_threshold_05": float(f1_score(y_test, y_proba >= 0.5, zero_division=0)),
        "baseline_class": metadata["baseline_class"],
        "baseline_f1": float(f1_score(y_test, baseline_pred, zero_division=0)),
    }
    write_json(METRICS_PATH, metrics)
    print(f"[TEST] F1={metrics['f1']:.4f}, baseline F1={metrics['baseline_f1']:.4f}")
    print(f"[OK] Évaluation enregistrée : {METRICS_PATH}")



if __name__ == "__main__":
    main()