"""Apply an F1 gate, keep a versioned candidate and promote accepted models."""



from __future__ import annotations



import argparse
import math
import shutil
from datetime import datetime, timezone



from pipeline_utils import (
    CANDIDATE_METADATA_PATH, CANDIDATE_PATH, METRICS_PATH, MODELS_DIR,
    REGISTRY_DIR, TEST_PATH, read_json, sha256, write_json,
)




def main(gate_f1: float = 0.70) -> bool:
    if not math.isfinite(gate_f1) or not 0 <= gate_f1 <= 1:
        raise ValueError("gate_f1 doit être compris entre 0 et 1.")
    metadata = read_json(CANDIDATE_METADATA_PATH)
    metrics = read_json(METRICS_PATH)



    # Reject old reports, including reports for an earlier threshold.
    expected = {
        "run_id": metadata["run_id"],
        "model_sha256": sha256(CANDIDATE_PATH),
        "test_sha256": sha256(TEST_PATH),
        "candidate_metadata_sha256": sha256(CANDIDATE_METADATA_PATH),
        "threshold": metadata["threshold"],
    }
    if any(metrics.get(key) != value for key, value in expected.items()):
        raise ValueError("Rapport périmé ou incohérent : relancez evaluate.py.")
    if metadata["model_sha256"] != expected["model_sha256"]:
        raise ValueError("Le candidat a changé : relancez train.py et evaluate.py.")
    if metadata["test_sha256"] != expected["test_sha256"]:
        raise ValueError("Le jeu de test a changé : relancez train.py.")
    for key in ["f1", "baseline_f1"]:
        if not math.isfinite(metrics[key]) or not 0 <= metrics[key] <= 1:
            raise ValueError(f"Métrique invalide : {key}")



    passed = metrics["f1"] >= gate_f1 and metrics["f1"] > metrics["baseline_f1"]
    # Use the run UUID for a safe, unique filename on Windows and Unix.
    filename = f"churn_model_{metadata['run_id']}.joblib"
    model_path = MODELS_DIR / filename
    shutil.copyfile(CANDIDATE_PATH, model_path)
    entry = {
        **metadata,
        "model_file": filename,
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "gate_f1": gate_f1,
        "passed_gate": passed,
    }



    history_path = REGISTRY_DIR / "metadata.json"
    history = read_json(history_path) if history_path.exists() else []
    # Re-registering the same run updates its entry, without duplicates.
    history = [item for item in history if item.get("run_id") != metadata["run_id"]]
    history.append(entry)
    write_json(history_path, history)



    if passed:
        shutil.copyfile(model_path, MODELS_DIR / "model.joblib")
        write_json(REGISTRY_DIR / "current_model.json", entry)
        (REGISTRY_DIR / "current_model.txt").write_text(filename, encoding="utf-8")
        print(f"[REGISTER] Modèle accepté et activé dans le registre local : {filename}")
    else:
        print(
            f"[REGISTER] Candidat refusé : F1={metrics['f1']:.4f}, "
            f"minimum={gate_f1:.4f}, baseline={metrics['baseline_f1']:.4f}. "
            "Le modèle courant est conservé."
        )
    return passed




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-f1", type=float, default=0.70)
    args = parser.parse_args()
    raise SystemExit(0 if main(gate_f1=args.gate_f1) else 1)