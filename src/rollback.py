"""Restore an accepted model and its metadata in the local lab registry."""


from __future__ import annotations


import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path


import joblib



ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REGISTRY_DIR = ROOT / "registry"
METADATA_PATH = REGISTRY_DIR / "metadata.json"
CURRENT_MODEL_PATH = REGISTRY_DIR / "current_model.txt"
NUM_COLS = ["tenure_months", "num_complaints", "avg_session_minutes"]



def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )



def main(target: str | None = None) -> None:
    if not METADATA_PATH.exists():
        raise FileNotFoundError("metadata.json absent : exécutez register.py.")
    items = json.loads(METADATA_PATH.read_text(encoding="utf-8-sig"))


    # A rejected candidate is not eligible for rollback.
    accepted = [item for item in items if item.get("passed_gate") is True]
    if not accepted:
        raise ValueError("Aucun modèle accepté dans le registre.")


    current = (
        CURRENT_MODEL_PATH.read_text(encoding="utf-8-sig").strip()
        if CURRENT_MODEL_PATH.exists() else None
    )
    if target is None:
        names = [item["model_file"] for item in accepted]
        if current not in names:
            raise ValueError(
                "Modèle courant absent des modèles acceptés. "
                "Utilisez --target avec un nom accepté."
            )
        index = names.index(current)
        if index == 0:
            raise ValueError("Aucun modèle accepté antérieur au modèle courant.")
        selected = accepted[index - 1]
    else:
        selected = next(
            (item for item in accepted if item["model_file"] == target), None,
        )
        if selected is None:
            raise ValueError(f"Modèle inconnu ou refusé par le gate : {target}")


    name = selected["model_file"]
    if Path(name).name != name:
        raise ValueError("model_file doit être un simple nom de fichier.")
    path = MODELS_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    expected_hash = selected.get("model_sha256")
    if expected_hash:
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError("Le fichier modèle ne correspond pas à son empreinte.")


    # Support metadata from both the corrected pipeline and the original lab.
    threshold = selected.get("threshold")
    if threshold is None:
        threshold = selected.get("metrics", {}).get("best_threshold")
    if threshold is None:
        raise ValueError("Seuil de décision absent des métadonnées du modèle.")
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Seuil de décision invalide.")


    # Recover reference statistics from the saved scaler; do not train again.
    model = joblib.load(path)
    scaler = model.named_steps["prep"].named_transformers_["num"]
    if hasattr(scaler, "named_steps"):
        scaler = scaler.named_steps["scaler"]
    if not len(scaler.mean_) == len(scaler.var_) == len(scaler.scale_) == len(NUM_COLS):
        raise ValueError("Le scaler n'est pas compatible avec les variables du lab.")
    stats = {
        col: {"mean": float(mean), "std": math.sqrt(float(var)), "scale": float(scale)}
        for col, mean, var, scale in zip(
            NUM_COLS, scaler.mean_, scaler.var_, scaler.scale_,
        )
    }
    entry = {**selected, "threshold": threshold}


    # All validation above finishes before any current-model file is changed.
    shutil.copyfile(path, MODELS_DIR / "model.joblib")
    write_json(REGISTRY_DIR / "current_model.json", entry)
    write_json(REGISTRY_DIR / "train_stats.json", stats)
    CURRENT_MODEL_PATH.write_text(name, encoding="utf-8")


    print(f"[OK] Modèle courant : {current or '(aucun)'} -> {name}")
    print(f"[OK] Seuil restauré : {threshold:.3f}")
    print("[OK] Alias, métadonnées et statistiques de référence synchronisés.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", help="Nom exact du fichier modèle accepté à activer.")
    args = parser.parse_args()
    try:
        main(target=args.target)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(status=1, message=f"[ERREUR] {exc}\n")