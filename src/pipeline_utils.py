"""Shared paths and helpers for the lab."""


from __future__ import annotations


import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REGISTRY_DIR = ROOT / "registry"
SPLITS_DIR = ROOT / "data" / "splits"


CANDIDATE_PATH = MODELS_DIR / "candidate.joblib"
CANDIDATE_METADATA_PATH = REGISTRY_DIR / "candidate.json"
TEST_PATH = SPLITS_DIR / "test.csv"
METRICS_PATH = ROOT / "reports" / "metrics.json"


NUM_COLS = [
    "tenure_months",
    "num_complaints",
    "avg_session_minutes",
]
CAT_COLS = ["plan_type", "region"]
FEATURES = NUM_COLS + CAT_COLS
TARGET = "churn"



def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))



def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )



def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()