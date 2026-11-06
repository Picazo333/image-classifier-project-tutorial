"""Entry point for the dogs-vs-cats classifier.

Training is the executed notebook ``src/explore.ipynb``.
This script checks the official zip, builds the class folders, and prints
saved test metrics when a checkpoint already exists.

``src/utils.py`` still has the template database helper. This project does
not open a database and does not read DATABASE_URL.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pipeline import MODEL_PATH, REPORTS, prepare_all, set_seed


def main() -> int:
    set_seed()
    summary = prepare_all(run_survey=True)
    public = {
        "cats": summary["extracted"]["cats"],
        "dogs": summary["extracted"]["dogs"],
        "split_counts": summary["split"]["counts"],
        "zip_sha256": summary["zip"]["sha256"],
        "zip_bytes": summary["zip"]["bytes"],
    }
    print(json.dumps(public, indent=2))
    metrics_path = REPORTS / "evaluation.json"
    if MODEL_PATH.is_file() and metrics_path.is_file():
        print(metrics_path.read_text())
        print(f"modelo: {MODEL_PATH} ({MODEL_PATH.stat().st_size} bytes)")
        print(
            "models/dogs_vs_cats_best.keras pesa menos de 100 MB y está en el "
            "repositorio. Otros *.keras siguen fuera de git."
        )
    else:
        print(
            "No hay checkpoint todavía. El entrenamiento está en "
            "src/explore.ipynb. El archivo de menos de 100 MB se versiona; "
            "el resto de *.keras no."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
