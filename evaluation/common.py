"""Shared helpers for AQUILA's evaluation scripts."""

from __future__ import annotations

import csv
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def write_csv(filename: str, fieldnames: list[str], rows: list[dict]) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / filename
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
