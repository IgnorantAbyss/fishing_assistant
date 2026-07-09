"""Print a concise, read-only summary from replay_results.csv."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarise an existing replay detector CSV.")
    parser.add_argument("--session", required=True, type=Path)
    args = parser.parse_args()
    results_path = args.session / "replay_results.csv"
    if not results_path.is_file():
        raise FileNotFoundError(f"Replay results not found: {results_path}")
    with results_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    counts = Counter(row["state"] for row in rows)
    print(f"frames: {len(rows)}")
    for state in ("IDLE", "WAITING", "READY", "HOOK", "PRESS", "GET", "UNKNOWN"):
        print(f"{state}: {counts.get(state, 0)}")
    unknown_ratio = counts.get("UNKNOWN", 0) / len(rows) if rows else 0.0
    print(f"UNKNOWN ratio: {unknown_ratio:.2%}")
    transitions: list[str] = []
    previous: str | None = None
    for row in rows:
        state = row["state"]
        if previous is not None and state != previous:
            transitions.append(f"{previous}->{state}")
        previous = state
    print("transitions: " + (", ".join(transitions) if transitions else "None"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
