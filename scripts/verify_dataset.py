#!/usr/bin/env python3
"""Check that the expected CSV files and key columns are present (stdlib only)."""

import csv
import sys
from pathlib import Path


def verify_dataset() -> bool:
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data" / "ml-latest-small-filtered"
    required = {
        "movies_with_plots.csv": {"movieId", "title", "year", "genres", "plot"},
        "ratings.csv": {"userId", "movieId", "rating", "timestamp"},
        "tags.csv": {"userId", "movieId", "tag", "timestamp"},
        "links.csv": {"movieId", "imdbId", "tmdbId"},
        "movies.csv": {"movieId", "title", "genres"},
    }
    if not data_dir.is_dir():
        print(f"Dataset directory not found: {data_dir}")
        return False
    ok = True
    for filename, expected_columns in required.items():
        path = data_dir / filename
        if not path.is_file():
            print(f"MISSING  {filename}")
            ok = False
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            count = sum(1 for _ in reader)
        absent = expected_columns - columns
        if absent:
            print(f"INVALID  {filename}: missing columns {', '.join(sorted(absent))}")
            ok = False
        else:
            print(f"OK       {filename}: {count:,} rows, {path.stat().st_size / 1024 / 1024:.2f} MiB")
    print("Dataset check passed." if ok else "Dataset check failed.")
    return ok


if __name__ == "__main__":
    sys.exit(0 if verify_dataset() else 1)
