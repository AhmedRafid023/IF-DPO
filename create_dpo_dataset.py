"""
create_dpo_dataset.py — Build IF Self-DPO datasets from influence scores.

Two pairing strategies:
  --strategy symmetric     rank i  (chosen)  vs rank n-1-i  (rejected) — mirror pairs
  --strategy next_neighbor rank i  (chosen)  vs rank i+1    (rejected) — adjacent pairs
  --strategy both          run both (default)

Usage:
    python create_dpo_dataset.py
    python create_dpo_dataset.py --strategy symmetric
    python create_dpo_dataset.py --strategy next_neighbor
"""

import argparse
import json
import os
import pandas as pd


# ── Config — edit before running ──────────────────────────────────────────────
class Config:
    CSV_INPUT      = "influence_scoring_results/influence_comparison.csv"
    OUTPUT_DIR     = "data"
    DATASET_INFO   = "data/dataset_info.json"

    # Output filenames
    SYMMETRIC_JSON     = "if_self_dpo_train.json"
    NEXT_NEIGHBOR_JSON = "if_self_dpo_train_next.json"
# ──────────────────────────────────────────────────────────────────────────────


def clean_text(text):
    """Remove invisible Unicode line/paragraph separators that break JSON."""
    if not isinstance(text, str):
        return text
    return text.replace("\u2028", " ").replace("\u2029", " ")


def load_and_sort(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}\nRun influence_scoring.py first.")
    df = pd.read_csv(csv_path).fillna("")
    return df.sort_values("score_datainf", ascending=False).reset_index(drop=True)


def build_entry(winner: pd.Series, loser: pd.Series) -> dict:
    return {
        "instruction": clean_text(winner["instruction"]),
        "input":       clean_text(winner.get("input", "")),
        "chosen":      clean_text(winner["chosen"]),
        "rejected":    clean_text(loser["chosen"]),   # loser's chosen becomes the rejected response
    }


def save_json(data: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"   Saved {len(data)} pairs → {path}")


def update_dataset_info(info_path: str, key: str, filename: str):
    """Register the new dataset in dataset_info.json."""
    info = {}
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)

    info[key] = {
        "file_name": filename,
        "ranking":   True,
        "columns": {
            "prompt":   "instruction",
            "query":    "input",
            "chosen":   "chosen",
            "rejected": "rejected",
        },
    }

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"   dataset_info.json updated with key '{key}'")


# ── Pairing strategies ─────────────────────────────────────────────────────────

def create_symmetric(df: pd.DataFrame) -> list:
    """
    Pair rank i with rank n-1-i.
    Rank 0 (best) vs rank n-1 (worst), rank 1 vs rank n-2, etc.
    Creates the widest possible quality gap for each pair.
    """
    n    = len(df)
    data = []
    for i in range(n):
        winner = df.iloc[i]
        loser  = df.iloc[n - 1 - i]
        data.append(build_entry(winner, loser))
    return data


def create_next_neighbor(df: pd.DataFrame) -> list:
    """
    Pair rank i with rank i+1 (circular).
    Creates fine-grained adjacent pairs — each winner is only slightly better than its loser.
    """
    n    = len(df)
    data = []
    for i in range(n):
        winner = df.iloc[i]
        loser  = df.iloc[(i + 1) % n]
        data.append(build_entry(winner, loser))
    return data


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["symmetric", "next_neighbor", "both"],
                        default="both")
    args = parser.parse_args()

    print(f"📂 Loading influence scores from {Config.CSV_INPUT}...")
    df = load_and_sort(Config.CSV_INPUT)
    print(f"   {len(df)} examples loaded and sorted by influence score.")

    run_sym  = args.strategy in ("symmetric",     "both")
    run_next = args.strategy in ("next_neighbor",  "both")

    if run_sym:
        print("\n🔀 Strategy: Symmetric (mirror pairs)")
        data = create_symmetric(df)
        path = os.path.join(Config.OUTPUT_DIR, Config.SYMMETRIC_JSON)
        save_json(data, path)
        update_dataset_info(Config.DATASET_INFO, "if_self_dpo_train", Config.SYMMETRIC_JSON)

    if run_next:
        print("\n🔁 Strategy: Next-neighbor (adjacent pairs)")
        data = create_next_neighbor(df)
        path = os.path.join(Config.OUTPUT_DIR, Config.NEXT_NEIGHBOR_JSON)
        save_json(data, path)
        update_dataset_info(Config.DATASET_INFO, "if_self_dpo_train_next", Config.NEXT_NEIGHBOR_JSON)

    print("\n✅ Done.")


if __name__ == "__main__":
    main()