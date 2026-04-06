"""
create_random_dpo_dataset.py — Build a Self-DPO dataset using random pairing.

Strategy:
    For each sample i (chosen), randomly assign a different sample j (rejected)
    such that:
      - j != i  (a sample cannot be paired with itself)
      - every sample is used as 'rejected' exactly once (no reuse)

    This is achieved by creating a random permutation of indices and shifting
    it by at least 1 position, guaranteeing a derangement-like assignment
    with zero reuse.

Usage:
    python create_random_dpo_dataset.py

Output:
    data/if_self_dpo_train_random.json
    dataset_info.json updated with key 'if_self_dpo_train_random'
"""

import json
import os
import random
import pandas as pd


# ── Config — edit before running ──────────────────────────────────────────────
class Config:
    CSV_INPUT    = "influence_scoring_results/influence_comparison.csv"
    OUTPUT_DIR   = "data"
    OUTPUT_JSON  = "if_self_dpo_train_random.json"
    DATASET_INFO = "data/dataset_info.json"
    SEED         = 42
# ──────────────────────────────────────────────────────────────────────────────


def clean_text(text):
    """Remove invisible Unicode line/paragraph separators that break JSON."""
    if not isinstance(text, str):
        return text
    return text.replace("\u2028", " ").replace("\u2029", " ")


def make_derangement(n: int, rng: random.Random) -> list[int]:
    """
    Returns a permutation of [0, n) where no index maps to itself,
    and every index appears exactly once as a 'rejected' target.

    Strategy: shuffle, then shift by a random offset in [1, n-1].
    This guarantees:
      - No self-pairing (offset >= 1)
      - Every sample used as rejected exactly once (it's a permutation)
    """
    indices = list(range(n))
    rng.shuffle(indices)

    # Pick a random shift amount between 1 and n-1
    shift = rng.randint(1, n - 1)
    rejected_indices = indices[shift:] + indices[:shift]

    # Verify no self-pairing (should never happen with shift >= 1 on shuffled list,
    # but we fix any accidental collisions by swapping with the next position)
    for i in range(n):
        if rejected_indices[i] == indices[i]:
            swap = (i + 1) % n
            rejected_indices[i], rejected_indices[swap] = rejected_indices[swap], rejected_indices[i]

    return list(zip(indices, rejected_indices))


def main():
    if not os.path.exists(Config.CSV_INPUT):
        raise FileNotFoundError(
            f"CSV not found: {Config.CSV_INPUT}\n"
            "Run influence_scoring.py first."
        )

    rng = random.Random(Config.SEED)

    print(f"📂 Loading scores from {Config.CSV_INPUT}...")
    df = pd.read_csv(Config.CSV_INPUT).fillna("")
    n  = len(df)
    print(f"   {n} examples loaded.")

    print(f"\n🎲 Building random derangement pairs (seed={Config.SEED})...")
    pairs = make_derangement(n, rng)

    # Verify no self-pairs and no repeated rejected indices
    chosen_indices   = [p[0] for p in pairs]
    rejected_indices = [p[1] for p in pairs]
    assert len(set(rejected_indices)) == n,   "❌ Some rejected samples used more than once!"
    assert all(c != r for c, r in pairs),     "❌ Self-pair detected!"
    print(f"   ✅ Verified: no self-pairs, no repeated rejected samples.")

    # Build dataset entries
    data = []
    for chosen_idx, rejected_idx in pairs:
        winner = df.iloc[chosen_idx]
        loser  = df.iloc[rejected_idx]
        data.append({
            "instruction": clean_text(winner["instruction"]),
            "input":       clean_text(winner.get("input", "")),
            "chosen":      clean_text(winner["chosen"]),
            "rejected":    clean_text(loser["chosen"]),
        })

    # Save JSON
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(Config.OUTPUT_DIR, Config.OUTPUT_JSON)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n   Saved {len(data)} pairs → {output_path}")

    # Update dataset_info.json
    info = {}
    if os.path.exists(Config.DATASET_INFO):
        with open(Config.DATASET_INFO, "r", encoding="utf-8") as f:
            info = json.load(f)

    info["if_self_dpo_train_random"] = {
        "file_name": Config.OUTPUT_JSON,
        "ranking":   True,
        "columns": {
            "prompt":   "instruction",
            "query":    "input",
            "chosen":   "chosen",
            "rejected": "rejected",
        },
    }

    with open(Config.DATASET_INFO, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"   dataset_info.json updated with key 'if_self_dpo_train_random'")
    print("\n✅ Done.")


if __name__ == "__main__":
    main()