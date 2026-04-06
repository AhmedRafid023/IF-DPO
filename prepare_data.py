"""
prepare_dpo_data.py — Download UltraFeedback DPO splits and convert to
the local JSON format that LLaMA-Factory already knows how to read.

Output format per entry:
    {
        "instruction": "<user message>",
        "input":       "",
        "chosen":      "<chosen assistant response>",
        "rejected":    "<rejected assistant response>"
    }

This matches the dataset_info.json entry:
    "columns": {
        "prompt":   "instruction",
        "query":    "input",
        "chosen":   "chosen",
        "rejected": "rejected"
    }

Usage:
    python prepare_dpo_data.py
"""

import json
import os
from datasets import load_dataset
from tqdm import tqdm


# ── Config — edit before running ──────────────────────────────────────────────
class Config:
    HF_DATASET  = "HuggingFaceH4/ultrafeedback_binarized"
    OUTPUT_DIR  = "data"
    TRAIN_FILE  = "ultrafeedback_dpo_train.json"
    TEST_FILE   = "ultrafeedback_dpo_test.json"
    DATASET_INFO = "data/dataset_info.json"

    # Set to an int (e.g. 1000) to download a subset, None for the full dataset
    TRAIN_LIMIT = None
    TEST_LIMIT  = None
# ──────────────────────────────────────────────────────────────────────────────


def extract_text(messages: list, role: str) -> str:
    """Extract the text content of the first message matching the given role."""
    for msg in messages:
        if msg.get("role") == role:
            return msg.get("content", "").strip()
    return ""


def convert_split(split_name: str, limit: int | None) -> list:
    """Download a split and convert each example to the local format."""
    print(f"\n📥 Loading {Config.HF_DATASET} / {split_name}...")
    ds = load_dataset(Config.HF_DATASET, split=split_name)

    if limit:
        ds = ds.select(range(min(limit, len(ds))))
        print(f"⚠️  Using subset of {len(ds)} examples.")
    else:
        print(f"   {len(ds)} examples loaded.")

    data    = []
    dropped = 0

    for ex in tqdm(ds, desc=f"Converting {split_name}"):
        chosen   = ex.get("chosen",   [])
        rejected = ex.get("rejected", [])

        instruction    = extract_text(chosen, "user")
        chosen_text    = extract_text(chosen, "assistant")
        rejected_text  = extract_text(rejected, "assistant")

        # Skip examples missing any required field
        if not instruction or not chosen_text or not rejected_text:
            dropped += 1
            continue

        data.append({
            "instruction": instruction,
            "input":       "",
            "chosen":      chosen_text,
            "rejected":    rejected_text,
        })

    print(f"   ✅ Converted: {len(data)} | Dropped (incomplete): {dropped}")
    return data


def save_json(data: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"   💾 Saved → {path}")


def update_dataset_info(info_path: str, train_file: str, test_file: str):
    """Register both splits in dataset_info.json."""
    info = {}
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)

    columns = {
        "prompt":   "instruction",
        "query":    "input",
        "chosen":   "chosen",
        "rejected": "rejected",
    }

    info["ultrafeedback_dpo"] = {
        "file_name": train_file,
        "ranking":   True,
        "columns":   columns,
    }
    info["ultrafeedback_dpo_test"] = {
        "file_name": test_file,
        "ranking":   True,
        "columns":   columns,
    }

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"\n   📋 dataset_info.json updated with 'ultrafeedback_dpo' and 'ultrafeedback_dpo_test'")


def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # Convert train split
    train_data = convert_split("train_prefs", Config.TRAIN_LIMIT)
    train_path = os.path.join(Config.OUTPUT_DIR, Config.TRAIN_FILE)
    save_json(train_data, train_path)

    # Convert test split
    test_data = convert_split("test_prefs", Config.TEST_LIMIT)
    test_path = os.path.join(Config.OUTPUT_DIR, Config.TEST_FILE)
    save_json(test_data, test_path)

    # Update dataset_info.json
    update_dataset_info(Config.DATASET_INFO, Config.TRAIN_FILE, Config.TEST_FILE)

    print(f"\n{'='*50}")
    print(f"  Done!")
    print(f"{'='*50}")
    print(f"  Train : {len(train_data):,} examples → {train_path}")
    print(f"  Test  : {len(test_data):,} examples  → {test_path}")
    print(f"\n  Example entry:")
    ex = train_data[0]
    print(f"    instruction : {ex['instruction'][:80]}...")
    print(f"    chosen      : {ex['chosen'][:80]}...")
    print(f"    rejected    : {ex['rejected'][:80]}...")


if __name__ == "__main__":
    main()