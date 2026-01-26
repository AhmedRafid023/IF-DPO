import argparse
import json
import os
import random
from datasets import load_dataset
from sklearn.model_selection import train_test_split

random.seed(42)

def format_dpo_data(example):
    # The dataset provides a list: [{"role": "user", ...}, {"role": "assistant", "content": "..."}]
    # We select the last message [-1] and get its "content"
    chosen_text = example["chosen"][-1]["content"]
    rejected_text = example["rejected"][-1]["content"]

    return {
        "instruction": example["prompt"].strip(),
        "input": "",
        "chosen": chosen_text.strip(),
        "rejected": rejected_text.strip()
    }

def update_dataset_info():
    info = {
        "dpo_train": {
            "file_name": "train.json",
            "ranking": True,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "chosen": "chosen",
                "rejected": "rejected"
            }
        },
        "self_dpo_train": {
            "file_name": "self_dpo_train.json",
            "ranking": True,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "chosen": "chosen",
                "rejected": "rejected"
            }
        },
        "dpo_test": {
            "file_name": "test.json",
            "ranking": True,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "chosen": "chosen",
                "rejected": "rejected"
            }
        }
    }

    os.makedirs("data", exist_ok=True)
    with open("data/dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print("✅ Registered DPO + Self-DPO datasets")

def main():
    print("⬇️ Loading UltraFeedback...")
    ds = load_dataset(
        "argilla/ultrafeedback-binarized-preferences-cleaned",
        split="train"
    )

    # ---- prompt-level split
    prompts = list(set(ds["prompt"]))
    train_prompts, test_prompts = train_test_split(
        prompts, test_size=0.1, random_state=42
    )

    train_prompt_set = set(train_prompts)

    train_data, test_data = [], []

    for ex in ds:
        formatted = format_dpo_data(ex)
        if ex["prompt"] in train_prompt_set:
            train_data.append(formatted)
        else:
            test_data.append(formatted)

    # ---- SELF-DPO CREATION
    print("🔁 Creating Self-DPO dataset...")

    # Pool of candidate rejected responses
    rejection_pool = [ex["chosen"] for ex in train_data]

    self_dpo_data = []
    for ex in train_data:
        rejected = random.choice(rejection_pool)

        # safety check: avoid identical chosen/rejected
        if rejected == ex["chosen"]:
            continue

        self_dpo_data.append({
            "instruction": ex["instruction"],
            "input": "",
            "chosen": ex["chosen"],
            "rejected": rejected
        })

    # ---- save files
    os.makedirs("data", exist_ok=True)
    with open("data/train.json", "w") as f:
        json.dump(train_data, f, indent=2)
    with open("data/self_dpo_train.json", "w") as f:
        json.dump(self_dpo_data, f, indent=2)
    with open("data/test.json", "w") as f:
        json.dump(test_data, f, indent=2)

    print(f"✅ Human DPO train: {len(train_data)}")
    print(f"✅ Self-DPO train: {len(self_dpo_data)}")
    print(f"✅ Test samples: {len(test_data)}")

    update_dataset_info()

if __name__ == "__main__":
    main()
