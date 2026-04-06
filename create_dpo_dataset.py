import os
import json
import random
from datasets import load_dataset

# Configuration
OUTPUT_DIR   = "data"
OUTPUT_JSON  = "self_dpo_train.json"
DATASET_INFO = "data/dataset_info.json"
SEED         = 42

def main():
    # Set seed for reproducible shuffling
    random.seed(SEED)
    
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Downloading/Loading the UltraFeedback dataset...")
    # 'train_prefs' is the split containing preference pairs in this dataset
    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")

    print("Extracting prompts and winning samples...")
    data_items = []
    winning_samples = []

    for row in dataset:
        prompt = row["prompt"]
        
        # In ultrafeedback_binarized, 'chosen' is a list of conversation messages.
        # We extract the last message's content, which is the assistant's winning response.
        chosen_content = row["chosen"][-1]["content"]
        
        data_items.append({
            "instruction": prompt,
            "input": "",
            "chosen": chosen_content
        })
        winning_samples.append(chosen_content)

    print("Shuffling winning samples to create unique losing samples...")
    # Create a copy of the winning samples and shuffle them
    losing_samples = winning_samples.copy()
    random.shuffle(losing_samples)

    # Prevent a sample from acting as a losing sample for its own prompt (A == A)
    for i in range(len(losing_samples)):
        if losing_samples[i] == winning_samples[i]:
            # Swap with the adjacent element
            swap_idx = (i + 1) % len(losing_samples)
            losing_samples[i], losing_samples[swap_idx] = losing_samples[swap_idx], losing_samples[i]

    print("Constructing the final Self-DPO dataset...")
    dpo_dataset = []
    for i, item in enumerate(data_items):
        item["rejected"] = losing_samples[i]
        dpo_dataset.append(item)

    # Save the JSON file
    output_dataset_path = os.path.join(OUTPUT_DIR, OUTPUT_JSON)
    print(f"Saving Self-DPO dataset to {output_dataset_path}...")
    with open(output_dataset_path, "w", encoding="utf-8") as f:
        json.dump(dpo_dataset, f, indent=2, ensure_ascii=False)

    # Update or create dataset_info.json for LLaMA-Factory compatibility
    print(f"Updating {DATASET_INFO}...")
    dataset_info = {}
    if os.path.exists(DATASET_INFO):
        try:
            with open(DATASET_INFO, "r", encoding="utf-8") as f:
                dataset_info = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: {DATASET_INFO} is not valid JSON. Overwriting.")

    # Register the new dataset
    dataset_info["self_dpo_ultrafeedback"] = {
        "file_name": OUTPUT_JSON,
        "ranking": True,
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "chosen": "chosen",
            "rejected": "rejected"
        }
    }

    with open(DATASET_INFO, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)

    print(f"Done! Created {len(dpo_dataset)} Self-DPO pairs.")

if __name__ == "__main__":
    main()