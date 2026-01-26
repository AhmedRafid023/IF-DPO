import json
import pandas as pd
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", default="output/gemma3-dpo/prediction/predictions.json", help="Path to LLaMA-Factory predictions.json")
    parser.add_argument("--test_file", default="data/test.json", help="Path to original test JSON with chosen/rejected")
    parser.add_argument("--output_csv", default="predictions.csv", help="Output CSV file")
    args = parser.parse_args()

    # Load predictions
    with open(args.pred_file) as f:
        preds = json.load(f)

    # Load gold test data
    with open(args.test_file) as f:
        test_data = json.load(f)

    # Sanity check
    if len(preds) != len(test_data):
        print(f"⚠️ Warning: predictions ({len(preds)}) and test data ({len(test_data)}) lengths differ.")

    rows = []
    for i, ex in enumerate(tqdm(test_data)):
        pred_text = preds[i]["generated_text"] if i < len(preds) else ""
        rows.append({
            "idx": i,
            "prompt": ex["instruction"],
            "input": ex.get("input", ""),
            "chosen": ex["chosen"],
            "rejected": ex["rejected"],
            "generated": pred_text,
            "logp_chosen": "",     # placeholder, can fill later
            "logp_rejected": "",   # placeholder
            "margin": "",          # placeholder
            "pairwise_correct": "" # placeholder
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    print(f"✅ CSV saved to {args.output_csv}")


if __name__ == "__main__":
    main()
