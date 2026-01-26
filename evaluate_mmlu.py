import lm_eval
import torch
import json
import os
from dotenv import load_dotenv  # <--- ADD THIS

load_dotenv()  # <--- ADD THIS (Loads HF_TOKEN from .env)
# --- Configuration ---
MODEL_PATH = "google/gemma-3-4b-it"
ADAPTER_PATH = "output/gemma3-dpo"  # Path to your trained LoRA adapter
BATCH_SIZE = 16                      # Lower to 1 or 2 if you get OOM
DEVICE = "cuda:0"
OUTPUT_DIR = "prediction/mmlu-dpo"

def main():
    print(f"🚀 Loading model: {MODEL_PATH}")
    print(f"📂 Loading adapter: {ADAPTER_PATH}")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # arguments for the Hugging Face model loader
    # trusting remote code is often needed for newer models
    model_args = f"pretrained={MODEL_PATH},peft={ADAPTER_PATH},trust_remote_code=True,dtype=bfloat16"

    # Run MMLU Evaluation
    # 'mmlu' runs all 57 subjects. This takes time!
    # To test quickly, change tasks=["mmlu_abstract_algebra"]
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=["mmlu"],      
        num_fewshot=5,       # Standard MMLU is 5-shot
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )

    # --- Process and Save Results ---
    if results is not None:
        # Extract the final aggregate score
        final_score = results['results'].get('mmlu', {}).get('acc,none', 0.0) * 100
        
        print("\n" + "="*50)
        print(f"🏆 Final MMLU Score: {final_score:.2f}%")
        print("="*50)

        # Save full details to JSON
        output_file = os.path.join(OUTPUT_DIR, "mmlu_scores.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ Detailed results saved to: {output_file}")
        
        # Optional: Print breakdown of top categories
        # MMLU has many subtasks, this gives a peek at the data
        print("\n--- Category Breakdown (Sample) ---")
        for task, metrics in list(results['results'].items())[:5]:
            acc = metrics.get('acc,none', 0) * 100
            print(f"  - {task}: {acc:.2f}%")

if __name__ == "__main__":
    main()