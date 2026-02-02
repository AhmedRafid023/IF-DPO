import lm_eval
import json
import os
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
MODEL_PATH = "allenai/Llama-3.1-Tulu-3-8B-SFT"
# ADAPTER_PATH = "output/tulu-dpo-lora"   # set to None if no LoRA
BATCH_SIZE = 8                          # TruthfulQA is light
DEVICE = "cuda:0"
OUTPUT_DIR = "prediction/truthfulqa_tulu"

def main():
    print(f"🚀 Base model: {MODEL_PATH}")
    print("🎯 Evaluation: TruthfulQA (MC)")
    # if ADAPTER_PATH:
    #     print(f"🧩 LoRA adapter: {ADAPTER_PATH}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Model args ---
    model_args = f"pretrained={MODEL_PATH},trust_remote_code=True,dtype=bfloat16"
    # if ADAPTER_PATH:
    #     model_args += f",peft={ADAPTER_PATH}"

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=["truthfulqa_mc2"],  # ✅ lm-eval official task
        num_fewshot=0,            # 🔥 standard for TruthfulQA
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )

    # --- Extract & save ---
    metrics = results["results"]["truthfulqa_mc2"]

    score = metrics.get("acc,none", 0.0) * 100

    print("\n" + "=" * 60)
    print(f"🏆 TruthfulQA MC2 Score: {score:.2f}%")
    print("=" * 60)

    output_file = os.path.join(OUTPUT_DIR, "truthfulqa_mc2.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"✅ Results saved to: {output_file}")

if __name__ == "__main__":
    main()
