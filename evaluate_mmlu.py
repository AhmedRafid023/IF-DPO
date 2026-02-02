import lm_eval
import json
import os
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
MODEL_PATH = "allenai/Llama-3.1-Tulu-3-8B-SFT"
# ADAPTER_PATH = "output/tulu-dpo-lora"   # set to None if no LoRA
BATCH_SIZE = 4
DEVICE = "cuda:0"
OUTPUT_DIR = "prediction/mmlu-5shot-10task"

# 🔟 Mini-MMLU task subset (diverse + commonly used)
MMLU_10_TASKS = [
    "mmlu_college_computer_science",
    "mmlu_high_school_mathematics",
    "mmlu_high_school_physics",
    "mmlu_college_chemistry",
    "mmlu_electrical_engineering",
    "mmlu_professional_law",
    "mmlu_professional_medicine",
    "mmlu_econometrics",
    "mmlu_philosophy",
    "mmlu_world_religions",
]

def main():
    print(f"🚀 Base model: {MODEL_PATH}")
    print("🎯 Evaluation mode: 5-SHOT Mini-MMLU (10 tasks)")
    # if ADAPTER_PATH:
    #     print(f"🧩 LoRA adapter: {ADAPTER_PATH}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Model args (minimal LoRA switch) ---
    model_args = f"pretrained={MODEL_PATH},trust_remote_code=True,dtype=bfloat16"
    # if ADAPTER_PATH:
    #     model_args += f",peft={ADAPTER_PATH}"

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=MMLU_10_TASKS,
        num_fewshot=5,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        limit=50,   # 🔥 remove or increase (100) for stronger signal
    )

    # --- Save & report ---
    if results is not None:
        scores = []
        for task in MMLU_10_TASKS:
            acc = results["results"][task]["acc,none"] * 100
            scores.append(acc)
            print(f"{task}: {acc:.2f}%")

        avg_score = sum(scores) / len(scores)

        print("\n" + "=" * 60)
        print(f"🏆 Average 5-SHOT Mini-MMLU (10 tasks): {avg_score:.2f}%")
        print("=" * 60)

        output_file = os.path.join(OUTPUT_DIR, "mmlu_5shot_10task.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"✅ Results saved to: {output_file}")

if __name__ == "__main__":
    main()
