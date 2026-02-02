import json
import os
from dotenv import load_dotenv
from alpaca_eval import evaluate

# =========================
# Configuration
# =========================
INPUT_FILE = "generated_prediction.jsonl"
ALPACA_INPUT_FILE = "Alpaca/alpacaeval_input.jsonl"
OUTPUT_DIR = "prediction/alpacaeval_results_tulu_base"
ANNOTATOR = "alpaca_eval_gpt4"

# =========================
# Helpers
# =========================
def extract_instruction(prompt: str) -> str:
    """
    Extracts the user instruction from a chat-style prompt.
    """
    if "<start_of_turn>user" in prompt:
        prompt = prompt.split("<start_of_turn>user", 1)[1]
    if "<end_of_turn>" in prompt:
        prompt = prompt.split("<end_of_turn>", 1)[0]
    return prompt.strip()

# =========================
# Step 1: Convert JSONL
# =========================
def convert_to_alpacaeval():
    print("🔄 Converting predictions to AlpacaEval format...")

    with open(INPUT_FILE, "r") as fin, open(ALPACA_INPUT_FILE, "w") as fout:
        for line in fin:
            ex = json.loads(line)

            instruction = extract_instruction(ex["prompt"])
            output = ex["predict"].strip()

            fout.write(json.dumps({
                "instruction": instruction,
                "output": output
            }) + "\n")

    print(f"✅ Conversion done: {ALPACA_INPUT_FILE}")

# =========================
# Step 2: Run AlpacaEval 2
# =========================
def run_alpacaeval():
    print("\n🚀 Running AlpacaEval 2 (GPT-4 judge)")
    print(f"📂 Inputs: {ALPACA_INPUT_FILE}")
    print(f"📁 Output dir: {OUTPUT_DIR}")

    evaluate(
        model_outputs=ALPACA_INPUT_FILE,
        annotators_config=ANNOTATOR,
        output_path=OUTPUT_DIR,
        is_overwrite=True,
    )

    print("\n🏁 AlpacaEval 2 completed successfully")

# =========================
# Main
# =========================
def main():
    load_dotenv()
    assert os.environ.get("OPENAI_API_KEY"), "❌ OPENAI_API_KEY not found in environment"

    convert_to_alpacaeval()
    run_alpacaeval()

    print(f"\n📊 Final results available in: {OUTPUT_DIR}")
    print("👉 Check leaderboard.csv for the main score")

if __name__ == "__main__":
    main()
