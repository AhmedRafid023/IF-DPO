import json
import os

def calculate_accuracy(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        return

    with open(file_path, 'r') as f:
        data = json.load(f)

    config = data.get("config", {})
    results = data.get("results", [])
    
    total = len(results)
    if total == 0:
        print("⚠️ No results found in the file.")
        return

    # Count correct answers
    correct = sum(1 for r in results if r.get("correct") is True)
    accuracy = (correct / total) * 100

    # Print Summary
    print("-" * 50)
    print(f"📊 GSM8K EVALUATION SUMMARY")
    print("-" * 50)
    print(f"🤖 Model:     {config.get('model_path')}")
    print(f"🧩 Adapter:   {config.get('adapter_path', 'None')}")
    print(f"📝 Samples:   {total}")
    print(f"🌡️ Temp:      {config.get('temperature')}")
    print("-" * 50)
    print(f"✅ Correct:   {correct}")
    print(f"❌ Incorrect: {total - correct}")
    print(f"🏆 Accuracy:  {accuracy:.2f}%")
    print("-" * 50)

if __name__ == "__main__":
    # Update this path if you move the script
    RESULT_FILE = "prediction/gsm8k-8shot-Llama-3.1-Tulu/Llama-3.1-Tulu-3-8B-SFT_results_batch.json"
    calculate_accuracy(RESULT_FILE)