import json
import re

def normalize_number(x):
    """Convert string number to canonical form"""
    try:
        if "." in x:
            x = str(int(float(x)))
        return x
    except:
        return None

def extract_gold_answer(text):
    """
    GSM8K gold answers ALWAYS use:
    #### <number>
    """
    if not text:
        return None
    if "####" not in text:
        return None
    ans = text.split("####")[-1].strip()
    return normalize_number(ans)

def extract_pred_answer(text):
    """
    Robust extraction for messy model outputs
    """
    if not text:
        return None

    # 1️⃣ If model accidentally outputs ####
    if "####" in text:
        return normalize_number(text.split("####")[-1].strip())

    # 2️⃣ Remove code blocks (```...```)
    text = re.sub(r"```.*?```", "", text, flags=re.S)

    # 3️⃣ Remove common noise tokens
    text = re.sub(r"(# Output:.*|def .*|print\(.*?\))", "", text)

    # 4️⃣ Extract all standalone numbers (integers or floats)
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)

    if not numbers:
        return None

    # GSM8K convention → final numeric answer is LAST meaningful number
    return normalize_number(numbers[-1])

def score_file(path):
    correct = 0
    total = 0

    print(f"📊 Scoring GSM8K file: {path}")

    with open(path, "r") as f:
        for line in f:
            ex = json.loads(line)

            pred = ex.get("predict", "")
            gold = ex.get("label", "")

            gold_ans = extract_gold_answer(gold)
            pred_ans = extract_pred_answer(pred)

            if gold_ans is None:
                continue

            total += 1
            if pred_ans == gold_ans:
                correct += 1

    if total == 0:
        print("❌ No valid GSM8K samples found.")
        return

    acc = 100 * correct / total
    print("\n✅ GSM8K Results")
    print(f"   Total samples: {total}")
    print(f"   Correct:       {correct}")
    print(f"   Accuracy:      {acc:.2f}%")

if __name__ == "__main__":
    score_file("prediction/gsm8k-tulu-base/generated_predictions.jsonl")
