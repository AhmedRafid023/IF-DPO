import torch
import re
import os
import json
import random
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


# ── Config — edit before running ──────────────────────────────────────────────
class EvalConfig:
    MODEL_PATH     = "allenai/Llama-3.1-Tulu-3-8B-SFT"
    ADAPTER_PATH   = None                              # set to adapter path if using DPO LoRA
    OUTPUT_DIR     = "prediction/gsm8k"
    MAX_NEW_TOKENS = 512
    TEMPERATURE    = 0.0
    BATCH_SIZE     = 8
    LIMIT          = None                              # set to int (e.g. 100) for quick test
    SEED           = 42
# ──────────────────────────────────────────────────────────────────────────────


FEW_SHOT_PROMPT = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Q: {question}
A:"""


def extract_numeric_answer(text):
    text = text.replace(',', '')
    if "The answer is" in text:
        text = text.split("The answer is")[-1]
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    return matches[-1].strip('.') if matches else None


def extract_ground_truth(text):
    match = re.search(r"####\s*([-+]?\d*\.\d+|\d+)", text)
    return match.group(1).replace(',', '') if match else None


def main():
    random.seed(EvalConfig.SEED)
    torch.manual_seed(EvalConfig.SEED)
    os.makedirs(EvalConfig.OUTPUT_DIR, exist_ok=True)

    print(f"🚀 Loading model: {EvalConfig.MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(EvalConfig.MODEL_PATH)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        EvalConfig.MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    if EvalConfig.ADAPTER_PATH:
        from peft import PeftModel
        print(f"🧩 Attaching adapter: {EvalConfig.ADAPTER_PATH}")
        model = PeftModel.from_pretrained(model, EvalConfig.ADAPTER_PATH)

    model.eval()

    split = f"test[:{EvalConfig.LIMIT}]" if EvalConfig.LIMIT else "test"
    dataset = load_dataset("gsm8k", "main", split=split)
    print(f"📂 Loaded {len(dataset)} examples.")

    results = []

    for i in tqdm(range(0, len(dataset), EvalConfig.BATCH_SIZE), desc="Evaluating GSM8K"):
        batch      = dataset.select(range(i, min(i + EvalConfig.BATCH_SIZE, len(dataset))))
        prompts    = [FEW_SHOT_PROMPT.format(question=ex["question"]) for ex in batch]
        gold       = [extract_ground_truth(ex["answer"]) for ex in batch]

        inputs       = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        input_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=EvalConfig.MAX_NEW_TOKENS,
                temperature=EvalConfig.TEMPERATURE,
                do_sample=EvalConfig.TEMPERATURE > 0,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j, output in enumerate(outputs):
            generated = tokenizer.decode(output[input_length:], skip_special_tokens=True)
            clean     = generated.split("Q:")[0].strip()
            pred      = extract_numeric_answer(clean)
            results.append({
                "question":       batch[j]["question"],
                "gold":           gold[j],
                "prediction":     pred,
                "correct":        pred == gold[j],
                "generated_text": clean,
            })

    correct = sum(r["correct"] for r in results)
    total   = len(results)
    acc     = correct / total

    print(f"\n{'='*50}")
    print(f"  GSM8K Results")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc*100:.2f}% ({correct}/{total})")
    print(f"{'='*50}")

    model_tag   = EvalConfig.MODEL_PATH.split("/")[-1]
    output_path = os.path.join(EvalConfig.OUTPUT_DIR, f"{model_tag}_gsm8k.json")
    with open(output_path, "w") as f:
        json.dump({"accuracy": acc, "correct": correct, "total": total, "results": results}, f, indent=2)
    print(f"✅ Results saved to {output_path}")


if __name__ == "__main__":
    main()