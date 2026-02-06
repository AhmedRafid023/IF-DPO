# import lm_eval
# import json
# import os
# from dotenv import load_dotenv

# load_dotenv()

# # --- Configuration ---
# MODEL_PATH = "google/gemma-3-4b-it"
# # ADAPTER_PATH = ""
# BATCH_SIZE = 16  # Note: Generation tasks take more VRAM than MMLU
# DEVICE = "cuda:0"
# OUTPUT_DIR = "prediction/gsm8k-8shot-gemma3"

# # Standard GSM8K uses 8-shot Chain-of-Thought (CoT)
# # 'gsm8k' = Direct answer (hard for small models)
# # 'gsm8k_cot' = Recommended (shows reasoning steps)
# GSM8K_TASKS = ["gsm8k_cot"]

# def main():
#     print(f"🚀 Base model: {MODEL_PATH}")
#     print(f"🎯 Evaluation mode: 8-SHOT GSM8K (CoT)")

#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     # --- Model args ---
#     # We use 'hf' for Transformers. 'gen_kwargs' helps control the output length.
#     # --- Model args ---
#     args_list = [
#         f"pretrained={MODEL_PATH}",
#         # f"peft={ADAPTER_PATH}",  # 💡 Comment this line out to disable the adapter
#         "trust_remote_code=True",
#         "dtype=bfloat16"
#     ]
#     model_args = ",".join(args_list)

#     results = lm_eval.simple_evaluate(
#         model="hf",
#         model_args=model_args,
#         tasks=GSM8K_TASKS,
#         num_fewshot=8,
#         batch_size=BATCH_SIZE,
#         device=DEVICE,
#         limit=20, # Uncomment to test a small subset first
#     )

#     # --- Save & report ---
#     if results is not None:
#         for task in GSM8K_TASKS:
#             # GSM8K typically reports 'exact_match,none' or 'strict-match'
#             # We look for the exact_match metric
#             metrics = results["results"][task]
#             score = metrics.get("exact_match,none", metrics.get("acc,none", 0)) * 100
#             print(f"\n🏆 {task} Score: {score:.2f}%")

#         output_file = os.path.join(OUTPUT_DIR, "gsm8k_results.json")
#         with open(output_file, "w") as f:
#             json.dump(results, f, indent=2, default=str)

#         print(f"✅ Results saved to: {output_file}")

# if __name__ == "__main__":
#     main()





import torch
import re
import os
import json
import random
import argparse
from tqdm import tqdm
from collections import Counter
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM
)

# --- Configuration Class ---
class EvalConfig:
    def __init__(self):
        # Model & Paths
        self.model_path = "allenai/Llama-3.1-Tulu-3-8B-SFT"
        self.adapter_path = None  
        self.output_dir = "prediction/gsm8k-8shot-Llama-3.1-Tulu"
        
        # Generation Settings
        self.max_new_tokens = 512 # Reduced for speed; usually enough for GSM8K
        self.temperature = 0.0  
        self.batch_size = 8    # 🔥 Increased for speed
        
        # Evaluation Settings
        self.limit = None       # Set to None for full dataset eval
        self.seed = 42

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}

# --- Utilities ---
def extract_numeric_answer(text):
    # Standardize whitespace and look for the typical answer pattern
    text = text.replace(',', '')
    if "The answer is" in text:
        text = text.split("The answer is")[-1]
    
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if matches:
        return matches[-1].strip('.')
    return None

def extract_ground_truth(text):
    match = re.search(r"####\s*([-+]?\d*\.\d+|\d+)", text)
    return match.group(1).replace(',', '') if match else None

# --- Prompt ---
FEW_SHOT_PROMPT = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Q: {question}
A:"""

def main():
    config = EvalConfig()
    
    # Setup
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)

    print(f'🚀 Loading Model: {config.model_path}')
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    
    # 🔥 CRITICAL: Left padding is required for batch generation
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path, 
        device_map='auto', 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    if config.adapter_path:
        from peft import PeftModel
        print(f"🧩 Attaching Adapter: {config.adapter_path}")
        model = PeftModel.from_pretrained(model, config.adapter_path)

    # Load dataset
    split_str = f'test[:{config.limit}]' if config.limit else 'test'
    dataset = load_dataset('gsm8k', "main", split=split_str)
    
    results = []

    # 🔥 Batched Loop
    for i in tqdm(range(0, len(dataset), config.batch_size), desc='Evaluating GSM8K'):
        batch_end = min(i + config.batch_size, len(dataset))
        batch = dataset.select(range(i, batch_end))
        
        # Format prompts
        prompts = [FEW_SHOT_PROMPT.format(question=ex['question']) for ex in batch]
        gold_answers = [extract_ground_truth(ex['answer']) for ex in batch]
        
        inputs = tokenizer(prompts, return_tensors='pt', padding=True).to(model.device)
        input_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                do_sample=config.temperature > 0,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode and extract for the whole batch
        for j, output in enumerate(outputs):
            # Decode only the generated part
            full_gen = tokenizer.decode(output[input_length:], skip_special_tokens=True)
            
            # Clean up the output if it started another Q:
            clean_gen = full_gen.split("Q:")[0].strip()
            
            prediction = extract_numeric_answer(clean_gen)
            
            results.append({
                "question": batch[j]['question'],
                "gold": gold_answers[j],
                "prediction": prediction,
                "correct": prediction == gold_answers[j],
                "generated_text": clean_gen
            })

    # Summary
    correct_count = sum(1 for r in results if r['correct'])
    total_count = len(results)
    acc = correct_count / total_count
    print(f"\n📊 Accuracy: {acc*100:.2f}% ({correct_count}/{total_count})")

    # Save Results
    model_name_clean = config.model_path.split('/')[-1]
    output_path = os.path.join(config.output_dir, f"{model_name_clean}_results_batch.json")
    with open(output_path, 'w') as f:
        json.dump({"config": config.to_dict(), "results": results}, f, indent=4)

    print(f"✅ Results saved to {output_path}")

if __name__ == '__main__':
    main()