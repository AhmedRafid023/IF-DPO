import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

# Load .env file for OPENAI_API_KEY
load_dotenv()

class DPOEvaluator:
    def __init__(self, model_name="gpt-4-turbo"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.judge_model = model_name

    def judge_pair(self, instruction, output_a, output_b):
        """
        Runs a pairwise comparison. 
        Returns 1 if output_a is better, 0 if output_b is better, 0.5 for tie.
        """
        prompt = f"""You are an impartial judge evaluating the quality of two AI responses to the same instruction.
Instruction: {instruction}

Response A: {output_a}
Response B: {output_b}

Which response is better? Consider accuracy, relevance, and helpfulness. 
Output only the letter 'A' or 'B'. If they are equally good, output 'Tie'."""

        try:
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=5
            )
            result = response.choices[0].message.content.strip().upper()
            if 'A' in result and 'B' not in result: return 1.0
            if 'B' in result and 'A' not in result: return 0.0
            return 0.5
        except Exception as e:
            print(f"Error calling OpenAI: {e}")
            return 0.5

    def evaluate_jsonl(self, file_path):
        results = []
        
        # Pre-count lines for progress bar
        with open(file_path, 'r') as f:
            total_lines = sum(1 for _ in f)

        print(f"Found {total_lines} samples. Starting GPT-4 evaluation...")

        with open(file_path, 'r') as f:
            # Wrap the iterator with tqdm for the progress bar
            for line in tqdm(f, total=total_lines, desc="Judging Samples", unit="pair"):
                data = json.loads(line)
                
                # Extract clean instruction if it has special tokens
                instruction = data['prompt'].split("user\n")[-1].split("<end_of_turn>")[0].strip()
                model_out = data['predict']
                ref_out = data.get('label', '')

                # 1. First order: (Model, Ref)
                score_1 = self.judge_pair(instruction, model_out, ref_out)
                
                # 2. Swapped order: (Ref, Model) to avoid position bias
                # 1.0 here means Ref is better, so we do (1 - score) to get Model's score
                score_2 = 1 - self.judge_pair(instruction, ref_out, model_out)
                
                # Average score (0, 0.25, 0.5, 0.75, or 1.0)
                final_win = (score_1 + score_2) / 2
                
                results.append({
                    "instruction": instruction,
                    "win": final_win,
                    "model_len": len(model_out),
                    "ref_len": len(ref_out)
                })
        
        return pd.DataFrame(results)

if __name__ == "__main__":
    # Check if API key exists
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in .env file.")
    else:
        evaluator = DPOEvaluator()
        
        # Path to your predictions
        file_path = "prediction/tulu_base/generated_predictions.jsonl"
        
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
        else:
            df = evaluator.evaluate_jsonl(file_path)
            
            # Calculations
            win_rate = df['win'].mean() * 100
            avg_model_len = df['model_len'].mean()
            avg_ref_len = df['ref_len'].mean()
            len_correlation = df['win'].corr(df['model_len'])
            
            # --- Output Results ---
            print("\n" + "="*30)
            print("DPO EVALUATION RESULTS")
            print("="*30)
            print(f"Win Rate: {win_rate:.2f}%")
            print(f"Avg Model Length: {avg_model_len:.1f}")
            print(f"Avg Reference Length: {avg_ref_len:.1f}")
            print(f"Length Correlation: {len_correlation:.2f}")
            print("-" * 30)
            
            if len_correlation > 0.5:
                print("⚠️  High length bias detected. Your model may be winning primarily due to verbosity.")
            
            # Optional: Save results to CSV
            df.to_csv("alpacaeval2/dpo_eval_detailed_results.csv", index=False)
            print("Detailed results saved to dpo_eval_detailed_results.csv")