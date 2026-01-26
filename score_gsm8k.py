import json
import re
import sys

def extract_answer(text):
    """
    Extracts the last number from the text.
    Standard GSM8K answers end with '#### <number>'
    """
    if not text:
        return None
    # Look for the standard '####' marker first
    if "####" in text:
        return text.split("####")[-1].strip()
    
    # Fallback: Find the very last number in the text (integer or float)
    # This regex handles commas (1,000) and decimals (1.5)
    numbers = re.findall(r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?', text)
    if numbers:
        return numbers[-1].replace(',', '')
    return None

def main(file_path):
    print(f"📊 Scoring file: {file_path}")
    
    correct = 0
    total = 0
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                
                # Get model output and ground truth
                # LLaMA Factory saves output in 'predict' and ground truth in 'label' or 'response'
                prediction = data.get('predict', '')
                ground_truth = data.get('label', '')
                
                # Extract the numbers
                pred_num = extract_answer(prediction)
                ref_num = extract_answer(ground_truth)
                
                if ref_num is None:
                    continue # Skip bad data
                    
                total += 1
                
                # Check for exact match (numerical equality)
                try:
                    # Convert to float to handle 42 vs 42.0
                    if pred_num and float(pred_num) == float(ref_num):
                        correct += 1
                except ValueError:
                    # If conversion fails, check string match
                    if pred_num == ref_num:
                        correct += 1

        if total == 0:
            print("❌ No valid samples found.")
            return

        accuracy = (correct / total) * 100
        print(f"\n✅ Final Results:")
        print(f"   Total Samples: {total}")
        print(f"   Correct:       {correct}")
        print(f"   Accuracy:      {accuracy:.2f}%")
        
    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}")

if __name__ == "__main__":
    # Point this to your generated file
    file_path = "prediction/gsm8k-dpo/generated_predictions.jsonl"
    main(file_path)