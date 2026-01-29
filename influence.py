import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

# =========================================
# 1. CONFIGURATION
# =========================================
class Config:
    # --- Set to False for the full run ---
    DEBUG_MODE = True 

    # --- Path Settings ---
    # Update this to point to your actual file location
    dataset_path = "data/train.json" 
    output_dir = "./influence_scoring_results"

    # --- Model Settings ---
    # Updated to the requested Tulu 3 model
    model_name = "allenai/Llama-3.1-Tulu-3-8B-SFT" 
    
    # --- LoRA Settings ---
    lora_rank = 8
    target_modules = ["q_proj", "v_proj"] 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Compute Settings ---
    dtype = torch.bfloat16 # Use bfloat16 for Llama 3 to save memory
    
    # --- Debug Limits ---
    if DEBUG_MODE:
        debug_train_size = 50
        debug_val_size = 10
    else:
        debug_train_size = None
        debug_val_size = None

# =========================================
# 2. DATA INFLUENCE ENGINE (Manual Implementation)
# =========================================
class DataInfEngine:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.val_grad_avg = None

    def get_flattened_grad(self):
        """Extracts and flattens gradients from trainable (LoRA) parameters."""
        grads = []
        for n, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                grads.append(p.grad.view(-1))
        if not grads:
            return None
        return torch.cat(grads)

    def compute_val_grad_avg(self, val_loader):
        """Computes the average gradient of the validation set (The 'Compass')."""
        print(" [DataInf] Computing Validation Gradient Average...")
        self.model.eval()
        self.model.zero_grad()
        total_grad = None
        count = 0
        
        for batch in tqdm(val_loader, desc="Val Grads"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            flat_grad = self.get_flattened_grad()
            if flat_grad is not None:
                if total_grad is None:
                    total_grad = torch.zeros_like(flat_grad)
                total_grad += flat_grad
                count += 1
            
            self.model.zero_grad()
            
            # [Optional] Clear cache to save VRAM on big models
            # torch.cuda.empty_cache() 
            
        if count == 0:
            raise ValueError("No gradients computed! Check if LoRA is active.")
            
        self.val_grad_avg = total_grad / count
        return self.val_grad_avg

    def compute_datainf_scores(self, train_loader, lambda_const_param=1000):
        """Computes influence scores for training samples."""
        if self.val_grad_avg is None: 
            raise ValueError("Run compute_val_grad_avg first!")
            
        print(" [DataInf] Computing Scores...")
        self.model.eval()
        scores = {}
        
        # --- Pass 1: Estimate Lambda (Damping Factor) ---
        squared_norms = []
        for i, batch in enumerate(tqdm(train_loader, desc="Pass 1 (Lambda)")):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.model.zero_grad()
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            g = self.get_flattened_grad()
            if g is not None:
                squared_norms.append(torch.sum(g ** 2).item())
            self.model.zero_grad()
            
        if not squared_norms:
             raise ValueError("No gradients found during Pass 1.")

        avg_squared_norm = np.mean(squared_norms)
        lambda_const = avg_squared_norm / lambda_const_param
        
        # --- Pass 2: Compute Influence Scores ---
        v = self.val_grad_avg
        for i, batch in enumerate(tqdm(train_loader, desc="Pass 2 (Scoring)")):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.model.zero_grad()
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            g = self.get_flattened_grad()
            
            # Influence Formula: - (Gradient * HVP)
            # Using First-Order approximation for speed/memory on large models
            g_dot_v = torch.dot(v, g)
            g_norm_sq = torch.sum(g ** 2)
            
            # Approximate Inverse Hessian Vector Product (Woodbury Identity simplified)
            c = g_dot_v / (lambda_const + g_norm_sq)
            hvp = (v - c * g) / lambda_const
            
            # Score = - (g * hvp)
            # Negative because influence measures effect on LOSS. 
            # We want to decrease val loss, so we look for highly negative influence?
            # Convention: High positive score = "Helpful for Val set"
            scores[i] = torch.dot(g, hvp).item() 
            
            self.model.zero_grad()
            
        return scores

# =========================================
# 3. DATA LOADING
# =========================================
def get_data():
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading data from {Config.dataset_path}...")
    
    # Load raw JSON
    ds = load_dataset("json", data_files=Config.dataset_path) 
    
    # Handle split structure
    if 'train' in ds:
        ds_train = ds['train']
    else:
        ds_train = ds['train'] # Usually 'train' is default split for generic files
        
    # Create Validation Split (The "Compass")
    ds = ds_train.train_test_split(test_size=0.1, seed=42)
    
    train_raw = ds['train']
    val_raw = ds['test']
    
    # Debug limits
    if Config.DEBUG_MODE:
        print(f"⚠️ DEBUG MODE: Truncating data to {Config.debug_train_size} samples.")
        train_raw = train_raw.select(range(min(len(train_raw), Config.debug_train_size)))
        val_raw = val_raw.select(range(min(len(val_raw), Config.debug_val_size)))

    def format_dpo(ex):
        # Formatting for SFT / Influence
        instr = ex.get('instruction', '')
        inp = ex.get('input', '')
        chosen = ex.get('chosen', '')
        
        # Simple Chat Template Format
        text = f"<|user|>\n{instr} {inp}\n<|assistant|>\n{chosen}" + tokenizer.eos_token
        
        out = tokenizer(text, truncation=True, max_length=512, padding="max_length")
        out["labels"] = out["input_ids"].copy()
        return out

    train_ds = train_raw.map(format_dpo)
    val_ds = val_raw.map(format_dpo)
    
    cols = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format("torch", columns=cols)
    val_ds.set_format("torch", columns=cols)
    
    return train_ds, val_ds, tokenizer, train_raw

# =========================================
# 4. MAIN EXECUTION
# =========================================
def main():
    os.makedirs(Config.output_dir, exist_ok=True)
    
    # 1. Load Data
    train_ds, val_ds, tokenizer, train_raw = get_data()
    collate = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    # 2. Load Model
    print(f"Loading Model: {Config.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        Config.model_name, 
        torch_dtype=Config.dtype,
        device_map="auto" # Auto-distribute to GPU
    )
    
    # 3. Attach LoRA
    # Influence functions are too expensive for full fine-tuning
    # We calculate influence ONLY on the LoRA adapters
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        r=Config.lora_rank, 
        target_modules=Config.target_modules, 
        lora_alpha=16
    )
    model = get_peft_model(model, peft_config)
    print(f"Trainable Parameters (Influence Target): {model.print_trainable_parameters()}")
    
    # 4. Prepare Loaders (Batch Size 1 is safest for gradients)
    train_loader_single = DataLoader(train_ds, batch_size=1, collate_fn=collate, shuffle=False)
    val_loader_single = DataLoader(val_ds, batch_size=1, collate_fn=collate, shuffle=False)
    
    # 5. Run Influence Engine
    print("\n=== RUNNING DATAINF ESTIMATION ===")
    datainf_engine = DataInfEngine(model, Config.device)
    
    # Step A: Compute Validation Gradient (Target Direction)
    datainf_engine.compute_val_grad_avg(val_loader_single)
    
    # Step B: Score Training Data
    scores_datainf = datainf_engine.compute_datainf_scores(train_loader_single)

    # 6. Save Results
    # Handle potential missing keys if using different dataset format
    try:
        rejected_col = train_raw["rejected"]
    except KeyError:
        rejected_col = [""] * len(train_raw)

    df = pd.DataFrame({
        "instruction": train_raw["instruction"],
        "chosen": train_raw["chosen"],
        "rejected": rejected_col,
        "score_datainf": [scores_datainf.get(i, 0.0) for i in range(len(train_raw))]
    })
    
    csv_path = os.path.join(Config.output_dir, "influence_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nDone! Saved to {csv_path}")

    # 7. Show Top Results
    print("\n--- Most Helpful Samples (Highest Score) ---")
    top_helpful = df.sort_values("score_datainf", ascending=False).head(3)
    for idx, row in top_helpful.iterrows():
        print(f"[Score: {row['score_datainf']:.4f}] {row['instruction'][:100]}...")

    print("\n--- Least Helpful / Harmful Samples (Lowest Score) ---")
    top_harmful = df.sort_values("score_datainf", ascending=True).head(3)
    for idx, row in top_harmful.iterrows():
        print(f"[Score: {row['score_datainf']:.4f}] {row['instruction'][:100]}...")

if __name__ == "__main__":
    main()