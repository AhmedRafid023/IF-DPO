import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType


# ── Config — edit before running ──────────────────────────────────────────────
class Config:
    # Model
    MODEL_NAME   = "allenai/Llama-3.1-Tulu-3-8B-SFT"

    # UltraFeedback splits
    # train_sft split  → candidates to be scored
    # test_sft split   → validation compass (what we want the model to do well on)
    HF_DATASET   = "HuggingFaceH4/ultrafeedback_binarized"
    TRAIN_SPLIT  = "train_prefs"
    VAL_SPLIT    = "test_prefs"

    # LoRA — influence is computed only over LoRA params (full model is too expensive)
    LORA_RANK    = 16
    LORA_ALPHA   = 32
    LORA_TARGETS = ["q_proj", "v_proj"]


    # Output (CHANGED TO JSONL)
    OUTPUT_DIR   = "influence_scoring_results"
    OUTPUT_JSONL = "influence_comparison.jsonl"

    # Compute
    DTYPE        = torch.bfloat16
    MAX_LENGTH   = 512
    LAMBDA_PARAM = 1000          # damping factor denominator

    # Set to an int (e.g. 200) for a quick smoke test, None for full dataset
    DEBUG_TRAIN  = None
    DEBUG_VAL    = None
# ──────────────────────────────────────────────────────────────────────────────


# ── DataInf Engine ─────────────────────────────────────────────────────────────

class DataInfEngine:
    def __init__(self, model, device):
        self.model        = model
        self.device       = device
        self.val_grad_avg = None

    def _flat_grad(self):
        grads = [p.grad.view(-1) for _, p in self.model.named_parameters()
                 if p.requires_grad and p.grad is not None]
        return torch.cat(grads) if grads else None

    def compute_val_grad_avg(self, val_loader):
        """Average gradient over the validation set — the 'compass' direction."""
        print("\n[DataInf] Computing validation gradient average...")
        self.model.eval()
        self.model.zero_grad()
        total_grad = None
        count      = 0

        for batch in tqdm(val_loader, desc="Val grads"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            loss  = self.model(**batch).loss
            loss.backward()

            g = self._flat_grad()
            if g is not None:
                total_grad = g.clone() if total_grad is None else total_grad + g
                count += 1
            self.model.zero_grad()

        if count == 0:
            raise ValueError("No gradients computed — check LoRA is active.")

        self.val_grad_avg = total_grad / count
        print(f"   Validation gradient computed over {count} batches.")
        return self.val_grad_avg

    def compute_scores(self, train_loader):
        """
        Two-pass DataInf scoring.

        Pass 1 — estimate lambda (damping factor) from average squared gradient norm.
        Pass 2 — for each training sample compute:
                    score = g · H⁻¹v
                  using the first-order Woodbury approximation:
                    H⁻¹v ≈ (v - c·g) / λ,  where c = (g·v) / (λ + ‖g‖²)
        Higher score = more helpful for the validation set.
        """
        if self.val_grad_avg is None:
            raise ValueError("Run compute_val_grad_avg first.")

        print("\n[DataInf] Pass 1 — estimating lambda...")
        self.model.eval()
        squared_norms = []

        for batch in tqdm(train_loader, desc="Pass 1 (lambda)"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.model.zero_grad()
            self.model(**batch).loss.backward()
            g = self._flat_grad()
            if g is not None:
                squared_norms.append(torch.sum(g ** 2).item())
            self.model.zero_grad()

        if not squared_norms:
            raise ValueError("No gradients in Pass 1.")

        lam = np.mean(squared_norms) / Config.LAMBDA_PARAM
        print(f"   Lambda = {lam:.6f}")

        print("\n[DataInf] Pass 2 — computing influence scores...")
        v      = self.val_grad_avg
        scores = {}

        for i, batch in enumerate(tqdm(train_loader, desc="Pass 2 (scores)")):
            batch = {k: v_.to(self.device) for k, v_ in batch.items()}
            self.model.zero_grad()
            self.model(**batch).loss.backward()
            g = self._flat_grad()
            if g is None:
                scores[i] = 0.0
                self.model.zero_grad()
                continue

            g_dot_v  = torch.dot(v, g)
            g_norm_sq = torch.sum(g ** 2)
            c        = g_dot_v / (lam + g_norm_sq)
            hvp      = (v - c * g) / lam
            scores[i] = torch.dot(g, hvp).item()

            self.model.zero_grad()

        return scores


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(tokenizer):
    print(f"\nLoading {Config.HF_DATASET}...")
    ds_train = load_dataset(Config.HF_DATASET, split=Config.TRAIN_SPLIT)
    ds_val   = load_dataset(Config.HF_DATASET, split=Config.VAL_SPLIT)

    if Config.DEBUG_TRAIN:
        ds_train = ds_train.select(range(min(len(ds_train), Config.DEBUG_TRAIN)))
        print(f"⚠️  Debug: using {len(ds_train)} training examples.")
    if Config.DEBUG_VAL:
        ds_val = ds_val.select(range(min(len(ds_val), Config.DEBUG_VAL)))
        print(f"⚠️  Debug: using {len(ds_val)} validation examples.")

    print(f"   Train: {len(ds_train)} | Val: {len(ds_val)}")

    def format_example(ex):
        """
        train_prefs / test_prefs splits store conversations in 'chosen' as a list
        of {role, content} dicts — same structure as the sft splits.
        We use only the 'chosen' column (the higher-quality response) for gradient
        computation. The influence score tells us how helpful each chosen response
        is for the validation set — we are NOT training on rejected here.
        """
        messages = ex.get("chosen", [])
        user_msg = ""
        asst_msg = ""
        for msg in messages:
            if msg["role"] == "user":
                user_msg = msg["content"]
            elif msg["role"] == "assistant":
                asst_msg = msg["content"]

        # Skip malformed examples silently — collator will handle empty tensors
        if not user_msg or not asst_msg:
            user_msg = user_msg or "empty"
            asst_msg = asst_msg or "empty"

        text = (
            f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n{asst_msg}<|eot_id|>"
        )
        out           = tokenizer(text, truncation=True, max_length=Config.MAX_LENGTH,
                                  padding="max_length")
        out["labels"] = out["input_ids"].copy()
        return out

    cols = ["input_ids", "attention_mask", "labels"]

    train_tokenized = ds_train.map(format_example, desc="Tokenising train")
    train_tokenized.set_format("torch", columns=cols)

    val_tokenized = ds_val.map(format_example, desc="Tokenising val")
    val_tokenized.set_format("torch", columns=cols)

    return train_tokenized, val_tokenized, ds_train


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Data
    train_ds, val_ds, train_raw = load_data(tokenizer)
    collate = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    train_loader = DataLoader(train_ds, batch_size=1, collate_fn=collate, shuffle=False)
    val_loader   = DataLoader(val_ds,   batch_size=1, collate_fn=collate, shuffle=False)

    # Model + LoRA
    print(f"\nLoading model: {Config.MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        torch_dtype=Config.DTYPE,
        device_map="auto",
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=Config.LORA_RANK,
        lora_alpha=Config.LORA_ALPHA,
        target_modules=Config.LORA_TARGETS,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    device = next(model.parameters()).device

    # Run DataInf
    engine = DataInfEngine(model, device)
    engine.compute_val_grad_avg(val_loader)
    scores = engine.compute_scores(train_loader)

    # Extract chosen text for each training example
    def get_chosen_text(ex):
        for msg in ex.get("chosen", []):
            if msg["role"] == "assistant":
                return msg["content"]
        return ""

    def get_prompt_text(ex):
        for msg in ex.get("chosen", []):
            if msg["role"] == "user":
                return msg["content"]
        return ""

    # Save as JSONL
    df = pd.DataFrame({
        "instruction": [get_prompt_text(ex) for ex in train_raw],
        "input":       [""] * len(train_raw),
        "chosen":      [get_chosen_text(ex) for ex in train_raw],
        "rejected":    [get_chosen_text(ex) for ex in train_raw],   # placeholder, overwritten by DPO scripts
        "score_datainf": [scores.get(i, 0.0) for i in range(len(train_raw))],
    })

    # CHANGED: Use JSONL export instead of CSV
    jsonl_path = os.path.join(Config.OUTPUT_DIR, Config.OUTPUT_JSONL)
    df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)

    print(f"\n{'='*50}")
    print(f"  Influence scoring complete")
    print(f"{'='*50}")
    print(f"  Examples scored : {len(df)}")
    print(f"  JSONL saved to  : {jsonl_path}")
    print(f"\n  Top 3 most helpful samples:")
    for _, row in df.nlargest(3, "score_datainf").iterrows():
        print(f"    [{row['score_datainf']:.4f}] {row['instruction'][:80]}...")
    print(f"\n  Top 3 least helpful samples:")
    for _, row in df.nsmallest(3, "score_datainf").iterrows():
        print(f"    [{row['score_datainf']:.4f}] {row['instruction'][:80]}...")


if __name__ == "__main__":
    main()