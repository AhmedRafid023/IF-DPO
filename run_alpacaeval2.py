"""
lc_alpaca_eval.py — Length-Controlled AlpacaEval-2 Benchmark

Process:
    1. Load the 805 AlpacaEval prompts
    2. Generate responses from your model
    3. Use GPT-4 Turbo to judge: model vs reference (GPT-4 Turbo baseline)
    4. Fit a GLM on (win, length_diff) to learn how much length drives wins
    5. Re-predict at length_diff=0 → length-controlled win rate

Usage:
    python lc_alpaca_eval.py

    Edit RunConfig below to change model, adapter, output dir, etc.
    OPENAI_API_KEY must be set in your .env file.

Requirements:
    pip install transformers peft datasets openai scikit-learn numpy tqdm
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm


# ── Config ─────────────────────────────────────────────────────────────────────
ALPACA_EVAL_DATASET  = "tatsu-lab/alpaca_eval"
ALPACA_EVAL_SPLIT    = "eval"
REFERENCE_MODEL_NAME = "gpt4_turbo"          # the baseline every model is compared against
MAX_NEW_TOKENS       = 2048
JUDGE_MAX_TOKENS     = 1024
TEMPERATURE          = 0.0                   # greedy for reproducibility
JUDGE_TEMPERATURE    = 0.0
GLM_FEATURES         = ["length_diff",       # model chars - reference chars
                         "length_diff_sq",   # quadratic term to capture non-linearity
                         "log_model_len",    # absolute length of model response
                         "log_ref_len"]      # absolute length of reference response


# ── Run config — edit these before running ────────────────────────────────────
class RunConfig:
    MODEL_NAME_OR_PATH = "allenai/Llama-3.1-Tulu-3-8B-SFT"
    ADAPTER_PATH       = None
    OUTPUT_DIR         = "eval_results/lc_alpaca"
    OPENAI_MODEL       = "gpt-4-turbo"
    BATCH_SIZE         = 64
    MAX_SAMPLES        = None    # set to an int (e.g. 50) for a quick smoke test
    SKIP_GENERATION    = False   # set True to reuse saved model_outputs.json
# ─────────────────────────────────────────────────────────────────────────────

# AlpacaEval 2 judge prompt — faithful to the official tatsu-lab implementation
JUDGE_PROMPT = """I want you to evaluate which of the following two AI assistant responses is better for the given user instruction.

<user_instruction>
{instruction}
</user_instruction>

<response_1>
{output_1}
</response_1>

<response_2>
{output_2}
</response_2>

Please evaluate both responses and determine which one better follows the instruction, is more helpful, accurate, and of higher quality overall.

Respond with ONLY "1" if Response 1 is better, or "2" if Response 2 is better. Do not include any explanation."""


# ── Step 1: Load AlpacaEval prompts ───────────────────────────────────────────

def load_alpaca_eval_prompts(max_samples: int | None = None) -> list[dict]:
    """Load the 805 AlpacaEval evaluation prompts with their reference outputs."""
    from datasets import load_dataset
    print("📂 Loading AlpacaEval prompts...")
    ds = load_dataset("json", data_files="https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_eval.json", split="train")
    data = list(ds)
    if max_samples:
        data = data[:max_samples]
    print(f"   Loaded {len(data)} prompts.")
    return data


# ── Step 2: Generate model responses ──────────────────────────────────────────

def generate_responses(
    prompts: list[dict],
    model_name_or_path: str,
    adapter_path: str | None,
    batch_size: int = 4,
) -> list[str]:
    """
    Generate responses from the model under evaluation.
    Loads as base + LoRA adapter if adapter_path is given.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"\n🤖 Loading model: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path:
        print(f"   Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()

    # Build chat-formatted prompts using the llama3 template
    def format_prompt(instruction: str) -> str:
        return (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{instruction.strip()}"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    print(f"   Generating responses for {len(prompts)} prompts (batch_size={batch_size})...")
    responses = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch = prompts[i : i + batch_size]
        formatted = [format_prompt(ex["instruction"]) for ex in batch]

        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(
            outputs[:, input_len:], skip_special_tokens=True
        )
        responses.extend([r.strip() for r in decoded])

    return responses


# ── Step 3: GPT-4 Turbo judging ───────────────────────────────────────────────

def judge_with_gpt4(
    prompts: list[dict],
    model_responses: list[str],
    openai_model: str = "gpt-4-turbo",
    output_dir: Path = Path("eval_results"),
) -> list[dict]:
    """
    For each prompt, ask GPT-4 Turbo to compare the model's response
    against the AlpacaEval reference response.

    Returns a list of dicts with:
        instruction, model_output, reference_output, winner (1 or 2),
        model_win (bool), model_len, ref_len, length_diff
    """
    from openai import OpenAI

    client   = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    cache_path = output_dir / "judgments_cache.json"

    # Load cache to resume if interrupted
    cache: dict[int, dict] = {}
    if cache_path.exists():
        cache = {int(k): v for k, v in json.loads(cache_path.read_text()).items()}
        print(f"   Resuming from cache: {len(cache)}/{len(prompts)} already judged.")

    results = []
    print(f"\n⚖️  Judging {len(prompts)} pairs with {openai_model}...")

    for i, (ex, model_out) in enumerate(tqdm(zip(prompts, model_responses), total=len(prompts), desc="Judging")):
        if i in cache:
            results.append(cache[i])
            continue

        ref_out     = ex.get("output", "")       # AlpacaEval reference (GPT-4 Turbo)
        instruction = ex["instruction"]

        # Randomise position to mitigate position bias — swap model/ref order
        # for even indices, keep original for odd
        if i % 2 == 0:
            out1, out2 = model_out, ref_out
            model_is_1 = True
        else:
            out1, out2 = ref_out, model_out
            model_is_1 = False

        prompt_text = JUDGE_PROMPT.format(
            instruction=instruction,
            output_1=out1,
            output_2=out2,
        )

        # Call GPT-4 with retries
        winner_raw = None
        for attempt in range(5):
            try:
                resp = client.chat.completions.create(
                    model=openai_model,
                    messages=[{"role": "user", "content": prompt_text}],
                    max_tokens=JUDGE_MAX_TOKENS,
                    temperature=JUDGE_TEMPERATURE,
                )
                winner_raw = resp.choices[0].message.content.strip()
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"\n   API error (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)

        if winner_raw is None:
            print(f"\n   ⚠️  Failed to get judgment for example {i}. Skipping.")
            continue

        # Parse "1" or "2" from response
        match = re.search(r"\b([12])\b", winner_raw)
        winner = int(match.group(1)) if match else None

        if winner is None:
            print(f"\n   ⚠️  Could not parse winner from: '{winner_raw}'. Skipping.")
            continue

        # Determine if model won, accounting for position swap
        if model_is_1:
            model_win = (winner == 1)
        else:
            model_win = (winner == 2)

        model_len  = len(model_out)
        ref_len    = len(ref_out)
        length_diff = model_len - ref_len

        record = {
            "idx":             i,
            "instruction":     instruction,
            "model_output":    model_out,
            "reference_output": ref_out,
            "winner":          winner,
            "model_win":       model_win,
            "model_len":       model_len,
            "ref_len":         ref_len,
            "length_diff":     length_diff,
        }
        results.append(record)
        cache[i] = record

        # Save cache every 50 examples
        if i % 50 == 0:
            cache_path.write_text(json.dumps(cache, indent=2))

    cache_path.write_text(json.dumps(cache, indent=2))
    return results


# ── Step 4 & 5: GLM fitting → LC win rate ─────────────────────────────────────

def compute_lc_winrate(results: list[dict]) -> dict:
    """
    Fit a Generalized Linear Model (logistic regression) on:
        win ~ length_diff + length_diff^2 + log(model_len) + log(ref_len)

    Then re-predict at length_diff=0 (and mean log lengths) to get the
    length-controlled win rate — i.e. "what would the win rate be if both
    responses were the same length?"

    Returns a dict with raw_winrate, lc_winrate, and GLM coefficients.
    """
    from sklearn.linear_model import LogisticRegression

    wins        = np.array([int(r["model_win"]) for r in results])
    length_diff = np.array([r["length_diff"]    for r in results], dtype=float)
    model_len   = np.array([r["model_len"]      for r in results], dtype=float)
    ref_len     = np.array([r["ref_len"]        for r in results], dtype=float)

    # Build feature matrix
    X = np.column_stack([
        length_diff,
        length_diff ** 2,
        np.log1p(model_len),
        np.log1p(ref_len),
    ])

    # Fit logistic regression (GLM with logit link)
    glm = LogisticRegression(max_iter=1000, fit_intercept=True)
    glm.fit(X, wins)

    # Raw win rate
    raw_winrate = wins.mean() * 100

    # Length-controlled: set length_diff=0, keep log lengths at their means
    X_controlled = np.column_stack([
        np.zeros(len(results)),                    # length_diff = 0
        np.zeros(len(results)),                    # length_diff^2 = 0
        np.full(len(results), np.log1p(model_len).mean()),  # mean log model len
        np.full(len(results), np.log1p(ref_len).mean()),    # mean log ref len
    ])

    lc_probs    = glm.predict_proba(X_controlled)[:, 1]
    lc_winrate  = lc_probs.mean() * 100

    coef_names  = GLM_FEATURES
    coefs       = dict(zip(coef_names, glm.coef_[0].tolist()))
    coefs["intercept"] = glm.intercept_[0]

    return {
        "raw_winrate":    round(raw_winrate,   2),
        "lc_winrate":     round(lc_winrate,    2),
        "n_examples":     len(results),
        "n_wins":         int(wins.sum()),
        "glm_coefs":      coefs,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Check for OpenAI key
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    output_dir = Path(RunConfig.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load prompts
    prompts = load_alpaca_eval_prompts(max_samples=RunConfig.MAX_SAMPLES)

    # ── Step 2: Generate responses
    outputs_path = output_dir / "model_outputs.json"

    if RunConfig.SKIP_GENERATION and outputs_path.exists():
        print(f"\n⏭️  Skipping generation. Loading from {outputs_path}")
        model_responses = json.loads(outputs_path.read_text())
    else:
        model_responses = generate_responses(
            prompts=prompts,
            model_name_or_path=RunConfig.MODEL_NAME_OR_PATH,
            adapter_path=RunConfig.ADAPTER_PATH,
            batch_size=RunConfig.BATCH_SIZE,
        )
        outputs_path.write_text(json.dumps(model_responses, indent=2))
        print(f"   Saved model outputs → {outputs_path}")

    # ── Step 3: Judge with GPT-4
    results = judge_with_gpt4(
        prompts=prompts,
        model_responses=model_responses,
        openai_model=RunConfig.OPENAI_MODEL,
        output_dir=output_dir,
    )

    # ── Steps 4 & 5: Fit GLM → LC win rate
    print("\n📊 Computing Length-Controlled Win Rate...")
    metrics = compute_lc_winrate(results)

    # ── Save and print results
    results_path = output_dir / "lc_alpaca_results.json"
    full_output  = {"metrics": metrics, "model": RunConfig.MODEL_NAME_OR_PATH, "results": results}
    results_path.write_text(json.dumps(full_output, indent=2))

    print("\n" + "=" * 50)
    print("  LC-AlpacaEval-2 Results")
    print("=" * 50)
    print(f"  Model          : {RunConfig.MODEL_NAME_OR_PATH}")
    print(f"  Examples       : {metrics['n_examples']}")
    print(f"  Raw Win Rate   : {metrics['raw_winrate']:.2f}%")
    print(f"  LC Win Rate    : {metrics['lc_winrate']:.2f}%   ← main metric")
    print(f"  GLM coefs      : {metrics['glm_coefs']}")
    print("=" * 50)
    print(f"\n  Full results → {results_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    if args.adapter_path is not None:
        RunConfig.ADAPTER_PATH = args.adapter_path
    if args.output_dir is not None:
        RunConfig.OUTPUT_DIR = args.output_dir
    main()