import lm_eval
import json
import os


# ── Config — edit before running ──────────────────────────────────────────────
class EvalConfig:
    MODEL_PATH   = "allenai/Llama-3.1-Tulu-3-8B-SFT"
    ADAPTER_PATH = None                              # set to adapter path if using DPO LoRA
    OUTPUT_DIR   = "prediction/ifeval"
    BATCH_SIZE   = 4
    DEVICE       = "cuda"
    LIMIT        = None                              # set to int (e.g. 100) for quick test
# ──────────────────────────────────────────────────────────────────────────────


def main():
    os.makedirs(EvalConfig.OUTPUT_DIR, exist_ok=True)

    print(f"🚀 Model: {EvalConfig.MODEL_PATH}")
    print(f"🎯 Evaluation: IFEval (instruction following)")

    model_args = f"pretrained={EvalConfig.MODEL_PATH},dtype=bfloat16"
    if EvalConfig.ADAPTER_PATH:
        print(f"🧩 Adapter: {EvalConfig.ADAPTER_PATH}")
        model_args += f",peft={EvalConfig.ADAPTER_PATH}"

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=["ifeval"],
        num_fewshot=0,               # IFEval is always 0-shot
        batch_size=EvalConfig.BATCH_SIZE,
        device=EvalConfig.DEVICE,
        limit=EvalConfig.LIMIT,
    )

    # IFEval reports four metrics — all are useful
    ifeval_results = results["results"]["ifeval"]
    prompt_strict  = ifeval_results.get("prompt_level_strict_acc,none", 0.0) * 100
    prompt_loose   = ifeval_results.get("prompt_level_loose_acc,none",  0.0) * 100
    inst_strict    = ifeval_results.get("inst_level_strict_acc,none",   0.0) * 100
    inst_loose     = ifeval_results.get("inst_level_loose_acc,none",    0.0) * 100

    print(f"\n{'='*50}")
    print(f"  IFEval Results")
    print(f"{'='*50}")
    print(f"  Prompt-level strict : {prompt_strict:.2f}%  ← main metric")
    print(f"  Prompt-level loose  : {prompt_loose:.2f}%")
    print(f"  Inst-level strict   : {inst_strict:.2f}%")
    print(f"  Inst-level loose    : {inst_loose:.2f}%")
    print(f"{'='*50}")

    output_path = os.path.join(EvalConfig.OUTPUT_DIR, "ifeval_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✅ Results saved to {output_path}")


if __name__ == "__main__":
    main()