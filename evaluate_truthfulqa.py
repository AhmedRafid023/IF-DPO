import lm_eval
import json
import os


# ── Config — edit before running ──────────────────────────────────────────────
class EvalConfig:
    MODEL_PATH   = "allenai/Llama-3.1-Tulu-3-8B-SFT"
    ADAPTER_PATH = None                              # set to adapter path if using DPO LoRA
    OUTPUT_DIR   = "prediction/truthfulqa"
    BATCH_SIZE   = 8
    DEVICE       = "cuda"
    LIMIT        = None                              # set to int (e.g. 100) for quick test
# ──────────────────────────────────────────────────────────────────────────────


def main():
    os.makedirs(EvalConfig.OUTPUT_DIR, exist_ok=True)

    print(f"🚀 Model: {EvalConfig.MODEL_PATH}")
    print(f"🎯 Evaluation: TruthfulQA MC2 (0-shot)")

    model_args = f"pretrained={EvalConfig.MODEL_PATH},dtype=bfloat16"
    if EvalConfig.ADAPTER_PATH:
        print(f"🧩 Adapter: {EvalConfig.ADAPTER_PATH}")
        model_args += f",peft={EvalConfig.ADAPTER_PATH}"

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=["truthfulqa_mc2"],
        num_fewshot=0,               # standard for TruthfulQA
        batch_size=EvalConfig.BATCH_SIZE,
        device=EvalConfig.DEVICE,
        limit=EvalConfig.LIMIT,
    )

    score = results["results"]["truthfulqa_mc2"].get("acc,none", 0.0) * 100

    print(f"\n{'='*50}")
    print(f"  TruthfulQA MC2 Results")
    print(f"{'='*50}")
    print(f"  Score : {score:.2f}%")
    print(f"{'='*50}")

    output_path = os.path.join(EvalConfig.OUTPUT_DIR, "truthfulqa_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✅ Results saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    if args.adapter_path is not None:
        EvalConfig.ADAPTER_PATH = args.adapter_path
    if args.output_dir is not None:
        EvalConfig.OUTPUT_DIR = args.output_dir
    main()