import lm_eval
import json
import os


# ── Config — edit before running ──────────────────────────────────────────────
class EvalConfig:
    MODEL_PATH   = "allenai/Llama-3.1-Tulu-3-8B-SFT"
    ADAPTER_PATH = None                              # set to adapter path if using DPO LoRA
    OUTPUT_DIR   = "prediction/mmlu"
    BATCH_SIZE   = 4
    DEVICE       = "cuda"
    NUM_FEWSHOT  = 5
    LIMIT        = None                              # set to int (e.g. 50) for quick test

    # All 57 canonical MMLU subjects
    TASKS = [
        "mmlu_abstract_algebra",
        "mmlu_anatomy",
        "mmlu_astronomy",
        "mmlu_business_ethics",
        "mmlu_clinical_knowledge",
        "mmlu_college_biology",
        "mmlu_college_chemistry",
        "mmlu_college_computer_science",
        "mmlu_college_mathematics",
        "mmlu_college_medicine",
        "mmlu_college_physics",
        "mmlu_computer_security",
        "mmlu_conceptual_physics",
        "mmlu_econometrics",
        "mmlu_electrical_engineering",
        "mmlu_elementary_mathematics",
        "mmlu_formal_logic",
        "mmlu_global_facts",
        "mmlu_high_school_biology",
        "mmlu_high_school_chemistry",
        "mmlu_high_school_computer_science",
        "mmlu_high_school_european_history",
        "mmlu_high_school_geography",
        "mmlu_high_school_government_and_politics",
        "mmlu_high_school_macroeconomics",
        "mmlu_high_school_mathematics",
        "mmlu_high_school_microeconomics",
        "mmlu_high_school_physics",
        "mmlu_high_school_psychology",
        "mmlu_high_school_statistics",
        "mmlu_high_school_us_history",
        "mmlu_high_school_world_history",
        "mmlu_human_aging",
        "mmlu_human_sexuality",
        "mmlu_international_law",
        "mmlu_jurisprudence",
        "mmlu_logical_fallacies",
        "mmlu_machine_learning",
        "mmlu_management",
        "mmlu_marketing",
        "mmlu_medical_genetics",
        "mmlu_miscellaneous",
        "mmlu_moral_disputes",
        "mmlu_moral_scenarios",
        "mmlu_nutrition",
        "mmlu_philosophy",
        "mmlu_prehistory",
        "mmlu_professional_accounting",
        "mmlu_professional_law",
        "mmlu_professional_medicine",
        "mmlu_professional_psychology",
        "mmlu_public_relations",
        "mmlu_security_studies",
        "mmlu_sociology",
        "mmlu_us_foreign_policy",
        "mmlu_virology",
        "mmlu_world_religions",
    ]
# ──────────────────────────────────────────────────────────────────────────────


def main():
    os.makedirs(EvalConfig.OUTPUT_DIR, exist_ok=True)

    print(f"🚀 Model: {EvalConfig.MODEL_PATH}")
    print(f"🎯 Evaluation: {EvalConfig.NUM_FEWSHOT}-shot MMLU (all {len(EvalConfig.TASKS)} subjects)")

    model_args = f"pretrained={EvalConfig.MODEL_PATH},dtype=bfloat16"
    if EvalConfig.ADAPTER_PATH:
        print(f"🧩 Adapter: {EvalConfig.ADAPTER_PATH}")
        model_args += f",peft={EvalConfig.ADAPTER_PATH}"

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=EvalConfig.TASKS,
        num_fewshot=EvalConfig.NUM_FEWSHOT,
        batch_size=EvalConfig.BATCH_SIZE,
        device=EvalConfig.DEVICE,
        limit=EvalConfig.LIMIT,
    )

    scores = []
    print(f"\n{'='*60}")
    print(f"  MMLU Results ({EvalConfig.NUM_FEWSHOT}-shot) — all {len(EvalConfig.TASKS)} subjects")
    print(f"{'='*60}")
    for task in EvalConfig.TASKS:
        acc = results["results"][task]["acc,none"] * 100
        scores.append(acc)
        print(f"  {task.replace('mmlu_', ''):<45} {acc:.2f}%")

    avg = sum(scores) / len(scores)
    print(f"{'─'*60}")
    print(f"  {'Average':<45} {avg:.2f}%")
    print(f"{'='*60}")

    output_path = os.path.join(EvalConfig.OUTPUT_DIR, "mmlu_results.json")
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