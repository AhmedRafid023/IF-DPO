"""
score_mmlu.py — Compute average MMLU accuracy from saved prediction JSON(s).

Usage:
    # Single result file
    python score_mmlu.py --path prediction/mmlu/base_sft/mmlu_results.json

    # Directory containing mmlu_results.json (auto-discovered)
    python score_mmlu.py --path prediction/mmlu/base_sft

    # Compare every sub-directory under a parent directory
    python score_mmlu.py --path prediction/mmlu
"""

import argparse
import json
import os
import sys

# All 57 canonical MMLU subjects (same order as evaluate_mmlu.py)
MMLU_TASKS = [
    "mmlu_abstract_algebra", "mmlu_anatomy", "mmlu_astronomy",
    "mmlu_business_ethics", "mmlu_clinical_knowledge", "mmlu_college_biology",
    "mmlu_college_chemistry", "mmlu_college_computer_science",
    "mmlu_college_mathematics", "mmlu_college_medicine", "mmlu_college_physics",
    "mmlu_computer_security", "mmlu_conceptual_physics", "mmlu_econometrics",
    "mmlu_electrical_engineering", "mmlu_elementary_mathematics",
    "mmlu_formal_logic", "mmlu_global_facts", "mmlu_high_school_biology",
    "mmlu_high_school_chemistry", "mmlu_high_school_computer_science",
    "mmlu_high_school_european_history", "mmlu_high_school_geography",
    "mmlu_high_school_government_and_politics", "mmlu_high_school_macroeconomics",
    "mmlu_high_school_mathematics", "mmlu_high_school_microeconomics",
    "mmlu_high_school_physics", "mmlu_high_school_psychology",
    "mmlu_high_school_statistics", "mmlu_high_school_us_history",
    "mmlu_high_school_world_history", "mmlu_human_aging", "mmlu_human_sexuality",
    "mmlu_international_law", "mmlu_jurisprudence", "mmlu_logical_fallacies",
    "mmlu_machine_learning", "mmlu_management", "mmlu_marketing",
    "mmlu_medical_genetics", "mmlu_miscellaneous", "mmlu_moral_disputes",
    "mmlu_moral_scenarios", "mmlu_nutrition", "mmlu_philosophy",
    "mmlu_prehistory", "mmlu_professional_accounting", "mmlu_professional_law",
    "mmlu_professional_medicine", "mmlu_professional_psychology",
    "mmlu_public_relations", "mmlu_security_studies", "mmlu_sociology",
    "mmlu_us_foreign_policy", "mmlu_virology", "mmlu_world_religions",
]


# ── helpers ──────────────────────────────────────────────────────────────────

def resolve_json_files(path: str) -> list[str]:
    """Return a list of mmlu_results.json paths to process.

    Accepts:
      - A direct path to a .json file.
      - A directory that contains mmlu_results.json → returns that single file.
      - A directory whose children each contain mmlu_results.json → returns all.
    """
    if os.path.isfile(path):
        return [path]

    if os.path.isdir(path):
        direct = os.path.join(path, "mmlu_results.json")
        if os.path.isfile(direct):
            return [direct]

        # Look one level deeper
        found = []
        for name in sorted(os.listdir(path)):
            candidate = os.path.join(path, name, "mmlu_results.json")
            if os.path.isfile(candidate):
                found.append(candidate)
        if found:
            return found

    sys.exit(f"❌  No mmlu_results.json found under: {path}")


def score_file(json_path: str, verbose: bool = False) -> float:
    """Load one mmlu_results.json and return average accuracy (0-100)."""
    with open(json_path) as f:
        data = json.load(f)

    results = data.get("results", {})
    scores = []
    missing = []

    for task in MMLU_TASKS:
        if task not in results:
            missing.append(task)
            continue
        acc = results[task].get("acc,none")
        if acc is None:
            missing.append(task)
            continue
        scores.append(acc * 100)

    if not scores:
        sys.exit(f"❌  No valid acc,none scores found in {json_path}")

    if missing and verbose:
        print(f"  ⚠️  Missing {len(missing)} subject(s): {', '.join(t.replace('mmlu_','') for t in missing)}")

    if verbose:
        print(f"\n  {'Subject':<45} {'Acc':>7}")
        print(f"  {'─'*45} {'─'*7}")
        for task, acc in zip(MMLU_TASKS, scores):
            print(f"  {task.replace('mmlu_', ''):<45} {acc:>6.2f}%")

    avg = sum(scores) / len(scores)
    return avg


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute average MMLU accuracy from prediction JSON(s)."
    )
    parser.add_argument(
        "--path", required=True,
        help="Path to a mmlu_results.json file, a directory containing one, "
             "or a parent directory whose sub-dirs each contain one."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-subject scores for each file."
    )
    args = parser.parse_args()

    json_files = resolve_json_files(args.path)

    print(f"\n{'='*60}")
    print(f"  MMLU Average Accuracy  ({len(MMLU_TASKS)} subjects)")
    print(f"{'='*60}")

    summary = []
    for json_path in json_files:
        label = os.path.basename(os.path.dirname(json_path))
        if args.verbose:
            print(f"\n📂  {label}  ({json_path})")
        avg = score_file(json_path, verbose=args.verbose)
        summary.append((label, avg))

    if len(summary) == 1:
        label, avg = summary[0]
        print(f"  {'Model':<45} {'Avg Acc':>7}")
        print(f"  {'─'*45} {'─'*7}")
        print(f"  {label:<45} {avg:>6.2f}%")
    else:
        # Multi-model comparison — sort best → worst
        summary.sort(key=lambda x: x[1], reverse=True)
        print(f"  {'Model':<45} {'Avg Acc':>7}")
        print(f"  {'─'*45} {'─'*7}")
        for label, avg in summary:
            print(f"  {label:<45} {avg:>6.2f}%")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
