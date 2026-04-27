#!/usr/bin/env bash
# Usage: bash run_experiments.sh <config.yaml> [<config.yaml> ...]
# Example: bash run_experiments.sh experiments/gsm8k.yaml experiments/mmlu.yaml
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "Usage: $0 <config.yaml> [<config.yaml> ...]"
    echo ""
    echo "Examples:"
    echo "  $0 experiments/gsm8k.yaml"
    echo "  $0 experiments/gsm8k.yaml experiments/mmlu.yaml"
    echo "  $0 experiments/*.yaml"
    exit 1
}

[[ $# -eq 0 ]] && usage

for CONFIG_FILE in "$@"; do
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "ERROR: Config file not found: $CONFIG_FILE"
        exit 1
    fi

    echo ""
    echo "========================================================"
    echo "Config : $CONFIG_FILE"
    echo "========================================================"

    python3 - "$CONFIG_FILE" "$SCRIPT_DIR" <<'PYEOF'
import sys
import os
import subprocess

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml is required. Install with: pip install pyyaml")
    sys.exit(1)

config_file = sys.argv[1]
base_dir = sys.argv[2]

with open(config_file) as f:
    config = yaml.safe_load(f)

script_path = os.path.join(base_dir, config["script"])
experiments = config.get("experiments", [])

if not experiments:
    print("No experiments defined in config.")
    sys.exit(0)

print(f"Dataset    : {config.get('dataset', 'unknown')}")
print(f"Script     : {script_path}")
print(f"Experiments: {len(experiments)}")

for i, exp in enumerate(experiments, 1):
    name         = exp["name"]
    adapter_path = exp.get("adapter_path")
    output_dir   = exp["output_dir"]

    print(f"\n{'='*60}")
    print(f"  [{i}/{len(experiments)}] {name}")
    print(f"  Adapter : {adapter_path or 'None (base SFT)'}")
    print(f"  Output  : {output_dir}")
    print(f"{'='*60}")

    cmd = [sys.executable, script_path, "--output_dir", output_dir]
    if adapter_path:
        cmd += ["--adapter_path", adapter_path]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nERROR: Experiment '{name}' failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n  Finished: {name}")

print(f"\nAll {len(experiments)} experiments completed.")
PYEOF

done

echo ""
echo "All configs processed successfully."
