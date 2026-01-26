import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Training/Testing Wrapper")
    parser.add_argument("--config", type=str, required=True, help="Path to the yaml config file")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"❌ Error: Config file '{args.config}' not found.")
        sys.exit(1)

    print(f"🚀 Starting Llama Factory with config: {args.config}")
    
    # Construct the command
    # We use 'train' for both training and prediction as per Llama Factory design
    cmd = ["llamafactory-cli", "train", args.config]

    try:
        # Run the command and stream output to console
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Process failed with exit code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()