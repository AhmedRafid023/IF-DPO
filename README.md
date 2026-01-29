# DPO Fine-Tuning Pipeline (Docker + Llama Factory)

This repository provides a containerized workflow for fine-tuning Large Language Models (LLMs) using **Direct Preference Optimization (DPO)**. It utilizes [Llama Factory](https://github.com/hiyouga/LLaMA-Factory) for the training backend and wraps the process in a clean Docker environment to ensure reproducibility.

## 📂 Project Structure

```text
.
├── configs/
│   ├── train.yaml          # Hyperparameters for DPO training
│   └── test.yaml           # Settings for inference/testing
├── data/
│   ├── dataset_info.json   # Auto-generated registry for Llama Factory
│   └── (train.json/test.json) # Generated data files
├── .env                    # Secrets (HF Token, WandB Key)
├── Dockerfile              # Environment definition
├── prepare_data.py         # Script to download & format datasets
├── train_model.py          # Wrapper to run training/testing commands
└── README.md


docker build -t dpo-v1 .


# This runs prepare_data.py inside the container
docker run --rm -v $(pwd):/app dpo-v1 python prepare_data.py --all



# This runs train_model.py which triggers llamafactory-cli
docker run --gpus all --ipc=host --rm --env-file .env -v $(pwd):/app dpo-v1 python train_model.py --config configs/train.yaml



docker run --gpus all --ipc=host --rm --env-file .env -v $(pwd):/app dpo-v1 python train_model.py --config configs/test.yaml


docker run --gpus all --ipc=host --rm --env-file .env -v $(pwd):/app dpo-v1 python influence.py