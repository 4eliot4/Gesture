#!/bin/bash
#SBATCH --job-name=gesture_training
#SBATCH --output=logs/gesture_training_%j.out
#SBATCH --error=logs/gesture_training_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# WANDB_MODE=disabled

# Change to the script directory
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

# Load .env file for API keys (wandb, huggingface, etc.)
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

set -x # so it doesn't print the api keys from the .env on the logs

# Save current git commit hash for reproducibility
export GIT_COMMIT=$(git rev-parse HEAD)
export GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Git commit: $GIT_COMMIT"
echo "Git branch: $GIT_BRANCH"

# Load any required modules
# module load cuda/11.8
# module load python/3.10

# Activate virtual environment
source .venv/bin/activate

echo $PYTHONPATH

# Append SLURM job ID to wandb run name if running on SLURM
if [ ! -z "$SLURM_JOB_ID" ]; then
    export WANDB_RUN_NAME_SUFFIX="${SLURM_JOB_ID}"
fi

# Run training with config file
python main.py --config configs/config.yaml

# Or with a different config
# python main.py --config configs/experiment1.yaml
