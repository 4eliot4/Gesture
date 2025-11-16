#!/bin/bash
#SBATCH --job-name=preprocess_videos
#SBATCH --account=team-ai
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --time=20:00:00
#SBATCH --output=logs/slurm-preprocess_videos-%j.out
#SBATCH --error=logs/slurm-preprocess_videos-%j.err

set -x
ulimit -0

export PYTHONBUFFERED=1

source .venv/bin/activate

python -u preprocess_videos.py \
    --input_folder "$SCRATCH/gesture/data/wlaslvideos" \
    --output_folder "$SCRATCH/gesture/data/wlaslvideos_processed" \
    --dataset_json "data/WLASL_v0.3.json" \
    --include_pose \
    --include_face \
    --include_hands \
    --min_detection_confidence 0.5 \
    --min_tracking_confidence 0.5 \
