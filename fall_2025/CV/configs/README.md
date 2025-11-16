# Configuration Files

This directory contains YAML configuration files for training experiments.

## Usage


Run training with a config file:
```bash
python main.py --config configs/config.yaml
```

## Configuration Structure

### wandb
- `project`: Wandb project name
- `entity`: Wandb team/entity (optional)
- `name`: Run name (auto-generated if null)
- `tags`: List of tags for the run
- `notes`: Description of the experiment
- `mode`: "online", "offline", or "disabled"

### data
- `gloss_mapping_file`: Path to CSV mapping glosses to labels
- `data_root`: Root directory containing processed videos
- `coordinate_system`: "shoulder_centered" or "original"
- `min_frames`: Minimum frames to include a sample
- `max_frames`: Maximum frames (sequences will be padded/truncated)
- `train_split`: Fraction of data for training (rest is validation)
- `random_seed`: Random seed for reproducibility

### dataloader
- `batch_size`: Batch size for training
- `num_workers`: Number of data loading workers
- `shuffle_train`: Whether to shuffle training data
- `pin_memory`: Pin memory for faster GPU transfer

### training
- `num_epochs`: Number of training epochs
- `learning_rate`: Initial learning rate
- `weight_decay`: L2 regularization weight
- `optimizer`: "adam", "adamw", or "sgd"
- `grad_clip`: Gradient norm clipping (null to disable)
- `scheduler`: Learning rate scheduler (null to disable)
- `checkpoint_dir`: Directory to save model checkpoints

### device
- `device`: "cuda", "cpu", or "auto"

## Environment Variables

API keys (wandb, huggingface, etc.) should be set via environment variables, typically loaded from a `.env` file in your slurm script:

```bash
# .env file example
WANDB_API_KEY=your_key_here
HF_TOKEN=your_token_here
```

The slurm script loads these automatically.
