# Gesture Recognition Training Tutorial

This tutorial will guide you through running a training session for the STGCN-based gesture recognition model on the WLASL (World Large-Scale Sign Language) dataset.

**Note:** Video preprocessing is NOT necessary if you're using the pre-processed parquet files from the shared data folder.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Data Setup](#data-setup)
3. [Environment Setup](#environment-setup)
4. [Configuration](#configuration)
5. [Running Training](#running-training)
6. [Monitoring Training](#monitoring-training)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- Python 3.12 or higher
- Access to a machine with GPU (CUDA-compatible)
- Access to `/work/team-ai/GESTURE/fall_2025_data/data` folder (for pre-processed data)
- Weights & Biases account (optional, for experiment tracking)

---

## Data Setup

### Step 1: Copy Pre-processed Data

The pre-processed parquet files are available in the shared team folder. You need to copy them to your scratch directory.

```bash
# Copy the pre-processed data to your scratch directory
cp -r /work/team-ai/GESTURE/fall_2025_data/data/wlaslvideos_processed $SCRATCH/gesture/data/

# Verify the data was copied successfully
ls $SCRATCH/gesture/data/wlaslvideos_processed
```

### Step 2: Verify Data Files

Each parquet file contains landmark data for one video with the following columns:
- `type`: Body part type (pose, face, left_hand, right_hand)
- `landmark_index`: Index of the landmark within the body part
- `frame`: Frame number
- `x`, `y`, `z`: 3D coordinates
- `visibility`: Landmark visibility score

The `train.csv` metadata file should have columns:
- `path`: Path to the parquet file
- `sign`: Sign name (gloss)
- `sequence_id`: Unique video identifier

---

## Environment Setup

### Step 1: Create Virtual Environment

```bash
# Navigate to the project directory
cd /path/to/GESTURE/fall_2025/CV

# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

### Step 2: Install Dependencies

The project uses `uv` package manager, but you can also use `pip`:

**Option A: Using uv (recommended)**
```bash
# Install uv if not already installed
pip install uv

# Install dependencies
uv sync
```

**Option B: Using pip**
```bash
# Install dependencies from pyproject.toml
pip install -e .
```

Key dependencies that will be installed:
- `torch>=2.8.0` - PyTorch for deep learning
- `mediapipe>=0.10.21` - Landmark extraction (not needed for training)
- `fastparquet>=2024.11.0` - Parquet file reading
- `wandb>=0.22.3` - Experiment tracking
- `numpy`, `pandas`, `pyyaml` - Data processing and configuration

### Step 3: Set Up Weights & Biases (Optional)

If you want to track your experiments with Weights & Biases:

```bash
# Login to wandb
wandb login

# Or create a .env file with your API key
echo "WANDB_API_KEY=your_api_key_here" > .env
```

You can get your API key from [https://wandb.ai/authorize](https://wandb.ai/authorize)

---

## Configuration

### Step 1: Review Configuration File

The main configuration file is located at [configs/config.yaml](configs/config.yaml). Here are the key settings you may want to modify:

```yaml
# Data settings
data:
  gloss_mapping_file: "data/gloss_map.csv"
  metadata_file: "$SCRATCH/gesture/data/wlaslvideos_processed/train.csv"
  cache_dir: "$SCRATCH/gesture/unfied_shoulder_centered"
  data_root: "$SCRATCH/gesture/data/wlaslvideos_processed"
  coordinate_system: "shoulder_centered"  # Normalizes to shoulder midpoint
  max_frames: 300  # Maximum frames per video
  train_split: 0.8  # 80% train, 20% validation
  body_parts: ["pose", "left_hand", "right_hand"]  # Body parts to use

# Model settings
model:
  name: "STGCN (conv1d)"
  num_blocks: 4  # Number of spatio-temporal blocks
  channels: 64  # Feature channels
  kernel_size: 3  # Temporal convolution kernel size
  dropout: 0.5

# Training settings
training:
  num_epochs: 500
  learning_rate: 0.001
  batch_size: 32
  optimizer: "adamw"
  scheduler: "cosine"  # Cosine annealing learning rate
  checkpoint_dir: "$SCRATCH/gesture/checkpoints"
  save_frequency: 5  # Save checkpoint every 5 epochs
```

### Step 2: Update Paths (If Necessary)

Make sure the following paths in `configs/config.yaml` point to your data:

1. **`data.metadata_file`**: Should point to your `train.csv` file
2. **`data.data_root`**: Should point to the folder containing parquet files
3. **`data.cache_dir`**: Directory where processed data will be cached
4. **`training.checkpoint_dir`**: Directory where model checkpoints will be saved

**Note:** Environment variables like `$SCRATCH` will be automatically expanded.

### Step 3: Create Logs Directory

```bash
mkdir -p logs
```

---

## Running Training

### Option 1: Local Training (CPU or Single GPU)

For quick testing or running on a local machine:

```bash
# Activate virtual environment
source .venv/bin/activate

# Run training with default config
python main.py --config configs/config.yaml

# Or with a custom run ID
python main.py --config configs/config.yaml --run-id my_experiment_01
```

### Option 2: SLURM Cluster Training (Recommended)

For running on an HPC cluster with SLURM:

```bash
# Submit job with default config
sbatch train.sh

# Or specify a custom config file
sbatch train.sh -c configs/my_custom_config.yaml
```

The [train.sh](train.sh) script will:
- Request 1 GPU, 30 CPUs, and 32GB RAM
- Run for up to 24 hours
- Load environment variables from `.env`
- Activate the virtual environment
- Run training with the specified config
- Save logs to `logs/gesture_training_<job_id>.out`

### What Happens During Training

1. **Data Loading**:
   - Loads metadata from `train.csv`
   - Filters to only include glosses in `gloss_map.csv` (127 classes)
   - Splits data into train (80%) and validation (20%)
   - Creates cache of processed tensors for faster loading

2. **Model Initialization**:
   - Creates STGCN model with specified architecture
   - Number of nodes determined by body parts (default: 59 nodes)
     - Pose: 17 landmarks
     - Left hand: 21 landmarks
     - Right hand: 21 landmarks
   - Moves model to GPU if available

3. **Training Loop**:
   - Trains for specified number of epochs (default: 500)
   - Each epoch:
     - Trains on training set with data augmentation
     - Evaluates on validation set
     - Logs metrics (loss, top-1 accuracy, top-5 accuracy)
     - Saves checkpoints every N epochs
     - Updates learning rate with cosine annealing

4. **Checkpointing**:
   - Saves model state, optimizer state, and training config
   - Checkpoint location: `$SCRATCH/gesture/checkpoints/<run_id>/`
   - Files: `checkpoint_epoch_<N>.pt`, `best_model.pt`

---

## Monitoring Training

### Option 1: Weights & Biases Dashboard

If you enabled wandb (default), you can monitor training in real-time:

1. Go to [https://wandb.ai](https://wandb.ai)
2. Navigate to your project (default: "gesture-recognition")
3. View metrics:
   - Training/validation loss
   - Top-1 and top-5 accuracy
   - Learning rate schedule
   - Gradient norms
   - GPU utilization

### Option 2: SLURM Logs

If running on SLURM, check the output logs:

```bash
# View latest training log
tail -f logs/gesture_training_<job_id>.out

# Check for errors
tail -f logs/gesture_training_<job_id>.err

# Check job status
squeue -u $USER
```

### Option 3: Checkpoint Files

Check saved checkpoints:

```bash
ls -lh $SCRATCH/gesture/checkpoints/<run_id>/
```

---

## Troubleshooting

### Issue: Out of Memory (OOM) Error

**Solution**: Reduce batch size in `configs/config.yaml`:
```yaml
dataloader:
  batch_size: 16  # Reduce from 32
```

### Issue: Data Not Found

**Error**: `FileNotFoundError: [Errno 2] No such file or directory: '...'`

**Solution**:
1. Verify you copied data from `/work/team-ai/GESTURE/fall_2025_data/data/wlaslvideos_processed`
2. Check that paths in `configs/config.yaml` are correct
3. Ensure `$SCRATCH` environment variable is set: `echo $SCRATCH`

### Issue: Slow Data Loading

**Solution**:
1. Increase number of workers (if you have enough CPUs):
   ```yaml
   dataloader:
     num_workers: 4  # Increase from 1
   ```
2. The first epoch will be slower as it creates the cache. Subsequent epochs will be faster.

---

## Additional Resources

- **Model Architecture**: See [src/models/stgcn.py](src/models/stgcn.py)
- **Data Loader**: See [src/data_loading/parquet_data_loader.py](src/data_loading/parquet_data_loader.py)
- **Configuration Details**: See [configs/README.md](configs/README.md)
- **Gloss Mapping**: See [data/gloss_map.csv](data/gloss_map.csv) for the 127 sign classes

---
