import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
import wandb
import os
import time

from src.models.stgcn import STGCN
from src.utils.config_loader import load_config
from src.data_loading.parquet_data_loader import UnifiedSkeletonDataset
from src.data_loading.utils.utils import collate_fn_pad_sequences, normalize_adjacency_matrix


def train(model, train_loader, val_loader, num_epochs, learning_rate, device, num_nodes, config):
    """Train the STGCN model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=config["training"]["weight_decay"])

    scheduler = None
    if config["training"]["scheduler"] == "cosine":
        total_steps = num_epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=config["training"]["scheduler_config"]["min_lr"]
        )
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    checkpoint_dir = Path(os.path.expandvars(config["training"]["checkpoint_dir"]))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_correct_top_5 = 0
        train_total = 0
        grad_norms = []

        for batch in train_loader:
            start_time = time.perf_counter()
            landmarks = batch['landmarks'].to(device)
            adjacency = batch['adjacency'].to(device)
            labels = batch['label'].to(device)

            # Normalize adjacency matrix
            adjacency_norm = normalize_adjacency_matrix(adjacency[0], num_nodes, device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(landmarks, adjacency_norm)

            # Loss and backward
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping and norm calculation
            if config["training"]["grad_clip"] is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            grad_norms.append(grad_norm.item())

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # Statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            _, predicted_top_5 = torch.topk(outputs, dim=1, k=5)
            train_correct += predicted.eq(labels).sum().item()
            train_correct_top_5 += predicted_top_5.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
            train_total += labels.size(0)

            # Log batch metrics
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": loss.item(),
                    "train/grad_norm": grad_norm.item(),
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/step_time": time.perf_counter() - start_time,
                },
                step=global_step
            )
            global_step += 1

        wandb.log(
            {
                "epoch": epoch + 1,
                "train/accuracy": 100 * train_correct / train_total,
                "train/accuracy_top5": 100 * train_correct_top_5 / train_total
            },
            step=global_step
        )

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_correct_top_5 = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                landmarks = batch['landmarks'].to(device)
                adjacency = batch['adjacency'].to(device)
                labels = batch['label'].to(device)

                adjacency_norm = normalize_adjacency_matrix(adjacency[0], num_nodes, device)

                outputs = model(landmarks, adjacency_norm)

                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                _, predicted_top_5 = torch.topk(outputs, dim=1, k=5)
                val_correct += predicted.eq(labels).sum().item()
                val_correct_top_5 += predicted_top_5.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
                val_total += labels.size(0)

        # Calculate metrics
        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        train_acc = 100.0 * train_correct / train_total
        train_acc_top5 = 100.0 * train_correct_top_5 / train_total
        val_acc = 100.0 * val_correct / val_total
        val_acc_top5 = 100.0 * val_correct_top_5 / val_total
        avg_grad_norm = np.mean(grad_norms)

        # Print stats
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}% (Top-5: {train_acc_top5:.2f}%)")
        print(f"  Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}% (Top-5: {val_acc_top5:.2f}%)")
        print(f"  Avg Grad Norm: {avg_grad_norm:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "val/loss": val_loss_avg,
            "val/accuracy": val_acc,
            "val/accuracy_top5": val_acc_top5,
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = checkpoint_dir / "best_model.pth"
            torch.save(model.state_dict(), model_path)
            print(f"  --> Best model saved! Val Acc: {val_acc:.2f}%")

            wandb.run.summary["best_val_acc"] = best_val_acc
            wandb.run.summary["best_epoch"] = epoch + 1

    print(f"\nTraining complete! Best val acc: {best_val_acc:.2f}%")
    return best_val_acc


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train STGCN model on gesture recognition")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load configuration
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Setup device
    if config["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["device"])
    print(f"Using device: {device}")

    # Initialize wandb
    import os
    if config["wandb"]["name"] and os.environ.get("WANDB_RUN_NAME_SUFFIX"):
        config["wandb"]["name"] = f'{config["wandb"]["name"]}_{os.environ.get("WANDB_RUN_NAME_SUFFIX")}'

    # Add git info to config
    git_commit = os.environ.get("GIT_COMMIT", "unknown")
    git_branch = os.environ.get("GIT_BRANCH", "unknown")

    wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        name=config["wandb"]["name"],
        tags=config["wandb"]["tags"],
        notes=config["wandb"]["notes"],
        mode=config["wandb"]["mode"],
        config={
            **config.to_dict(),
            "git_commit": git_commit,
            "git_branch": git_branch,
        },
    )
    print(f"Wandb initialized: {wandb.run.name}")
    print(f"Git commit: {git_commit}")
    print(f"Git branch: {git_branch}")

    # --- Create dataset ---
    print("\nLoading dataset...")

    # Load gloss mapping
    import pandas as pd
    gloss_df = pd.read_csv(config["data"]["gloss_mapping_file"])
    gloss_map = dict(zip(gloss_df['gloss'], gloss_df['label']))

    # Create cache directory
    cache_dir = config["data"]["cache_dir"]

    dataset = UnifiedSkeletonDataset(
        data_root=config["data"]["data_root"],
        cache_dir=config["data"]["cache_dir"],
        gloss_map=gloss_map,
        metadata_file=config["data"]["metadata_file"],
        min_frames=config["data"]["min_frames"],
        max_frames=config["data"]["max_frames"],
        body_parts=["pose", "left_hand", "right_hand"]
    )

    # Split dataset
    torch.manual_seed(config["data"]["random_seed"])
    train_size = int(config["data"]["train_split"] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # Create dataloaders with fixed max_frames
    from functools import partial
    collate_fn = partial(collate_fn_pad_sequences, max_frames=config["data"]["max_frames"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["dataloader"]["batch_size"],
        shuffle=config["dataloader"]["shuffle_train"],
        collate_fn=collate_fn,
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"]["pin_memory"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["dataloader"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"]["pin_memory"],
    )

    # Get model parameters from data
    sample_batch = next(iter(train_loader))
    num_nodes = sample_batch['landmarks'].shape[2]
    num_features = sample_batch['landmarks'].shape[3]
    num_timesteps = config["data"]["max_frames"]
    num_classes = dataset.num_classes

    print("\nModel parameters:")
    print(f"  Num nodes: {num_nodes}, Num features: {num_features}")
    print(f"  Num timesteps (fixed): {num_timesteps}, Num classes: {num_classes}")

    # Create model
    model = STGCN(
        num_nodes=num_nodes,
        num_features=num_features,
        num_classes=num_classes,
        num_blocks=config["model"].get("num_blocks", 4),
        channels=config["model"].get("channels", 64),
        spatial_channels=config["model"].get("spatial_channels", 16),
        kernel_size=config["model"].get("kernel_size", 3),
        mlp_hidden=tuple(config["model"].get("mlp_hidden", [256, 128])),
        dropout=config["model"].get("dropout", 0.5),
    ).to(device)

    # model = torch.compile(model)

    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters\n")

    # Train
    best_val_acc = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config["training"]["num_epochs"],
        learning_rate=config["training"]["learning_rate"],
        device=device,
        num_nodes=num_nodes,
        config=config,
    )

    # Finish wandb run
    wandb.finish()
    print("Wandb run finished")
