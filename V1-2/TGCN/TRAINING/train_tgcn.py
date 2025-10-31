#!/usr/bin/env python3
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.optim as optim

from models.GAT.gat_model import *
from models.TGCN.tgcn_model import GCN_muti_att

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')


class TGCNDataset(Dataset):
    """
    Loads video feature files where each subfolder name is a numeric label.
    Each .pt file has shape (T, 300) and is reshaped to (T, 75, 4), then
    padded or truncated to a fixed max_frames.
    """
    def __init__(self, features_root, max_frames=64):
        self.features_root = features_root
        self.max_frames = max_frames
        self.samples = []
        for label_folder in os.listdir(features_root):
            folder_path = os.path.join(features_root, label_folder)
            if os.path.isdir(folder_path):
                try:
                    label = int(label_folder)
                except ValueError:
                    logging.warning(f"Skipping non-numeric folder {label_folder}")
                    continue
                for f in os.listdir(folder_path):
                    if f.endswith("_features.pt"):
                        self.samples.append((os.path.join(folder_path, f), label))
        logging.info(f"Dataset built with {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        try:
            video_tensor = torch.load(file_path)
        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}")
            video_tensor = torch.zeros((0, 300))

        if video_tensor.numel() > 0 and video_tensor.shape[1] == 300:
            T = video_tensor.shape[0]
            video_tensor = video_tensor.view(T, 75, 4)
        else:
            video_tensor = torch.zeros((0, 75, 4))

        T = video_tensor.shape[0]
        if T < self.max_frames:
            pad = torch.zeros((self.max_frames - T, 75, 4), dtype=video_tensor.dtype)
            video_tensor = torch.cat([video_tensor, pad], dim=0)
        elif T > self.max_frames:
            video_tensor = video_tensor[:self.max_frames]

        return video_tensor, label, os.path.basename(file_path)


def plot_curves(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def compute_class_weights(samples, label_to_idx):
    counts = {}
    for _, label in samples:
        counts[label] = counts.get(label, 0) + 1
    total = sum(counts.values())
    weights = np.zeros(len(label_to_idx), dtype=np.float32)
    for label, count in counts.items():
        idx = label_to_idx[label]
        weights[idx] = total / (count * len(label_to_idx))
    return torch.tensor(weights)


def run_training(features_root, max_frames, configs, save_model_to=None, val_split=0.2):
    # Build full dataset
    full_dataset = TGCNDataset(features_root=features_root, max_frames=max_frames)

    # Split into train and validation sets
    total_samples = len(full_dataset)
    val_size = int(total_samples * val_split)
    train_size = total_samples - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    logging.info(f"Train samples: {train_size}, Val samples: {val_size}")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=configs.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=configs.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    # Prepare labels
    all_labels = [label for (_, label) in full_dataset.samples]
    unique_labels = sorted(set(all_labels))
    label_to_idx = {label: label for label in unique_labels}
    num_classes = len(unique_labels)
    logging.info(f"Detected {num_classes} unique classes.")

    # Model
    model = GCN_muti_att(
        input_feature=max_frames * 4,
        hidden_feature=64,
        num_class=num_classes,
        p_dropout=configs.drop_p,
        num_stage=configs.num_stages,
        num_nodes=75
    )
    device = torch.device("mps")
    logging.info(f"Using device: {device}")
    model.to(device)

    # Loss, optimizer, scheduler
    class_weights = compute_class_weights(full_dataset.samples, label_to_idx).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        model.parameters(),
        lr=configs.init_lr,
        eps=configs.adam_eps,
        weight_decay=configs.adam_weight_decay
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_acc = 0.0
    best_epoch = -1
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    all_gt, all_preds = [], []

    for epoch in range(configs.max_epochs):
        # Training
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for features, labels, _ in train_loader:
            bsz, T, N, F = features.shape
            features = features.permute(0, 2, 1, 3).contiguous().view(bsz, N, T*F).to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * bsz
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += bsz

        train_losses.append(total_loss/total)
        train_accs.append(correct/total)
        logging.info(f"Epoch {epoch+1}/{configs.max_epochs}: Train loss={train_losses[-1]:.4f}, acc={train_accs[-1]:.4f}")

        # Validation
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        epoch_gt, epoch_preds = [], []
        with torch.no_grad():
            for features, labels, _ in val_loader:
                bsz, T, N, F = features.shape
                features = features.permute(0, 2, 1, 3).contiguous().view(bsz, N, T*F).to(device)
                labels = torch.tensor(labels, dtype=torch.long).to(device)

                outputs = model(features)
                loss = criterion(outputs, labels)

                total_loss += loss.item() * bsz
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += bsz
                epoch_gt.extend(labels.cpu().numpy())
                epoch_preds.extend(preds.cpu().numpy())

        val_losses.append(total_loss/total)
        val_accs.append(correct/total)
        scheduler.step(val_losses[-1])
        f1 = f1_score(epoch_gt, epoch_preds, average='weighted')
        logging.info(f"Epoch {epoch+1}: Val loss={val_losses[-1]:.4f}, acc={val_accs[-1]:.4f}, F1={f1:.4f}")

        all_gt.extend(epoch_gt)
        all_preds.extend(epoch_preds)

        # Checkpoint
        if val_accs[-1] > best_val_acc:
            best_val_acc = val_accs[-1]
            best_epoch = epoch
            if save_model_to:
                ckpt_dir = os.path.join(save_model_to, "checkpoints", "unified")
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f"tgcn_epoch{epoch+1}_valacc{val_accs[-1]:.4f}.pth")
                torch.save(model.state_dict(), ckpt_path)
                logging.info(f"Checkpoint saved: {ckpt_path}")

    logging.info(f"Training complete. Best val acc={best_val_acc:.4f} at epoch {best_epoch+1}")
    plot_curves(train_losses, val_losses, train_accs, val_accs)
    cm = confusion_matrix(all_gt, all_preds)
    class_names = [str(lbl) for lbl in unique_labels]
    try:
        import utils.utils as utils
        utils.plot_confusion_matrix(cm, classes=class_names, normalize=True)
    except Exception as e:
        logging.error(f"Could not plot confusion matrix: {e}")



# CUSTOM CODE TO RUN TRAINING ON SCITAS, YOU MAY NEED TO ADAPT SOME THINGS
if __name__ == "__main__":
    from configs.configs import Config
    root = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(root)
    features_root = os.path.join(root, 'training/features_out')
    max_frames = 64
    config_file = os.path.join(root, 'configs/asl1.ini')
    configs = Config(config_file)
    logging.info("Starting TGCN training with separate train/val splits")
    run_training(features_root, max_frames, configs, save_model_to='output', val_split=0.2)
    logging.info("Finished training.")
