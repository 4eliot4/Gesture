from typing import Dict, List
import torch

def collate_fn_pad_sequences(batch: List[Dict], max_frames: int = 150) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader that pads sequences to a fixed length.

    Args:
        batch: List of samples from __getitem__
        max_frames: Fixed maximum number of frames (default 150)

    Returns:
        Batched dictionary with padded sequences and masks
    """
    # Get dimensions
    num_landmarks = batch[0]['landmarks'].shape[1]
    num_features = batch[0]['landmarks'].shape[2]
    batch_size = len(batch)

    # Initialize padded tensors
    landmarks_padded = torch.zeros(batch_size, max_frames, num_landmarks, num_features)
    masks = torch.zeros(batch_size, max_frames, dtype=torch.bool)
    adjacency_matrices = torch.stack([sample['adjacency'] for sample in batch])
    labels = torch.stack([sample['label'] for sample in batch])
    num_frames = torch.stack([sample['num_frames'] for sample in batch])

    # Fill in actual data
    for i, sample in enumerate(batch):
        seq_len = sample['num_frames'].item()
        seq_len = min(seq_len, max_frames)
        landmarks_padded[i, :seq_len] = sample['landmarks'][:seq_len]
        masks[i, :seq_len] = True

    return {
        'landmarks': landmarks_padded,
        'adjacency': adjacency_matrices,
        'mask': masks,
        'label': labels,
        'num_frames': num_frames,
    }

def normalize_adjacency_matrix(A: torch.Tensor, num_nodes: int, device):
    A_tilde = A + torch.eye(num_nodes).to(device)
    D_tilde = torch.diag(torch.sum(A_tilde, dim=1))
    D_tilde_inv_sqrt = torch.pow(D_tilde, -0.5)
    D_tilde_inv_sqrt[torch.isinf(D_tilde_inv_sqrt)] = (
        0.0  # Handle cases of isolated nodes
    )

    A_hat = torch.mm(torch.mm(D_tilde_inv_sqrt, A_tilde), D_tilde_inv_sqrt)

    return A_hat
