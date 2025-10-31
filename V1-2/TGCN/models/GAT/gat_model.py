import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Alias torch.nn.functional as UN for brevity.
import torch.nn.functional as UN

# Mixed precision imports using the new torch.amp namespace.
from torch.amp import autocast, GradScaler

# ----------------------------------------------------------
# -------------------- Model Definition --------------------------
# Graph Attention Layer definition.
class GraphAttentionLayer(nn.Module):
    """
    A simplified spatial graph attention layer.
    For each node pair (i, j):
       e_{ij} = LeakyReLU( a^T [W h_i || W h_j] )
    and attention coefficients:
       α_{ij} = softmax_j(e_{ij})
    """
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.empty((2*out_features, 1)))
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # h: (B, N, in_features)
        B, N, _ = h.size()
        Wh = self.W(h)  # (B, N, out_features)
        Wh_i = Wh.unsqueeze(2).expand(B, N, N, self.out_features)
        Wh_j = Wh.unsqueeze(1).expand(B, N, N, self.out_features)
        a_input = torch.cat([Wh_i, Wh_j], dim=-1)  # (B, N, N, 2*out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(-1)  # (B, N, N)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(B, -1, -1)
        e = e.masked_fill(adj == 0, -9e15)
        attention = torch.softmax(e, dim=-1)
        attention = UN.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.bmm(attention, Wh)  # (B, N, out_features)
        if self.concat:
            return UN.elu(h_prime)
        else:
            return h_prime

class MultiHeadDynamicGAT(nn.Module):
    def __init__(self, nfeat, nhid, nheads, dropout=0.1, alpha=0.2):
        super(MultiHeadDynamicGAT, self).__init__()
        self.heads = nn.ModuleList([
            GraphAttentionLayer(nfeat, nhid, dropout, alpha, concat=True)
            for _ in range(nheads)
        ])
        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout, alpha, concat=False)
    
    def forward(self, x, adj):
        head_outputs = [head(x, adj) for head in self.heads]  # Each: (B, N, nhid)
        x_cat = torch.cat(head_outputs, dim=-1)               # (B, N, nhid * nheads)
        x_cat = UN.dropout(x_cat, p=0.1, training=self.training)
        x_out = self.out_att(x_cat, adj)                      # (B, N, nhid)
        return x_out

class TemporalAttention(nn.Module):
    """Applies attention over the temporal dimension to get a weighted sum of temporal features."""
    def __init__(self, feature_dim):
        super(TemporalAttention, self).__init__()
        self.attn_fc = nn.Linear(feature_dim, 1)

    def forward(self, x):
        # x: (B, T, feature_dim)
        scores = self.attn_fc(x)              # (B, T, 1)
        weights = torch.softmax(scores, dim=1) # (B, T, 1)
        weighted = torch.sum(weights * x, dim=1)  # (B, feature_dim)
        return weighted

class GatingFusion(nn.Module):
    """Fuses two feature vectors using a learnable gate."""
    def __init__(self, feature_dim):
        super(GatingFusion, self).__init__()
        self.fc = nn.Linear(feature_dim * 2, feature_dim)

    def forward(self, x1, x2):
        # x1, x2: (B, feature_dim)
        combined = torch.cat([x1, x2], dim=-1)
        gate = torch.sigmoid(self.fc(combined))
        fused = gate * x1 + (1 - gate) * x2
        return fused

class DynamicGATTemporalModel(nn.Module):
    def __init__(self, max_frames, hidden_feature, num_class, num_nodes):
        """
        Args:
            max_frames (int): Number of frames (after padding/truncation).
            hidden_feature (int): Hidden feature dimension for the spatial GAT.
            num_class (int): Number of output classes.
            num_nodes (int): Number of nodes (landmarks).
        """
        super(DynamicGATTemporalModel, self).__init__()
        self.num_nodes = num_nodes
        self.max_frames = max_frames
        self.hidden_feature = hidden_feature

        # The spatial GAT processes each frame independently.
        # It expects input of shape (B, num_nodes, 8) per frame.
        self.spatial_gat = MultiHeadDynamicGAT(nfeat=8, nhid=hidden_feature, nheads=4, dropout=0.1, alpha=0.2)
        
        # Temporal attention module.
        self.temporal_att = TemporalAttention(feature_dim=hidden_feature)
        
        # Gating fusion to combine the outputs from raw and augmented streams.
        self.gate_fusion = GatingFusion(feature_dim=hidden_feature)
        
        # Final classification.
        self.fc = nn.Linear(hidden_feature, num_class)

    def forward(self, x):
        # x: (B, T, num_nodes, 16)
        B, T, N, C = x.size()  # C=16
        # Split the combined features into raw and augmented: each gets 8 features.
        raw = x[..., :8]  # (B, T, N, 8)
        aug = x[..., 8:]  # (B, T, N, 8)

        # Process each time frame by merging batch and time dimensions.
        raw = raw.view(B * T, N, 8)  # (B*T, N, 8)
        aug = aug.view(B * T, N, 8)  # (B*T, N, 8)

        # Create a static, fully-connected adjacency matrix.
        adj = torch.ones(N, N, device=x.device)

        # Apply the spatial GAT separately on each branch.
        raw_spatial = self.spatial_gat(raw, adj)  # (B*T, N, hidden_feature)
        aug_spatial = self.spatial_gat(aug, adj)    # (B*T, N, hidden_feature)

        # Pool spatially (average over nodes) to get per-frame features.
        raw_frame_feat = torch.mean(raw_spatial, dim=1)  # (B*T, hidden_feature)
        aug_frame_feat = torch.mean(aug_spatial, dim=1)    # (B*T, hidden_feature)

        # Reshape back to sequence format.
        raw_seq = raw_frame_feat.view(B, T, self.hidden_feature)  # (B, T, hidden_feature)
        aug_seq = aug_frame_feat.view(B, T, self.hidden_feature)    # (B, T, hidden_feature)

        # Apply temporal attention to both streams.
        raw_temporal = self.temporal_att(raw_seq)  # (B, hidden_feature)
        aug_temporal = self.temporal_att(aug_seq)    # (B, hidden_feature)

        # Fuse the two streams using a gating mechanism.
        fused = self.gate_fusion(raw_temporal, aug_temporal)  # (B, hidden_feature)

        # Final classification.
        out = self.fc(fused)  # (B, num_class)
        return torch.log_softmax(out, dim=1)

# New subclass that adds hand emphasis.
class DynamicGATTemporalModelWithHandEmphasis(DynamicGATTemporalModel):
    def __init__(self, max_frames, hidden_feature, num_class, num_nodes, hand_indices, hand_scale=1.5):
        """
        Args:
            hand_indices (list or tensor): indices of the nodes corresponding to hand landmarks.
            hand_scale (float): factor by which to scale hand node features.
            The rest of the arguments are as in the parent model.
        """
        super(DynamicGATTemporalModelWithHandEmphasis, self).__init__(max_frames, hidden_feature, num_class, num_nodes)
        # Save hand indices and scaling factor.
        self.register_buffer('hand_indices', torch.tensor(hand_indices))
        self.hand_scale = hand_scale

    def forward(self, x):
        # x: (B, T, num_nodes, 16)
        B, T, N, C = x.size()
        # Split into raw and augmented features (each: 8 channels).
        raw = x[..., :8]  # (B, T, N, 8)
        aug = x[..., 8:]  # (B, T, N, 8)
        
        # Apply hand emphasis by scaling the features corresponding to the hand nodes.
        raw[:, :, self.hand_indices, :] *= self.hand_scale
        aug[:, :, self.hand_indices, :] *= self.hand_scale

        # Process each frame: merge batch and time dimensions.
        raw = raw.view(B * T, N, 8)   # (B*T, N, 8)
        aug = aug.view(B * T, N, 8)   # (B*T, N, 8)
        
        # Create a static, fully-connected adjacency matrix.
        adj = torch.ones(N, N, device=x.device)
        
        # Apply the spatial GAT on each branch.
        raw_spatial = self.spatial_gat(raw, adj)  # (B*T, N, hidden_feature)
        aug_spatial = self.spatial_gat(aug, adj)    # (B*T, N, hidden_feature)
        
        # Pool spatially (average over nodes) to get per-frame features.
        raw_frame_feat = torch.mean(raw_spatial, dim=1)  # (B*T, hidden_feature)
        aug_frame_feat = torch.mean(aug_spatial, dim=1)    # (B*T, hidden_feature)
        
        # Reshape back to sequence format.
        raw_seq = raw_frame_feat.view(B, T, self.hidden_feature)  # (B, T, hidden_feature)
        aug_seq = aug_frame_feat.view(B, T, self.hidden_feature)    # (B, T, hidden_feature)
        
        # Apply temporal attention to both streams.
        raw_temporal = self.temporal_att(raw_seq)  # (B, hidden_feature)
        aug_temporal = self.temporal_att(aug_seq)    # (B, hidden_feature)
        
        # Fuse the two streams using a gating mechanism.
        fused = self.gate_fusion(raw_temporal, aug_temporal)  # (B, hidden_feature)
        
        # Final classification.
        out = self.fc(fused)  # (B, num_class)
        return torch.log_softmax(out, dim=1)
