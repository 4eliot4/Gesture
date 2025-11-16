import math

import torch
from torch import nn
import torch.nn.functional as F


# From the paper https://arxiv.org/abs/1709.04875

class TimeBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        assert kernel_size % 2 == 1, "Use an odd kernel_size for 'same' padding."

        # We use Conv2d is used as the input we will expect is of 4D, so using kernel (1, kernel_size) makes it behave as 1D

        pad_t = kernel_size // 2
        self.out_channels = out_channels

        self.conv = nn.Conv2d(
            in_channels,
            2 * out_channels,
            (1, kernel_size),
            padding=(0, pad_t),  # keep time dimension unchanged
        )  # 2 * out_channels as we will separate the results, and this makes it just one fused operation
        self.conv_2 = nn.Conv2d(
            in_channels, out_channels, (1, kernel_size), padding=(0, pad_t)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform one graph-convolution step using the normalized Laplacian.

        Args:
            x (torch.Tensor): Node feature matrix of shape (Batch, F_in, N, Time),
                                where N is number of nodes and F_in is feature size.
        Returns:
            torch.Tensor: Updated node features of shape (Batch, F_out, N, Time_f).
        """

        out = self.conv(x)  # (batch, 2 * out_channels, N, time_f)
        p_out = out[:, 0 : self.out_channels]  # (batch, out_channels, N, time_f)
        q_out = out[:, self.out_channels :]  # (batch, out_channels, N, time_f)

        temp = p_out * F.sigmoid(q_out)  # gate
        x = F.relu(temp + self.conv_2(x)) # residual connection

        return x # Batch, F_out, N, Time_f


class STGCNBlock(nn.Module):
    def __init__(
        self, in_channels: int, spatial_channels: int, out_channels: int, num_nodes: int
    ):
        super().__init__()

        self.temp_block_1 = TimeBlock(
            in_channels=in_channels, out_channels=out_channels
        )

        # Here spatial just means that we go back to a spatial dimension
        self.Theta = nn.Parameter(
            torch.FloatTensor(out_channels, spatial_channels)
        )  # The filter. (k, out_channels, spatial_channels), and here hops k=1

        self.temp_block_2 = TimeBlock(
            in_channels=spatial_channels, out_channels=out_channels
        )
        self.batch_norm = nn.BatchNorm2d(num_nodes)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.Theta)

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        """
        Perform one graph-convolution step using the normalized Laplacian.

        Args:
            x (torch.Tensor): Node feature matrix of shape (Batch, F_in, N, Time),
                                where N is number of nodes and F_in is feature size.
            A_hat (torch.Tensor): Normalized Laplacian (or adjacency) matrix
                                        of the graph, shape (N, N).

        Returns:
            torch.Tensor: Updated node features of shape (Batch, F_out, N, time).
        """

        temp_conv = self.temp_block_1(x)  # (batch, out_channel, N, time)
        temp_conv = F.relu(temp_conv)

        # 1st order approximation Graph convolution
        # Filter theta applies a frequency response g(λ) to the signal’s spectral components at those eigenvalues (so it acts on the components of signal x along each eigenvector of Laplacian)

        # Channel mix (Θ): (B, out_channel, N, time) @ (Cout, Cspat) -> (B, Cspat, N, time)
        h = torch.einsum("bfnt,fs->bsnt", temp_conv, self.Theta)
        # Node mix (Â): (N, N) @ (B, Cspat, N, Time) -> (B, Cspat, N, Time)
        y = torch.einsum("ij,bsjt->bsit", A_hat, h)
        conv = F.relu(y)

        # temp block again
        temp_conv = self.temp_block_2(conv)
        x = temp_conv
        # x = F.relu(temp_conv)  # (batch, out_channels, N, time)

        # Permute to (batch, N, time, out_channels) to normalize over N
        x = x.permute(0, 2, 3, 1)
        x = self.batch_norm(x)
        # Permute back to (batch, out_channels, N, time)
        x = x.permute(0, 3, 1, 2)
        return x


class STGCN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_features: int,
        num_classes: int,
        num_blocks: int = 4,
        channels: int = 64,
        spatial_channels: int = 16,
        kernel_size: int = 3,
        mlp_hidden=(256, 128),
        dropout: float = 0.5,
    ):
        super().__init__()

        # num_blocks = amount of k-hops graph model can see

        assert num_blocks >= 2

        blocks = []
        in_c = num_features
        for b in range(num_blocks):
            blocks.append(STGCNBlock(in_c, spatial_channels, channels, num_nodes))
            in_c = channels
        self.blocks = nn.ModuleList(blocks)

        self.head_temporal = (
            TimeBlock(channels, channels, kernel_size=kernel_size)
        )

        # MLP head after global average pooling over (N, T)
        mlp_layers = []
        prev = channels
        for h in mlp_hidden:
            mlp_layers += [
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ]
            prev = h
        mlp_layers += [nn.Linear(prev, num_classes)]
        self.classifier = nn.Sequential(*mlp_layers)

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        """
        Perform one graph-convolution step using the normalized Laplacian.

        Args:
            x (torch.Tensor): Node feature matrix of shape (Batch, Time, N, F_in),
                                where N is number of nodes and F_in is feature size.
            A_hat (torch.Tensor): Normalized Laplacian (or adjacency) matrix
                                        of the graph, shape (N, N).

        Returns:
            torch.Tensor: Updated node features of shape (Batch, F_out, N, time).
        """
        x = x.permute(0, 3, 2, 1)  # (batch, in_features, N, time)

        for block in self.blocks:
            residual = x
            x = block(x, A_hat) # (batch, out_channel, N, time)
            if x.shape == residual.shape:
                x = x + residual

        x = self.head_temporal(x)  # (batch, out_channel, N, time)

        # global average pooling across nodes and time
        # NOTE: Don't know if we want to average across nodes
        x = x.mean(dim=(2, 3))  # (batch, channels)

        logits = self.classifier(x)  # (B, num_classes)

        return logits
