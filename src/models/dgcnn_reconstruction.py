import torch
import torch.nn as nn

from src.attention.attention_module import MotionAttention


def knn(x, k):
    dist = torch.cdist(x, x)
    idx = dist.topk(k=k, dim=-1, largest=False)[1]
    return idx


def get_graph_feature(x, k=16):
    B, N, C = x.shape
    idx = knn(x, k=k)

    idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
    idx = idx + idx_base
    idx = idx.reshape(-1)

    x_flat = x.reshape(B * N, C)
    neighbors = x_flat[idx].reshape(B, N, k, C)

    x_center = x.unsqueeze(2).repeat(1, 1, k, 1)
    edge_feature = torch.cat([neighbors - x_center, x_center], dim=-1)
    return edge_feature


class EdgeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k=16):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        B, N, C = x.shape
        edge = get_graph_feature(x, self.k)
        edge = edge.reshape(B * N * self.k, 2 * C)

        feat = self.mlp(edge)
        feat = feat.reshape(B, N, self.k, -1)
        return feat.max(dim=2)[0]


class DGCNNReconstructor(nn.Module):
    def __init__(self, k=16, global_dim=256):
        super().__init__()
        self.k = k

        self.attention = MotionAttention(hidden_dim=64)

        self.edge1 = EdgeConvBlock(3, 64, k=k)
        self.edge2 = EdgeConvBlock(64, 128, k=k)
        self.edge3 = EdgeConvBlock(128, 256, k=k)

        self.global_mlp = nn.Sequential(
            nn.Linear(64 + 128 + 256, global_dim),
            nn.ReLU(),
            nn.Linear(global_dim, global_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(64 + 128 + 256 + global_dim + 3, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, x, prev_chunk):
        """
        Args:
            x:          [B, N, 3]
            prev_chunk: [B, N, 3]

        Returns:
            out:     [B, N, 3]
            weights: [B, N]
        """
        weights = self.attention(x, prev_chunk)

        x_gated = x * (1.0 + weights)

        f1 = self.edge1(x_gated)
        f1 = f1 * (1.0 + weights)

        f2 = self.edge2(f1)
        f2 = f2 * (1.0 + weights)

        f3 = self.edge3(f2)
        f3 = f3 * (1.0 + weights)

        local_feat = torch.cat([f1, f2, f3], dim=2)

        global_feat = torch.max(local_feat, dim=1)[0]
        global_feat = self.global_mlp(global_feat)
        global_feat = global_feat.unsqueeze(1).repeat(1, x.shape[1], 1)

        combined = torch.cat([local_feat, global_feat, x_gated], dim=2)
        delta = self.decoder(combined)

        out = x + delta
        return out, weights.squeeze(-1)
