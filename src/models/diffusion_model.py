import torch
import torch.nn as nn


class ConditionalDiffusionModel(nn.Module):
    def __init__(self, hidden_dim=256, global_dim=256):
        super().__init__()

        self.point_encoder = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, global_dim),
            nn.ReLU(),
        )

        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim, global_dim),
            nn.ReLU(),
            nn.Linear(global_dim, global_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(global_dim + global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, x_t, corrupted, t):
        B, N, _ = x_t.shape

        t = t.float().view(B, 1, 1)
        t = t.repeat(1, N, 1) / 50.0

        residual = x_t - corrupted

        inp = torch.cat([x_t, corrupted, residual, t], dim=2)

        point_feat = self.point_encoder(inp)

        global_feat = torch.max(point_feat, dim=1)[0]
        global_feat = self.global_mlp(global_feat)
        global_feat = global_feat.unsqueeze(1).repeat(1, N, 1)

        combined = torch.cat([point_feat, global_feat], dim=2)
        pred_noise = self.decoder(combined)

        return pred_noise
