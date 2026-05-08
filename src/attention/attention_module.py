import torch
import torch.nn as nn


class MotionAttention(nn.Module):
    """
    Attention using:
      - distance from sensor
      - speed magnitude
      - radial motion
      - time-to-collision proxy
    """

    def __init__(self, hidden_dim=64, max_ttc=50.0):
        super().__init__()
        self.max_ttc = max_ttc

        self.mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, curr, prev):
        """
        Args:
            curr: [B, N, 3] current chunk
            prev: [B, N, 3] previous-frame chunk

        Returns:
            weights: [B, N, 1]
        """
        eps = 1e-6

        dist = torch.norm(curr, dim=2, keepdim=True)
        dist_feat = torch.log1p(dist)

        pairwise = torch.cdist(curr, prev)
        nn_idx = pairwise.argmin(dim=2)

        B, N, _ = curr.shape
        prev_nn = prev.gather(1, nn_idx.unsqueeze(-1).expand(B, N, 3))

        motion_vec = curr - prev_nn

        speed = torch.norm(motion_vec, dim=2, keepdim=True)
        speed = speed / (speed.mean(dim=1, keepdim=True) + eps)

        ray_dir = curr / (torch.norm(curr, dim=2, keepdim=True) + eps)

        radial_vel = (motion_vec * ray_dir).sum(dim=2, keepdim=True)
        radial_feat = radial_vel / (radial_vel.abs().mean(dim=1, keepdim=True) + eps)

        approaching = (radial_vel < 0).float()
        ttc = dist / (-radial_vel + eps)
        ttc = approaching * ttc + (1.0 - approaching) * self.max_ttc
        ttc = torch.clamp(ttc, 0.0, self.max_ttc)

        max_ttc_tensor = torch.tensor(1.0 + self.max_ttc, device=curr.device)
        ttc_feat = torch.log1p(ttc) / torch.log(max_ttc_tensor)

        feats = torch.cat([dist_feat, speed, radial_feat, ttc_feat], dim=2)
        return self.mlp(feats)
