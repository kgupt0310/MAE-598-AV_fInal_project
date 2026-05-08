import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src.corruption.corruption import PointCloudCorruption


class KITTISectorDataset(Dataset):
    """
    Splits each LiDAR frame into angular sectors.

    Returns:
        corrupted:  [N, 3]
        clean:      [N, 3]
        prev_chunk: [N, 3]
    """

    def __init__(
        self,
        root_dir,
        sequences,
        num_points=2048,
        num_sectors=16,
        max_frames_per_seq=None,
        scale=50.0,
    ):
        self.root_dir = root_dir
        self.sequences = sequences
        self.num_points = num_points
        self.num_sectors = num_sectors
        self.max_frames_per_seq = max_frames_per_seq
        self.scale = scale

        self.files = []
        self.frame_pairs = []
        self.samples = []
        self.corruptor = PointCloudCorruption()

        self._index_files()
        self._build_sector_samples()

    def _index_files(self):
        for seq in self.sequences:
            seq_dir = os.path.join(self.root_dir, seq, "velodyne")
            if not os.path.exists(seq_dir):
                print(f"[WARN] Missing sequence directory: {seq_dir}")
                continue

            bin_files = sorted(
                [
                    os.path.join(seq_dir, f)
                    for f in os.listdir(seq_dir)
                    if f.endswith(".bin")
                ]
            )

            if self.max_frames_per_seq is not None:
                bin_files = bin_files[: self.max_frames_per_seq]

            self.files.extend(bin_files)

            for i, curr_file in enumerate(bin_files):
                prev_file = bin_files[max(i - 1, 0)]
                self.frame_pairs.append((curr_file, prev_file))

        print(f"[INFO] Loaded {len(self.files)} frames")

    def __len__(self):
        return len(self.samples)

    def load_bin(self, path):
        pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        return pts[:, :3]

    def sector_mask(self, points, sector_idx):
        x = points[:, 0]
        y = points[:, 1]
        angles = np.arctan2(y, x)
        sector_edges = np.linspace(-np.pi, np.pi, self.num_sectors + 1)

        if sector_idx == self.num_sectors - 1:
            return (angles >= sector_edges[sector_idx]) & (
                angles <= sector_edges[sector_idx + 1]
            )

        return (angles >= sector_edges[sector_idx]) & (
            angles < sector_edges[sector_idx + 1]
        )

    def split_into_sectors(self, points):
        sectors = []
        for sector_idx in range(self.num_sectors):
            sector_pts = points[self.sector_mask(points, sector_idx)]
            if len(sector_pts) > 0:
                sectors.append(sector_pts)
        return sectors

    def sample_fixed_size(self, pts):
        if len(pts) >= self.num_points:
            idx = np.random.choice(len(pts), self.num_points, replace=False)
            return pts[idx]

        idx = np.random.choice(len(pts), self.num_points - len(pts), replace=True)
        return np.concatenate([pts, pts[idx]], axis=0)

    def _build_sector_samples(self):
        print("[INFO] Building sector sample index...")

        for curr_file, prev_file in self.frame_pairs:
            curr = self.load_bin(curr_file) / self.scale
            prev = self.load_bin(prev_file) / self.scale

            for sector_idx in range(self.num_sectors):
                curr_pts = curr[self.sector_mask(curr, sector_idx)]
                prev_pts = prev[self.sector_mask(prev, sector_idx)]

                if len(curr_pts) == 0 or len(prev_pts) == 0:
                    continue

                self.samples.append((curr_file, prev_file, sector_idx))

        print(f"[INFO] Total sector chunks: {len(self.samples)}")

    def __getitem__(self, idx):
        curr_file, prev_file, sector_idx = self.samples[idx]

        curr = self.load_bin(curr_file) / self.scale
        prev = self.load_bin(prev_file) / self.scale

        clean = curr[self.sector_mask(curr, sector_idx)]
        prev_chunk = prev[self.sector_mask(prev, sector_idx)]

        clean = self.sample_fixed_size(clean).astype(np.float32)
        prev_chunk = self.sample_fixed_size(prev_chunk).astype(np.float32)

        corrupted = self.corruptor(clean.copy(), self.num_points)

        return (
            torch.tensor(corrupted, dtype=torch.float32),
            torch.tensor(clean, dtype=torch.float32),
            torch.tensor(prev_chunk, dtype=torch.float32),
        )
