import os

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset

from src.corruption.corruption import PointCloudCorruption


class KITTIDataset(Dataset):
    """
    Lazy kNN chunk dataset for KITTI Velodyne frames.

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
        scale=50.0,
        max_frames_per_seq=500,
        chunks_per_frame=1,
        center_selection="random",
        start_frame=0,
    ):
        self.root_dir = root_dir
        self.sequences = sequences
        self.num_points = num_points
        self.scale = scale
        self.max_frames_per_seq = max_frames_per_seq
        self.chunks_per_frame = chunks_per_frame
        self.center_selection = center_selection
        self.start_frame = start_frame

        self.samples = []
        self.corruptor = PointCloudCorruption()

        self._index_frames()

    def __len__(self):
        return len(self.samples)

    def load_bin(self, path):
        pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        return pts[:, :3]

    def _sample_fixed_size(self, points):
        if len(points) >= self.num_points:
            idx = np.random.choice(len(points), self.num_points, replace=False)
            return points[idx]

        idx = np.random.choice(len(points), self.num_points - len(points), replace=True)
        return np.concatenate([points, points[idx]], axis=0)

    def get_knn_chunk(self, points, center_point):
        if len(points) < self.num_points:
            return self._sample_fixed_size(points)

        nbrs = NearestNeighbors(n_neighbors=self.num_points).fit(points)
        idx = nbrs.kneighbors([center_point], return_distance=False)[0]
        return points[idx]

    def _choose_centers(self, points, num_centers):
        if self.center_selection == "random":
            replace = len(points) < num_centers
            return np.random.choice(len(points), size=num_centers, replace=replace)

        if self.center_selection != "fps_xy":
            raise ValueError(
                "center_selection must be either 'random' or 'fps_xy', "
                f"got {self.center_selection!r}"
            )

        xy = points[:, :2]
        first_idx = int(np.argmin(np.linalg.norm(xy, axis=1)))
        centers = [first_idx]

        min_dist_sq = np.sum((xy - xy[first_idx]) ** 2, axis=1)
        for _ in range(1, num_centers):
            next_idx = int(np.argmax(min_dist_sq))
            centers.append(next_idx)
            dist_sq = np.sum((xy - xy[next_idx]) ** 2, axis=1)
            min_dist_sq = np.minimum(min_dist_sq, dist_sq)

        return np.array(centers, dtype=np.int64)

    def _index_frames(self):
        print("[INFO] Indexing frames...")

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

            if self.start_frame > 0:
                bin_files = bin_files[self.start_frame :]

            if self.max_frames_per_seq is not None:
                bin_files = bin_files[: self.max_frames_per_seq]

            for i, curr_path in enumerate(bin_files):
                curr = self.load_bin(curr_path)
                if len(curr) < self.num_points:
                    continue

                prev_path = bin_files[max(i - 1, 0)]

                num_centers = min(self.chunks_per_frame, len(curr))
                centers = self._choose_centers(curr, num_centers)

                for center_idx in centers:
                    self.samples.append((curr_path, prev_path, int(center_idx)))

        print(f"[INFO] Total samples: {len(self.samples)}")

    def __getitem__(self, idx):
        curr_path, prev_path, center_idx = self.samples[idx]

        curr = self.load_bin(curr_path) / self.scale
        prev = self.load_bin(prev_path) / self.scale

        center_point = curr[center_idx]

        clean = self.get_knn_chunk(curr, center_point)
        prev_chunk = self.get_knn_chunk(prev, center_point)

        corrupted = self.corruptor(clean.copy(), self.num_points)

        return (
            torch.tensor(corrupted, dtype=torch.float32),
            torch.tensor(clean, dtype=torch.float32),
            torch.tensor(prev_chunk, dtype=torch.float32),
        )
