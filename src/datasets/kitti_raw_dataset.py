import os

import numpy as np
from torch.utils.data import Dataset


class KITTIRawDataset(Dataset):
    """
    Loads full KITTI LiDAR frames without normalization.
    Good for raw visualization, BEV, and realistic scene display.
    """

    def __init__(self, root_dir, sequences, max_frames_per_seq=None):
        self.root_dir = root_dir
        self.sequences = sequences
        self.max_frames_per_seq = max_frames_per_seq

        self.files = []
        self._index_files()

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

        print(f"[INFO] Raw viz loaded {len(self.files)} frames")

    def __len__(self):
        return len(self.files)

    def load_bin(self, path):
        pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        return pts[:, :3]

    def __getitem__(self, idx):
        file_path = self.files[idx]
        points = self.load_bin(file_path)
        return points, file_path
