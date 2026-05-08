import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.project_paths import default_data_root


def load_bin(path):
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return pts[:, :3]


def build_parser():
    parser = argparse.ArgumentParser(description="Visualize raw, unnormalized KITTI LiDAR.")
    parser.add_argument("--data-root", default=str(default_data_root()))
    parser.add_argument("--sequence", default="09")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--save", default=None, help="Optional PNG output path.")
    parser.add_argument("--no-show", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()

    seq_dir = Path(args.data_root) / args.sequence / "velodyne"
    if not seq_dir.exists():
        raise FileNotFoundError(f"Sequence directory does not exist: {seq_dir}")

    bin_files = sorted(
        [
            seq_dir / f
            for f in os.listdir(seq_dir)
            if f.endswith(".bin")
        ]
    )
    if not bin_files:
        raise RuntimeError(f"No .bin files found in {seq_dir}")

    frame_idx = min(max(args.frame_index, 0), len(bin_files) - 1)
    file_path = bin_files[frame_idx]
    points = load_bin(file_path)

    print("Loaded:", file_path)
    print("Shape:", points.shape)

    fig = plt.figure(figsize=(16, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.15)
    ax1.set_title("Raw KITTI LiDAR 3D")
    ax1.set_axis_off()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(points[:, 0], points[:, 1], s=0.15)
    ax2.set_title("Raw KITTI LiDAR BEV")
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=200)
        print(f"Saved figure: {args.save}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
