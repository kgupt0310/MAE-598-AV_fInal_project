import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.corruption.corruption import PointCloudCorruption
from src.models.dgcnn_reconstruction import DGCNNReconstructor
from src.utils.torch_runtime import configure_cpu_threads
from src.utils.project_paths import default_data_root, project_root


def load_bin(path):
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return pts[:, :3]


def choose_centers(points, num_centers, center_selection):
    if center_selection == "random":
        replace = len(points) < num_centers
        return np.random.choice(len(points), size=num_centers, replace=replace)

    if center_selection != "fps_xy":
        raise ValueError(f"Unknown center selection: {center_selection}")

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


def build_knn_chunks(
    curr_points,
    prev_points,
    chunk_size=2048,
    chunk_multiplier=2,
    max_chunks=0,
    center_selection="random",
):
    chunks = []
    prev_chunks = []

    if len(curr_points) < chunk_size or len(prev_points) < chunk_size:
        return chunks, prev_chunks

    nbrs_curr = NearestNeighbors(n_neighbors=chunk_size).fit(curr_points)
    nbrs_prev = NearestNeighbors(n_neighbors=chunk_size).fit(prev_points)

    num_chunks = max(1, len(curr_points) // chunk_size + 1)
    num_centers = min(len(curr_points), num_chunks * chunk_multiplier)

    if max_chunks > 0:
        num_centers = min(num_centers, max_chunks)

    centers = choose_centers(curr_points, num_centers, center_selection)

    for center_idx in centers:
        center = curr_points[center_idx]
        curr_idx = nbrs_curr.kneighbors([center], return_distance=False)[0]
        prev_idx = nbrs_prev.kneighbors([center], return_distance=False)[0]

        chunks.append(curr_points[curr_idx])
        prev_chunks.append(prev_points[prev_idx])

    return chunks, prev_chunks


def load_model(path, device):
    model = DGCNNReconstructor(k=16).to(device)
    state = torch.load(path, map_location=device)

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)
    model.eval()
    return model


def plot_3d(ax, pts, title):
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.25)
    ax.set_title(title)
    ax.set_axis_off()


def plot_3d_overlay(ax, gt, pred):
    ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], s=0.2, color="gray", alpha=0.25)
    ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], s=0.35, color="red", alpha=0.65)
    ax.set_title("Overlay 3D")
    ax.set_axis_off()


def plot_bev(ax, pts, title):
    ax.scatter(pts[:, 0], pts[:, 1], s=0.2)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")


def plot_bev_overlay(ax, gt, pred):
    ax.scatter(gt[:, 0], gt[:, 1], s=0.2, color="gray", alpha=0.25, label="Ground Truth")
    ax.scatter(pred[:, 0], pred[:, 1], s=0.3, color="red", alpha=0.65, label="Reconstructed")
    ax.set_title("Overlay BEV")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()


def build_parser():
    parser = argparse.ArgumentParser(description="Visualize DGCNN chunk reconstruction.")
    parser.add_argument("--data-root", default=str(default_data_root()))
    parser.add_argument("--model-path", default=str(project_root() / "experiments" / "dgcnn_motion_attention.pth"))
    parser.add_argument("--sequence", default="09")
    parser.add_argument("--frame-index", type=int, default=1)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--scale", type=float, default=50.0)
    parser.add_argument("--chunk-multiplier", type=int, default=2)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--center-selection", choices=["random", "fps_xy"], default="random")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-prefix", default=None, help="Optional output prefix for PNGs.")
    parser.add_argument("--no-show", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()

    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("Using device:", device)
    if device.type == "cpu":
        print("Torch CPU threads:", configure_cpu_threads())

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint does not exist: {model_path}")

    model = load_model(model_path, device)
    print("Model loaded!")

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
    prev_idx = max(0, frame_idx - 1)

    curr_points = load_bin(bin_files[frame_idx]) / args.scale
    prev_points = load_bin(bin_files[prev_idx]) / args.scale

    print("Current points:", curr_points.shape)

    chunks, prev_chunks = build_knn_chunks(
        curr_points,
        prev_points,
        chunk_size=args.num_points,
        chunk_multiplier=args.chunk_multiplier,
        max_chunks=args.max_chunks,
        center_selection=args.center_selection,
    )
    print("Number of chunks:", len(chunks))

    if not chunks:
        raise RuntimeError("No chunks were created. Check frame size and --num-points.")

    corruptor = PointCloudCorruption()
    corruptor.remove_prob = 0.08
    corruptor.noise_std = 0.003
    corruptor.num_outliers = 10

    corrupted_chunks = []
    reconstructed_chunks = []

    with torch.no_grad():
        for chunk, prev_chunk in zip(chunks, prev_chunks):
            corrupted = corruptor(chunk.copy(), args.num_points)

            corrupted_tensor = torch.tensor(corrupted, dtype=torch.float32).unsqueeze(0).to(device)
            prev_tensor = torch.tensor(prev_chunk, dtype=torch.float32).unsqueeze(0).to(device)

            pred, _ = model(corrupted_tensor, prev_tensor)

            corrupted_chunks.append(corrupted)
            reconstructed_chunks.append(pred.squeeze(0).cpu().numpy())

    clean_full = np.vstack(chunks)
    corrupted_full = np.vstack(corrupted_chunks)
    reconstructed_full = np.vstack(reconstructed_chunks)

    min_len = min(len(clean_full), len(corrupted_full), len(reconstructed_full))
    clean_plot = clean_full[:min_len]
    corrupted_plot = corrupted_full[:min_len]
    reconstructed_plot = reconstructed_full[:min_len]

    fig_3d = plt.figure(figsize=(22, 6))
    ax1 = fig_3d.add_subplot(1, 4, 1, projection="3d")
    plot_3d(ax1, corrupted_plot, "Corrupted 3D")

    ax2 = fig_3d.add_subplot(1, 4, 2, projection="3d")
    plot_3d(ax2, reconstructed_plot, "DGCNN Reconstruction 3D")

    ax3 = fig_3d.add_subplot(1, 4, 3, projection="3d")
    plot_3d(ax3, clean_plot, "Clean Chunk GT 3D")

    ax4 = fig_3d.add_subplot(1, 4, 4, projection="3d")
    plot_3d_overlay(ax4, clean_plot, reconstructed_plot)

    fig_3d.tight_layout()

    fig_bev, axes = plt.subplots(1, 4, figsize=(22, 5))
    plot_bev(axes[0], corrupted_plot, "Corrupted BEV")
    plot_bev(axes[1], reconstructed_plot, "DGCNN Reconstructed BEV")
    plot_bev(axes[2], clean_plot, "Clean Chunk GT BEV")
    plot_bev_overlay(axes[3], clean_plot, reconstructed_plot)

    fig_bev.tight_layout()

    if args.save_prefix:
        fig_3d.savefig(f"{args.save_prefix}_3d.png", dpi=200)
        fig_bev.savefig(f"{args.save_prefix}_bev.png", dpi=200)
        print(f"Saved figures with prefix: {args.save_prefix}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
