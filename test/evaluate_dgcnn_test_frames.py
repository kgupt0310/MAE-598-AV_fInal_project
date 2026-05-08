import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(Path(__file__).resolve().parent))

from src.corruption.corruption import PointCloudCorruption
from src.loss.chamfer_loss import reconstruction_loss
from src.models.dgcnn_reconstruction import DGCNNReconstructor
from src.utils.torch_runtime import configure_cpu_threads
from test_dgcnn_full_cloud_visualization import build_knn_chunks, load_bin


def load_model(path, device):
    model = DGCNNReconstructor(k=16).to(device)
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate DGCNN on held-out KITTI frames.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--sequence", default="00")
    parser.add_argument("--start-frame", type=int, default=51)
    parser.add_argument("--num-frames", type=int, default=15)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--scale", type=float, default=50.0)
    parser.add_argument("--chunks-per-frame", type=int, default=30)
    parser.add_argument("--center-selection", choices=["random", "fps_xy"], default="fps_xy")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-csv", default="experiments/seq00_test_metrics.csv")
    return parser


def main():
    args = build_parser().parse_args()
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("Using device:", device)
    if device.type == "cpu":
        print("Torch CPU threads:", configure_cpu_threads())

    seq_dir = Path(args.data_root) / args.sequence / "velodyne"
    bin_files = sorted([seq_dir / f for f in os.listdir(seq_dir) if f.endswith(".bin")])
    if not bin_files:
        raise RuntimeError(f"No .bin files found in {seq_dir}")

    end_frame = args.start_frame + args.num_frames
    if end_frame > len(bin_files):
        raise RuntimeError(
            f"Requested frames {args.start_frame}-{end_frame - 1}, "
            f"but sequence has only {len(bin_files)} frames."
        )

    model = load_model(args.model_path, device)
    corruptor = PointCloudCorruption()

    rows = []
    totals = {"loss": 0.0, "chamfer": 0.0, "l2": 0.0, "repulsion": 0.0, "stability": 0.0, "attn": 0.0}
    total_chunks = 0

    with torch.no_grad():
        for frame_idx in range(args.start_frame, end_frame):
            curr = load_bin(bin_files[frame_idx]) / args.scale
            prev = load_bin(bin_files[max(0, frame_idx - 1)]) / args.scale

            chunks, prev_chunks = build_knn_chunks(
                curr,
                prev,
                chunk_size=args.num_points,
                max_chunks=args.chunks_per_frame,
                center_selection=args.center_selection,
            )

            frame_totals = {key: 0.0 for key in totals}
            frame_chunks = 0

            for clean, prev_chunk in zip(chunks, prev_chunks):
                corrupted = corruptor(clean.copy(), args.num_points)
                corrupted_t = torch.tensor(corrupted, dtype=torch.float32).unsqueeze(0).to(device)
                clean_t = torch.tensor(clean, dtype=torch.float32).unsqueeze(0).to(device)
                prev_t = torch.tensor(prev_chunk, dtype=torch.float32).unsqueeze(0).to(device)

                pred, attn_weights = model(corrupted_t, prev_t)
                loss, parts = reconstruction_loss(pred, clean_t, corrupted_t, attn_weights)

                values = {
                    "loss": loss.item(),
                    "chamfer": parts["chamfer"],
                    "l2": parts["l2"],
                    "repulsion": parts["repulsion"],
                    "stability": parts["stability"],
                    "attn": parts["attn"],
                }

                for key, value in values.items():
                    frame_totals[key] += value
                    totals[key] += value

                frame_chunks += 1
                total_chunks += 1

            row = {"frame": frame_idx, "chunks": frame_chunks}
            for key, value in frame_totals.items():
                row[key] = value / max(frame_chunks, 1)
            rows.append(row)

            print(
                f"Frame {frame_idx:06d} | chunks: {frame_chunks} | "
                f"loss: {row['loss']:.5f} | chamfer: {row['chamfer']:.5f} | l2: {row['l2']:.5f}"
            )

    avg = {key: value / max(total_chunks, 1) for key, value in totals.items()}
    print("\nHeld-out test average:")
    print(
        f"chunks={total_chunks} | loss={avg['loss']:.6f} | "
        f"chamfer={avg['chamfer']:.6f} | l2={avg['l2']:.6f} | attn={avg['attn']:.6f}"
    )

    save_path = Path(args.save_csv)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", newline="") as f:
        fieldnames = ["frame", "chunks", "loss", "chamfer", "l2", "repulsion", "stability", "attn"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({"frame": "AVERAGE", "chunks": total_chunks, **avg})
    print(f"Saved test metrics: {save_path}")

    print("\nTraining corruption summary:")
    print("Gaussian noise: std=0.008 normalized units, applied to all points (~0.4 m before scaling)")
    print("Point dropout: remove_prob=0.15, about 15% of points removed")
    print("Outliers: num_outliers=30 per 2048-point chunk, about 1.46% added outliers")


if __name__ == "__main__":
    main()
