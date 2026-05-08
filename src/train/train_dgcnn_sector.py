import argparse
import csv
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.datasets.kitti_sector_dataset import KITTISectorDataset
from src.loss.chamfer_loss import reconstruction_loss
from src.models.dgcnn_reconstruction import DGCNNReconstructor
from src.train.train_dgcnn import DEFAULT_TRAIN_SEQS, choose_device
from src.utils.project_paths import default_data_root, parse_sequences, project_root


def build_parser():
    parser = argparse.ArgumentParser(description="Train optional sector DGCNN model.")
    parser.add_argument("--data-root", default=str(default_data_root()))
    parser.add_argument("--sequences", default=DEFAULT_TRAIN_SEQS)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--num-sectors", type=int, default=16)
    parser.add_argument("--scale", type=float, default=50.0)
    parser.add_argument("--max-frames-per-seq", type=int, default=500)
    parser.add_argument("--max-batches", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-dir", default=str(project_root() / "experiments"))
    parser.add_argument("--save-name", default="dgcnn_sector_motion_attention.pth")
    parser.add_argument("--history-name", default="dgcnn_sector_motion_attention_history.csv")
    return parser


def main():
    args = build_parser().parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(
            f"KITTI data root does not exist: {data_root}. "
            "Pass --data-root or set KITTI_SEQUENCES_DIR."
        )

    device = choose_device(args.device)
    print("Using device:", device)

    max_frames = args.max_frames_per_seq if args.max_frames_per_seq > 0 else None
    max_batches = args.max_batches if args.max_batches > 0 else None

    dataset = KITTISectorDataset(
        root_dir=str(data_root),
        sequences=parse_sequences(args.sequences),
        num_points=args.num_points,
        num_sectors=args.num_sectors,
        max_frames_per_seq=max_frames,
        scale=args.scale,
    )

    if len(dataset) == 0:
        raise RuntimeError("No sector samples were indexed. Check data root and sequence IDs.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    model = DGCNNReconstructor(k=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    save_dir = Path(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    save_path = save_dir / args.save_name
    history_path = save_dir / args.history_name

    history = []

    for epoch in range(args.epochs):
        model.train()

        total_loss = 0.0
        total_chamfer = 0.0
        total_l2 = 0.0
        total_attn = 0.0
        count = 0

        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break

            corrupted = batch[0].to(device)
            clean = batch[1].to(device)
            prev_chunk = batch[2].to(device)

            pred, attn_weights = model(corrupted, prev_chunk)

            loss, parts = reconstruction_loss(
                pred=pred,
                clean=clean,
                corrupted=corrupted,
                attn_weights=attn_weights,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_chamfer += parts["chamfer"]
            total_l2 += parts["l2"]
            total_attn += parts["attn"]
            count += 1

        row = {
            "epoch": epoch + 1,
            "loss": total_loss / max(count, 1),
            "chamfer": total_chamfer / max(count, 1),
            "l2": total_l2 / max(count, 1),
            "attn": total_attn / max(count, 1),
        }
        history.append(row)

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Loss: {row['loss']:.4f} | "
            f"Chamfer: {row['chamfer']:.4f} | "
            f"L2: {row['l2']:.4f} | "
            f"Attn: {row['attn']:.4f}"
        )

    torch.save(model.state_dict(), save_path)

    with open(history_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "chamfer", "l2", "attn"])
        writer.writeheader()
        writer.writerows(history)

    print(f"Saved DGCNN sector model at: {save_path}")
    print(f"Saved training history at: {history_path}")


if __name__ == "__main__":
    main()
