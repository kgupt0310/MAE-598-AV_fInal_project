import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_history(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="Plot DGCNN training history CSV.")
    parser.add_argument("--history", required=True)
    parser.add_argument("--save", required=True)
    args = parser.parse_args()

    rows = read_history(args.history)
    if not rows:
        raise RuntimeError(f"No rows found in {args.history}")

    epochs = [int(row["epoch"]) for row in rows]
    loss = [float(row["loss"]) for row in rows]
    chamfer = [float(row["chamfer"]) for row in rows]
    l2 = [float(row["l2"]) for row in rows]
    attn = [float(row["attn"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(epochs, loss, label="Total loss", linewidth=2)
    axes[0].plot(epochs, chamfer, label="Chamfer", linewidth=2)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, l2, label="L2", linewidth=2)
    axes[1].plot(epochs, attn, label="Attention term", linewidth=2)
    axes[1].set_title("Auxiliary Terms")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    plt.tight_layout()

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    print(f"Saved history plot: {save_path}")


if __name__ == "__main__":
    main()
