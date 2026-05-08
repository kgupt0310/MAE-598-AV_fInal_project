# LiDAR Point Cloud Reconstruction / Denoising

Chunk-based KITTI Velodyne denoising/reconstruction using DGCNN EdgeConv blocks with motion-aware attention.

The main pipeline is:

```text
clean KITTI LiDAR chunk
    -> artificial corruption
    -> DGCNN + motion-aware attention
    -> reconstructed chunk
    -> compare against clean chunk
```

This repository intentionally trains on fixed-size chunks, not full raw frames. `NUM_POINTS=2048` means each training sample is a 2048-point chunk.

## Setup

Install dependencies in your Python environment:

```bash
pip install -r requirements.txt
```

Place KITTI sequence data in one of these locations:

```text
data/dataset/sequences
```

or point scripts to your dataset with:

```powershell
$env:KITTI_SEQUENCES_DIR="D:\path\to\dataset\sequences"
```

Expected layout:

```text
sequences/
  00/
    velodyne/
      *.bin
  01/
    velodyne/
      *.bin
```

## Raw LiDAR Sanity Check

Use this before model visualization. It loads raw, unnormalized LiDAR:

```bash
python test/test_raw_kitti_only.py --data-root data/dataset/sequences --sequence 09
```

## Train Main Model

The main model is DGCNN with motion-aware attention using kNN chunks:

```bash
python src/train/train_dgcnn.py --data-root data/dataset/sequences
```

The checkpoint is saved by default to:

```text
experiments/dgcnn_motion_attention.pth
```

## Visualize Reconstruction

```bash
python test/test_dgcnn_full_cloud_visualization.py --data-root data/dataset/sequences --sequence 09
```

The visualization compares corrupted chunks, reconstructed chunks, clean chunks, and overlays in 3D and BEV.

## Optional Sector Experiment

Sector chunking is an optional experiment for LiDAR-sweep-like chunks:

```bash
python src/train/train_dgcnn_sector.py --data-root data/dataset/sequences
```

The current best framing is scalable local LiDAR denoising/reconstruction, not perfect full-scene recovery.
