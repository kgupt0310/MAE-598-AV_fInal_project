#!/bin/bash
#SBATCH -J lidar_seq00
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-08:00:00
#SBATCH -p public
#SBATCH -q class
#SBATCH -G 1
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err

set -e

module purge
module load mamba/latest

# Create this environment once on Sol:
# mamba create -y -n lidar310 -c conda-forge python=3.10 numpy scipy scikit-learn matplotlib pip
# source activate lidar310
# python -m pip install torch
source activate lidar310

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export TORCH_NUM_THREADS=8

cd /home/dnakum/you-are-helping-us-continue-an

DATA_ROOT="/home/dnakum/dataset/dataset/sequences"
MODEL_PATH="experiments/dgcnn_seq00_first51_fps30_sol.pth"
HISTORY_PATH="experiments/dgcnn_seq00_first51_fps30_sol_history.csv"

echo "Project directory: $(pwd)"
echo "Data root: ${DATA_ROOT}"
echo "Checking Python environment..."
python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
PY

echo "Starting training..."
python -u src/train/train_dgcnn.py \
  --data-root "${DATA_ROOT}" \
  --sequences "00" \
  --epochs 30 \
  --max-frames-per-seq 51 \
  --start-frame 0 \
  --chunks-per-frame 30 \
  --center-selection fps_xy \
  --max-batches 0 \
  --num-points 2048 \
  --num-workers 2 \
  --save-name "$(basename "${MODEL_PATH}")" \
  --history-name "$(basename "${HISTORY_PATH}")"

echo "Plotting training history..."
python -u scripts/plot_training_history.py \
  --history "${HISTORY_PATH}" \
  --save "experiments/dgcnn_seq00_first51_fps30_sol_training_history.png"

echo "Evaluating held-out frames 000051 to 000065..."
python -u test/evaluate_dgcnn_test_frames.py \
  --data-root "${DATA_ROOT}" \
  --model-path "${MODEL_PATH}" \
  --sequence 00 \
  --start-frame 51 \
  --num-frames 15 \
  --chunks-per-frame 30 \
  --center-selection fps_xy \
  --save-csv "experiments/seq00_test15_metrics_sol.csv"

echo "Saving one visual result from first held-out test frame..."
python -u test/test_dgcnn_full_cloud_visualization.py \
  --data-root "${DATA_ROOT}" \
  --model-path "${MODEL_PATH}" \
  --sequence 00 \
  --frame-index 51 \
  --max-chunks 30 \
  --center-selection fps_xy \
  --save-prefix "experiments/seq00_test_frame_000051_sol" \
  --no-show

echo "Done. Outputs are in experiments/."
