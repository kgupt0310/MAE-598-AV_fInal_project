#!/bin/bash
set -e

module purge
module load mamba/latest

mamba create -y -n lidar310 -c conda-forge python=3.10 numpy scipy scikit-learn matplotlib pip
source activate lidar310
python -m pip install torch

python - <<'PY'
import torch
import numpy
import sklearn
print("Environment ready")
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
PY
