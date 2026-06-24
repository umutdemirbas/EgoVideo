#!/bin/bash
# Create a new conda env for StillFast + EgoVideo (vit-1b backbone)

set -e

ENV_NAME=stillfast2

conda create -n $ENV_NAME python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# PyTorch 2.3 + CUDA 12.1
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# flash_attn (compiles from source, takes ~10 min)
pip install flash-attn --no-build-isolation

# detectron2
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Core StillFast dependencies (from requirements.txt)
pip install pytorch-lightning==2.3.0 torchmetrics
pip install lmdb fvcore iopath portalocker
pip install yacs omegaconf hydra-core
pip install wandb tensorboard
pip install einops timm

# Data / vision
pip install pycocotools opencv-python Pillow
pip install scipy pandas numpy matplotlib

# Misc utilities from requirements.txt
pip install tqdm tabulate psutil pydot
pip install GitPython setproctitle shortuuid

echo "Done. Activate with: conda activate $ENV_NAME"
