#!/bin/bash
#SBATCH --job-name=stillfast_vit1b
#SBATCH --gpus=a100_80gb:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --time=10:00:00
#SBATCH --output=/cluster/scratch/udemirbas/EgoVideo/logs/stillfast_%j.out
#SBATCH --error=/cluster/scratch/udemirbas/EgoVideo/logs/stillfast_%j.err
#SBATCH --mail-user=udemirbas@ethz.ch
#SBATCH --mail-type=ALL

eval "$(conda shell.bash hook)"
conda activate stillfast2

export MASTER_PORT=$((29500 + RANDOM % 100))

cd /cluster/scratch/udemirbas/EgoVideo/cvpr-2024/stillfast

python main.py \
  --cfg configs/sta/STILL_FAST_R50_vit1b_EGO4D_v2.yaml \
  --train \
  --exp vit1b_train \
  NUM_DEVICES 1 \
  TRAIN.BATCH_SIZE 8 \
  VAL.BATCH_SIZE 16 \
  SOLVER.ACCELERATOR gpu \
  SOLVER.STRATEGY ddp_static_graph \
  DATA_LOADER.NUM_WORKERS 4 \
  EGO4D_STA.STILL_FRAMES_PATH /cluster/work/cvg/data/Ego4d/v2_ud/v2/object_frames \
  EGO4D_STA.FAST_LMDB_PATH /cluster/work/cvg/data/Ego4d/v2_ud/v2/lmdb \
  EGO4D_STA.ANNOTATION_DIR /cluster/work/cvg/data/Ego4d/v2_ud/v2/annotations