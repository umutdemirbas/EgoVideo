# Option 1: EgoVideo Backbone + LAS Latent Head for Ego4D STA

## Overview

This is **Option 1** implementation: Sequential integration of EgoVideo backbone with LAS latent model for Short-Term Action Anticipation (STA) task on Ego4D dataset.

```
Video Input [B, C, T, H, W]
    ↓
EgoVideo Backbone (pretrained, frozen)
    ↓
Vision Features [B, 512]
    ↓
Project to LAS Dimension [B, 768]
    ↓
LAS Latent Encoder
    ↓
Action Features [B, 128]
    ↓
┌──────────┬──────────┬─────────┬──────────┐
Verb Head  Noun Head  TTC Head  Score Head
    ↓          ↓          ↓          ↓
[B, 81]    [B, 128]    [B, 1]     [B]
    ↓          ↓          ↓          ↓
Verb Class Noun Class TTC Values Confidence
```

## Files

### New Implementation Files
- **`stillfast/models/egovideo_latent_sta.py`**
  - `STAHeads`: Prediction heads for verb, noun, ttc, and confidence
  - `EgoVideoLatentSTA`: Main model combining EgoVideo + LAS

- **`stillfast/tasks/egovideo_latent_sta.py`**
  - `EgoVideoLatentSTATask`: PyTorch Lightning task for training/eval/testing

- **`configs/egovideo_latent_sta.yaml`**
  - Configuration with all hyperparameters

### Updated Files
- **`stillfast/models/build.py`** - Model registry
- **`main.py`** - Task dispatcher

## Usage

### 1. Prerequisites

```bash
# Ensure EgoVideo backbone checkpoint is available
ls /path/to/ckpt_4frames.pth

# Ensure LAS config is available
ls /cluster/scratch/udemirbas/LAS/configs/latent/model_config.yaml
```

### 2. Configure

Edit `configs/egovideo_latent_sta.yaml`:

```yaml
MODEL:
  EGOVIDEO:
    CKPT_PATH: /path/to/ckpt_4frames.pth  # Update this!
    NUM_FRAMES: 4
    FREEZE: True
  
  LATENT:
    CONFIG_PATH: /cluster/scratch/udemirbas/LAS/configs/latent/model_config.yaml
    FREEZE_ENCODER: False
```

Also update data paths:
```yaml
EGO4D_STA:
  STILL_FRAMES_PATH: ../data/short_term_anticipation/still_img_v1
  FAST_LMDB_PATH: ../data/short_term_anticipation/fast_lmdb_v1_6s/
  ANNOTATION_DIR: ../ego4d_data/v1/annotations/
```

### 3. Train

```bash
cd /cluster/scratch/udemirbas/EgoVideo/cvpr-2024/stillfast

python main.py \
  --config configs/egovideo_latent_sta.yaml \
  --output-dir ./outputs/egovideo_latent_sta \
  --num-devices 4
```

### 4. Test

```bash
python main.py \
  --config configs/egovideo_latent_sta.yaml \
  --checkpoint ./outputs/egovideo_latent_sta/checkpoints/best.ckpt
```

## Architecture Details

### Model Flow

1. **EgoVideo Backbone**
   - Input: Video frames [B, 3, T, 224, 224]
   - Output: Vision features [B, 512]
   - Status: Frozen (no gradients)
   - Purpose: Extract rich egocentric visual features

2. **Projection Layer**
   - Input: EgoVideo features [B, 512]
   - Output: Projected features [B, 768]
   - Purpose: Bridge EgoVideo to LAS latent dimension

3. **LAS Latent Encoder**
   - Input: Projected features [B, 768]
   - Output: Action features [B, 128]
   - Status: Fine-tunable (can be frozen if desired)
   - Purpose: Extract action-specific representations

4. **STA Prediction Heads**
   - **Verb Head**: Predicts action verb (81 classes)
   - **Noun Head**: Predicts object noun (128 classes)
   - **TTC Head**: Predicts time-to-contact (regression)
   - **Score Head**: Predicts confidence (0-1)

### Loss Computation

```
Total Loss = w_verb * CrossEntropy(verb_pred, verb_label)
           + w_noun * CrossEntropy(noun_pred, noun_label)
           + w_ttc * SmoothL1(ttc_pred, ttc_target)
```

Default weights: `[1.0, 1.0, 1.0]`

## Key Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EGOVIDEO.CKPT_PATH` | - | Path to EgoVideo checkpoint |
| `EGOVIDEO.NUM_FRAMES` | 4 | Number of frames for EgoVideo |
| `EGOVIDEO.FREEZE` | True | Freeze EgoVideo backbone |
| `LATENT.FREEZE_ENCODER` | False | Freeze LAS encoder |
| `SOLVER.BASE_LR` | 0.0001 | Learning rate |
| `SOLVER.MAX_EPOCH` | 20 | Training epochs |
| `TRAIN.BATCH_SIZE` | 8 | Batch size |

## Performance Characteristics

| Aspect | Value |
|--------|-------|
| Model Size | ~1.2 GB (both backbones) |
| Inference Speed | Medium (two forward passes) |
| Training Speed | Medium |
| Memory Usage | High (requires GPU with >16GB) |
| Expected Accuracy | High (combines two strong models) |

## Training Tips

### For Better Convergence
- Start with `FREEZE_ENCODER: True` in LAS, then unfreeze
- Use warmup: `WARMUP_STEPS: 4000`
- Monitor `loss_verb`, `loss_noun`, `loss_ttc` separately

### For Memory Efficiency
- Reduce `TRAIN.BATCH_SIZE` (try 4 or 8)
- Enable mixed precision: `PRECISION: 16` (already enabled)
- Use `EGOVIDEO.FREEZE: True` (default)

### For Better Performance
- Increase `MAX_EPOCH` to 50-100
- Fine-tune learning rate based on dataset
- Unfreeze `LATENT.FREEZE_ENCODER` after initial training
- Experiment with `STA_LOSS_WEIGHTS`

## Troubleshooting

### Out of Memory
```yaml
TRAIN:
  BATCH_SIZE: 4  # Reduce batch size
```

### EgoVideo not loading
```bash
# Verify checkpoint exists
ls /path/to/ckpt_4frames.pth

# Update path in config
EGOVIDEO:
  CKPT_PATH: /correct/path/ckpt_4frames.pth
```

### LAS model not loading
```bash
# Verify LAS config exists
ls /cluster/scratch/udemirbas/LAS/configs/latent/model_config.yaml

# Check LAS dependencies
pip list | grep latent
```

### Poor convergence
- Check that labels are in correct range (verb: 0-80, noun: 0-127)
- Verify batch size is reasonable (8-16)
- Try lower learning rate: 5e-5 or 1e-5

## Output Format

Test predictions are saved in Ego4D format:
```json
{
  "version": "1.0",
  "challenge": "ego4d_short_term_object_interaction_anticipation",
  "results": {
    "video_uid": [
      {
        "box": [x1, y1, x2, y2],
        "score": 0.95,
        "noun_category_id": 42,
        "verb_category_id": 15,
        "time_to_contact": 1.2
      }
    ]
  }
}
```

## Citation

If you use this implementation, please cite:

```bibtex
@article{pei2024egovideo,
  title={EgoVideo: Exploring Egocentric Foundation Model and Downstream Adaptation},
  author={Pei et al.},
  journal={arXiv preprint arXiv:2406.18070},
  year={2024}
}

@inproceedings{ragusa2023stillfast,
  author={Francesco Ragusa and Giovanni Maria Farinella and Antonino Furnari},
  title={StillFast: An End-to-End Approach for Short-Term Object Interaction Anticipation},
  booktitle={CVPR Workshops},
  year={2023}
}
```

## References

- [EgoVideo Technical Report](https://arxiv.org/abs/2406.18070)
- [Ego4D Challenge](https://ego4d-data.org/)
- [StillFast Paper](https://arxiv.org/abs/2304.03959)
