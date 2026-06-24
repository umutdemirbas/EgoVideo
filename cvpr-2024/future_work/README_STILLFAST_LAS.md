# StillFast (EgoVideo-V) + LAS Latent Model for Ego4D STA

Two overlapping 8-frame sequences with temporal difference computation in LAS encoder space.

## Architecture

### Core Concept
```
9 Total Frames
├─ Sequence 1: [Frame 0, 1, 2, 3, 4, 5, 6, 7]  → Still Frame 7 + Fast Frames [0-7]
└─ Sequence 2: [Frame 1, 2, 3, 4, 5, 6, 7, 8]  → Still Frame 8 + Fast Frames [1-8]

Processing Pipeline:
        Sequence 1              Sequence 2
         (8 frames)              (8 frames)
            ↓                        ↓
      StillFast:                StillFast:
      Still 7 (2D)             Still 8 (2D)
      Fast [0-7] (3D)          Fast [1-8] (3D)
            ↓                        ↓
      Features_1               Features_2
      [B, 256, D]             [B, 256, D]
            ↓                        ↓
         Project (if needed)
         [B, 256, D_encoder]    [B, 256, D_encoder]
            ↓                        ↓
      LAS forward_encoder   LAS forward_encoder
            ↓                        ↓
      Encoded_1 [B,256,D]    Encoded_2 [B,256,D]
            └────────┬─────────────┘
                     ↓
        ⭐ Temporal Difference ⭐
        action_patches = Encoded_2 - Encoded_1
                     ↓
              MAP Block
          Action Latents [B, 4*128]
                     ↓
           STA Heads → Predictions
```

## Key Design Decisions

### Why 9 Frames?
- Sequence 1: 8 frames [0-7]
- Sequence 2: 8 frames [1-8] (offset by 1)
- Allows computing temporal difference with 1-frame granularity
- Matches the paper's approach for capturing action changes

### Why Still Frame = Last Frame?
From StillFast paper:
- Still branch (2D ResNet50): Captures high-res spatial details of the **current** moment
- Fast branch (3D EgoVideo-V): Captures temporal dynamics leading to that moment
- Using Frame 7 (last) for Seq1 and Frame 8 (last) for Seq2 gives most recent spatial state

### Why Difference After Encoder?
Matches your paper's approach:
- Process both sequences independently through LAS encoder
- Compute difference in **encoder feature space**: `Encoded_Seq2 - Encoded_Seq1`
- This captures what changed in the learned representation (more semantic than raw frame differences)
- MAP block then pools these differences into action latents

### Why No Projection?
- Use StillFast features **naturally** without forced projection
- If StillFast outputs don't match LAS encoder dimension, apply adaptive projection
- Otherwise pass through directly to LAS

## Implementation Files

### Core Model: `stillfast/models/egovideo_latent_sta.py`
```python
class StillFastLatentSTA(nn.Module):
    # Processes 9 frames into two 8-frame sequences
    # Extracts natural StillFast features
    # Passes each through LAS encoder
    # Computes temporal difference
    # Maps to STA predictions
```

### Task: `stillfast/tasks/egovideo_latent_sta.py`
```python
class StillFastLatentSTATask(BaseTask):
    # PyTorch Lightning training/validation/test wrapper
    # Handles metrics and checkpointing
```

### Configuration: `configs/stillfast_latent_sta.yaml`
```yaml
DATA.FAST.NUM_FRAMES: 9  # ⚠️ Critical: 9 frames for two 8-frame sequences
MODEL.STILLFAST.FREEZE: True
MODEL.LATENT.FREEZE_ENCODER: False
```

## Data Format

**Input batch:**
```python
batch = {
    'video': torch.randn(B, 9, 3, H, W),           # 9 frames total
    'boxes': torch.randn(B, N, 4),                 # Bounding boxes
    'verb_labels': torch.randint(0, 81, (B, N)),   # Ground truth verbs
    'noun_labels': torch.randint(0, 128, (B, N)),  # Ground truth nouns
    'ttc_targets': torch.randn(B, N),              # Time-to-contact targets
}
```

**Training output:**
```python
{
    'loss_verb': Tensor,      # Cross-entropy loss for verb prediction
    'loss_noun': Tensor,      # Cross-entropy loss for noun prediction
    'loss_ttc': Tensor,       # Smooth L1 loss for TTC regression
    'loss': Tensor            # Total weighted loss
}
```

**Inference output:**
```python
[
    {
        'boxes': [N, 4],              # Bounding boxes
        'verbs': [N],                 # Verb class predictions (argmax)
        'nouns': [N],                 # Noun class predictions (argmax)
        'ttcs': [N, 1],               # Time-to-contact values
        'scores': [N]                 # Confidence scores
    },
    ...  # One dict per batch item
]
```

## Forward Pass Flow

```python
# 1. Extract sequences
seq1_frames = video[:, :8]      # [B, 8, 3, H, W]
seq2_frames = video[:, 1:9]     # [B, 8, 3, H, W]
still_1 = seq1_frames[:, -1]    # [B, 3, H, W] - Frame 7
still_2 = seq2_frames[:, -1]    # [B, 3, H, W] - Frame 8

# 2. StillFast processing (natural features, no projection required)
features_seq1 = stillfast(still_1, seq1_frames)  # [B, C, H, W] or [B, C, T, H, W]
features_seq2 = stillfast(still_2, seq2_frames)  # [B, C, H, W] or [B, C, T, H, W]

# 3. Reshape to patch representation
feat_seq1 = spatial_pooling(features_seq1)  # [B, 256, D]
feat_seq2 = spatial_pooling(features_seq2)  # [B, 256, D]

# 4. Project if needed (adaptive based on actual dimension)
if D != encoder_dim:
    feat_seq1 = proj(feat_seq1)  # [B, 256, encoder_dim]
    feat_seq2 = proj(feat_seq2)  # [B, 256, encoder_dim]

# 5. LAS encoder processing (each sequence separately)
encoded_seq1 = las.forward_encoder(feat_seq1.unsqueeze(1), pad_mask)  # [256, encoder_dim]
encoded_seq2 = las.forward_encoder(feat_seq2.unsqueeze(1), pad_mask)  # [256, encoder_dim]

# 6. Temporal difference in encoder space (⭐ Key step)
action_patches = encoded_seq2 - encoded_seq1  # [256, encoder_dim]

# 7. Action latent pooling
action_features = las.transE_MAP_block(action_patches)  # [B, n_tokens, latent_dim]
action_features = flatten(action_features)              # [B, n_tokens*latent_dim]

# 8. STA predictions
verb_logits, noun_logits, ttc_values, scores = sta_heads(action_features)
```

## Configuration Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `NUM_FRAMES` | 9 | Two 8-frame sequences with 1-frame offset |
| `STILLFAST.FREEZE` | True | Keep pretrained StillFast fixed |
| `LATENT.FREEZE_ENCODER` | False | Allow LAS encoder fine-tuning |
| `STA_LOSS_WEIGHTS` | [1.0, 1.0, 1.0] | Equal loss balance |

## Training

```bash
python main.py --config configs/stillfast_latent_sta.yaml
```

With custom parameters:
```bash
python main.py --config configs/stillfast_latent_sta.yaml \
               NUM_DEVICES=4 \
               SOLVER.BASE_LR=0.0002 \
               SOLVER.MAX_EPOCH=30
```

## Key Code Locations

**Where sequences are split:**
```python
# stillfast/models/egovideo_latent_sta.py, forward()
seq1_frames = video[:, :8]   # Frames 0-7
seq2_frames = video[:, 1:9]  # Frames 1-8
still_1 = seq1_frames[:, -1] # Frame 7
still_2 = seq2_frames[:, -1] # Frame 8
```

**Where temporal difference is computed:**
```python
# After LAS encoder processing
encoded_seq1_list = self.latent_model.forward_encoder(feat_seq1_input, pad_mask)
encoded_seq2_list = self.latent_model.forward_encoder(feat_seq2_input, pad_mask)
# ⭐ Difference in encoder space
action_patches = encoded_seq2 - encoded_seq1
# MAP block aggregates into action latents
action_features = self.latent_model.transE_MAP_block(action_patches)
```

## Model Registration

**In `stillfast/models/build.py`:**
```python
from .egovideo_latent_sta import StillFastLatentSTA
if name == "stillfast_latent_sta":
    return StillFastLatentSTA(cfg)
```

**In `main.py`:**
```python
from stillfast.tasks.egovideo_latent_sta import StillFastLatentSTATask
elif cfg.TASK == "stillfast_latent_sta":
    TaskType = StillFastLatentSTATask
```

## Loss Computation

```python
verb_loss = CrossEntropyLoss(verb_logits, verb_labels)
noun_loss = CrossEntropyLoss(noun_logits, noun_labels)
ttc_loss = SmoothL1Loss(ttc_values, ttc_targets) [only for valid targets]

total_loss = 1.0 * verb_loss + 1.0 * noun_loss + 1.0 * ttc_loss
```

All losses are computed after expanding predictions to match number of boxes.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | Verify `MODEL.NAME: 'stillfast_latent_sta'` in config |
| Dimension mismatch | Check `NUM_FRAMES: 9` in DATA.FAST config |
| OOM errors | Reduce `NUM_DEVICES` or batch size |
| Poor performance | Set `FREEZE_ENCODER: False` to fine-tune LAS |
| NaN loss | Check data normalization and label ranges |

## Files Status

✅ `stillfast/models/egovideo_latent_sta.py` - Model implementation (ready)
✅ `stillfast/tasks/egovideo_latent_sta.py` - Task wrapper (ready)
✅ `configs/stillfast_latent_sta.yaml` - Configuration (ready)
✅ `stillfast/models/build.py` - Registry updated (ready)
✅ `main.py` - Task dispatcher updated (ready)

## References

- **StillFast Paper**: Multi-branch architecture with Still (2D) and Fast (3D) branches
- **LAS Paper**: Latent action sequences with temporal difference computation
- **Ego4D STA**: Short-term action anticipation task with verb/noun/TTC targets
