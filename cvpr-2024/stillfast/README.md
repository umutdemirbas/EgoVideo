# StillFast: An End-to-End Approach for Short-Term Object Interaction Anticipation

This is the official github repository of the following publication:

F. Ragusa, G. M. Farinella, A. Furnari. StillFast: An End-to-End Approach for Short-Term Object Interaction Anticipation. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops. 2023.

[project web page](https://iplab.dmi.unict.it/stillfast/) | [paper](https://arxiv.org/abs/2304.03959)


## Citing StillFast Paper
If you find our work useful in your research, please use the following BibTeX entry for citation.
```
 @InProceedings{ragusa2023stillfast,
 author={Francesco Ragusa and Giovanni Maria Farinella and Antonino Furnari},
 title={StillFast: An End-to-End Approach for Short-Term Object Interaction Anticipation}, 
 booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
 year      = {2023}
 }

```

## Installation
### Requirements

#### Anaconda
An Anaconda environment with the requirements is provided in `environment.yml`. If you are using Anaconda, you can create a suitable environment with:

`conda env create -f environment.yml`

Then, activate the environment:

`conda activate stillfast`

#### Pip
We provide a list of libraries in requirements.txt. You can easy install these libraries using pip:

`pip install -r requirements.txt`

### Wandb
Wandb is enabled by default. To use it set the credentials in `wandb/settings`:

```
entity = yournickname
project = yourprojectname
base_url = https://api.wandb.ai
```
Then, login with `wandb login`.


## Model Zoo and Baselines
We provided pretrained models on EGO4D `v1` and `v2`:


| pretraining | Still | Fast | model |  config  |
| ------------- | -------------| ------------- | ------------- | ------------- | 
| EGO4D v1 | ResNet R50 | X3D_M |  [`link`](https://iplab.dmi.unict.it/sharing/StillFast/models/StillFast_EGO4D_v1.ckpt) | configs/sta/STILL_FAST_R50_X3DM_EGO4D_v1.yaml |
| EGO4D v2 | ResNet R50 | X3D_M | [`link`](https://iplab.dmi.unict.it/sharing/StillFast/models/StillFast_EGO4D_v2.ckpt) | configs/sta/STILL_FAST_R50_X3DM_EGO4D_v2.yaml |


## EGO4D Dataset
To train/test the model on the EGO4D dataset, follow the instructions provided here to download the dataset and its annotations for the Short-Term Object Interaction Anticipation task:

`https://github.com/EGO4D/forecasting/blob/main/SHORT_TERM_ANTICIPATION.md`


## Training

To train StillFast on the EGO4D dataset, execute the following command:

`python main.py --cfg configs/sta/STILLFAST_R50_X3DM_EGO4d-V2.yaml --train --exp unique_experiment_name`

Outputs will be logged to wandb and stored under the folder `output/sta/StillFast_unique_experiment_name/version_0/`

If you repeat the command, experiments will be saved under the `version_1` subdirectory and so on.

## Validation
Trained models can be validated using the following command:

`python main.py --val --test_dir output/sta/StillFast_unique_experiment_name/version_x/`

where `x` is the version number of your experiment.
After the validation phase, predictions will be saved in a json file under:

`output/sta/StillFast_unique_experiment_name/version_x/results/val.json`

Results will be printed, but you may obtain the final ones using the official [`evaluate_short_term_anticipation_results.py` script](https://github.com/EGO4D/forecasting/blob/main/SHORT_TERM_ANTICIPATION.md#evaluating-the-results).

You can evaluate the results with the following command:   

`python /path/to/forecasting/tools/short_term_anticipation/evaluate_short_term_anticipation_results.py output/sta/StillFast_unique_experiment_name/version_x/results/val.json /path/to/ego4d/annotations/fho_sta_val.json`

## Test

The `main.py` program also allows to run the model on the EGO4D test set and produce a json file to be sent to the [`leaderboard`](https://eval.ai/web/challenges/challenge-page/1623/leaderboard/3910). To test models, you can use the following commands:

`python main.py --test --test_dir output/sta/StillFast_unique_experiment_name/version_x/`

After the test phase, predictions will be saved in a json file under:

`output/sta/StillFast_unique_experiment_name/version_x/results/test.json`

To obtain results, submit the `test.json` file to the [`EGO4D Short Term Object Interaction Anticipation Challenge page`](https://eval.ai/web/challenges/challenge-page/1623/overview).

---

## ETH Euler Cluster Adaptation (ViT-g/14 Backbone)

The following documents all modifications made to run StillFast with the EgoVideo InternVideo2 ViT-g/14 backbone (config `STILL_FAST_R50_vit1b_EGO4D_v2.yaml`) on the ETH Euler HPC cluster using the `stillfast2` conda environment.

### Architecture

The ViT-g/14 variant replaces the X3D-M fast backbone with InternVideo2 ViT-g/14 (1.3B parameters, embed_dim=1408, depth=40, patch_size=14). Pretrained weights are loaded from `ckpt_4frames.pth`. The still backbone remains ResNet-50.

### Environment (`stillfast2`)

Created via `~/create_stillfast2_env.sh`:

- Python 3.10, PyTorch 2.3.0+cu121
- `flash-attn==2.5.8` (pre-built wheel only; source compilation is not feasible on Euler)
- `pytorch-lightning==2.3.0`, `detectron2` (GitHub), `transformers==4.40.0`

### Pre-downloaded Weights

Euler compute nodes have no internet access. Cache these on the login node before submitting jobs:

```bash
wget https://download.pytorch.org/models/resnet50-0676ba61.pth -P ~/.cache/torch/hub/checkpoints/
wget https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth -P ~/.cache/torch/hub/checkpoints/
```

The EgoVideo checkpoint: `/cluster/scratch/udemirbas/EgoVideo/backbone/model/checkpoint/ckpt_4frames.pth`

### Data Paths

| Data | Path |
|------|------|
| Still frames | `/cluster/work/cvg/data/Ego4d/v2_ud/v2/object_frames` |
| Fast LMDB | `/cluster/work/cvg/data/Ego4d/v2_ud/v2/lmdb` |
| Annotations | `/cluster/work/cvg/data/Ego4d/v2_ud/v2/annotations` |

---

### Code Changes

#### 1. VisionTransformer Stub (NEW FILE)

**File**: `../AVION/avion/models/internvideo2/transformer.py`

The original AVION repo's `VisionTransformer` was in a private fork and unavailable. This stub was created from scratch:

- Subclasses EgoVideo's `PretrainVisionTransformer` (from `backbone/model/vision_encoder.py`) so that checkpoint key names (`blocks.*`, `cls_token`, `pos_embed`, `clip_projector.*`) match `ckpt_4frames.pth` exactly.
- Adds `forward(x, return_f3d=True)` which reshapes transformer output tokens from `[B, T*H'*W'+1, 1408]` to `[B, 1408, T, H', W']` (dropping the cls token), matching what StillFast's 3D FPN expects.
- **Decouples flash attention from fused ops**: independently checks whether `flash_attn` (core) and `DropoutAddRMSNorm` (fused norm) are importable. This allows using flash attention without the fused CUDA kernels:

```python
use_flash_attn=has_flash and use_flash_attn,        # True  (core works)
use_fused_rmsnorm=has_fused_norm and use_fused_rmsnorm,  # False (kernel missing)
use_fused_mlp=has_fused_norm and use_fused_mlp,          # False (kernel missing)
```

#### 2. Backbone Builder

**File**: `stillfast/models/backbone_utils_3d.py`

- Added AVION to `sys.path` and imports for `VisionTransformer`, `CLIP`, `TextTransformer`.
- Patched `CLIP.forward` to call `self.visual(x, return_f3d=True)` directly.
- Set `use_fused_rmsnorm=False` and `use_fused_mlp=False` (keeping `use_flash_attn=True`).
- **Changed `num_frames` from 4 to 8** to match the dataloader config (`NUM_FRAMES: 8`).
- **Added positional embedding interpolation**: the checkpoint has `pos_embed` shaped for 4 frames (1025 tokens = 4x256+1). The model needs 8 frames (2049 tokens). Before `load_state_dict`, `visual.pos_embed` and `visual.clip_pos_embed` are reshaped to `[1, D, T_old, HW]` and bilinearly interpolated along the temporal axis to `T_new=8`, then reassembled with the cls token.
- Added `channels_list = [1408, 1408, 1408, 1408]` for the `vit-1b` backbone name.

#### 3. Vision Encoder Assert

**File**: `../backbone/model/vision_encoder.py` (line 420)

Commented out the assertion:
```python
# assert use_flash_attn == use_fused_rmsnorm == use_fused_mlp
```
This forced all three flags to be equal, which prevented using flash attention without the fused norm/MLP ops.

#### 4. Flash Attention dtype Auto-cast

**File**: `../backbone/model/flash_attention_class.py` (line 37)

Changed the hard assertion:
```python
# Before:
assert qkv.dtype in [torch.float16, torch.bfloat16]

# After:
if qkv.dtype not in [torch.float16, torch.bfloat16]:
    qkv = qkv.to(torch.bfloat16)
```

Gradient checkpointing (`use_checkpoint=True`, all 40 layers) replays the forward pass outside the AMP autocast context, delivering fp32 tensors to flash attention. This auto-cast ensures compatibility.

#### 5. Torchvision Compatibility

**File**: `stillfast/models/faster_rcnn.py` (lines 5-12)

`torchvision.models.detection.faster_rcnn.model_urls` was removed in torchvision >= 0.13. Added a try/except with a hardcoded fallback dictionary:

```python
try:
    from torchvision.models.detection.faster_rcnn import model_urls
except ImportError:
    model_urls = {
        "fasterrcnn_resnet50_fpn_coco": "https://...258fb6c6.pth",
        "fasterrcnn_mobilenet_v3_large_320_fpn_coco": "https://...907ea3f9.pth",
        "fasterrcnn_mobilenet_v3_large_fpn_coco": "https://...fb6a5cc7.pth",
    }
```

#### 6. PyTorch Lightning 2.3 — Trainer Args

**File**: `main.py` (lines 72-73)

Two deprecated `Trainer` arguments updated:

| Old (PL 1.x) | New (PL 2.x) |
|---|---|
| `replace_sampler_ddp=...` | `use_distributed_sampler=...` |
| `precision=16` | `precision="16-mixed"` |

#### 7. PyTorch Lightning 2.3 — Epoch End Hooks

**File**: `stillfast/tasks/sta.py`

The `*_epoch_end(self, outs)` hooks were removed in PL 2.0. Migrated to the new accumulation pattern:

- Added `self._val_outputs = []` and `self._test_outputs = []` in `__init__`.
- `validation_step` now appends to `self._val_outputs` instead of returning a dict.
- `validation_epoch_end(self, outs)` renamed to `on_validation_epoch_end(self)`, reads from `self._val_outputs`, and calls `self._val_outputs.clear()` at the end.
- Same pattern applied to `test_step` / `on_test_epoch_end` with `self._test_outputs`.

---

### Running on Euler

#### Quick Smoke Test (interactive A100 node)

```bash
srun --gpus=a100_80gb:1 --mem-per-cpu=32G --cpus-per-task=1 --time=00:30:00 --pty bash
eval "$(conda shell.bash hook)"
conda activate stillfast2
cd /cluster/scratch/udemirbas/EgoVideo/cvpr-2024/stillfast

python main.py \
  --cfg configs/sta/STILL_FAST_R50_vit1b_EGO4D_v2.yaml \
  --train --fast_dev_run \
  NUM_DEVICES 1 TRAIN.BATCH_SIZE 1 DATA_LOADER.NUM_WORKERS 1 \
  EGO4D_STA.STILL_FRAMES_PATH /cluster/work/cvg/data/Ego4d/v2_ud/v2/object_frames \
  EGO4D_STA.FAST_LMDB_PATH /cluster/work/cvg/data/Ego4d/v2_ud/v2/lmdb \
  EGO4D_STA.ANNOTATION_DIR /cluster/work/cvg/data/Ego4d/v2_ud/v2/annotations
```

**Note**: Requires A100-80GB. RTX 4090 (24GB) will OOM during validation.

#### Full Training (SLURM batch)

```bash
sbatch scripts/run_stillfast_train.sh
```

See `scripts/run_stillfast_train.sh` for configuration (1x A100-80GB, 24h wall time).

---

### Known Limitations

1. **Fused ops unavailable**: `DropoutAddRMSNorm` and `FusedMLP` require the `dropout_layer_norm` CUDA kernel which is only available when `flash-attn` is compiled from source. Source compilation fails on Euler because the conda CUDA toolkit's CCCL headers are incompatible with nvcc 12.1. Impact: ~15-20% throughput loss on norm/MLP layers. Core flash attention (the main 2-4x speedup) works fine from the pre-built wheel.

2. **Positional embedding interpolation**: The checkpoint was trained with 4 frames but the model uses 8 frames. Temporal positional embeddings are bilinearly interpolated, which is standard practice but may cause minor accuracy differences until fine-tuning adapts the embeddings.

3. **cuDNN warnings**: `CUDNN_STATUS_NOT_SUPPORTED` warnings on Conv3d operations are benign — cuDNN automatically falls back to a supported algorithm.
