# Training Guide

## Quick Start

```bash
# Install dependencies
pip install -e ".[train]"

# Train baseline (testing/debugging)
python -m src.distillation.train --model baseline --offline --max-epochs 2

# Train reasoning model (recommended)
python -m src.distillation.train --model reasoning \
    --reasoning-weight 0.25 --batch-size 32 --accumulation-steps 2
```

## Data Preparation

### Pre-extract Frames Locally

Recommended for training. Better performance than streaming.

#### 1. Extract Frames from HuggingFace

```bash
# Extract train split (e.g., 2K samples)
python scripts/extract_frames.py --split train --resolution 224 --num-samples 2000

# Extract validation split
python scripts/extract_frames.py --split val --resolution 224 --num-samples 1000
```

Output: `extracted_frames/{split}/sample_XXXXX.npz` containing 4 cameras × 4 frames.

#### 2. Generate Teacher Labels (Requires GPU)

Requires Alpamayo-R1 model (~24GB VRAM):

```bash
# Generate trajectory labels and CoC embeddings
python scripts/generate_teacher_labels.py --split train --batch-size 4
python scripts/generate_teacher_labels.py --split val --batch-size 4
```

Output: `teacher_labels/{split}_labels.npz` with trajectories and embeddings.

#### 3. Generate Ego History (Required)

Provides velocity context for unicycle kinematic integration:

```bash
python scripts/generate_ego_history.py --split all
```

Output: `teacher_labels/{split}_ego_history.npz` with 16 past poses at 10Hz (1.6s history)

#### 4. Merge Shards (If Using Multiple GPUs)

```bash
python scripts/merge_shards.py --split train --num-shards 4
```

## CLI Reference

### Model Selection

| Argument | Options | Default | Description |
|----------|---------|---------|-------------|
| `--model` | `baseline`, `reasoning` | required | Student model type |
| `--model-size` | `110m`, `500m` | `500m` | Model configuration preset |

**Model Size Comparison:**

| Size | Backbone | Params | Train Speed | Use Case |
|------|----------|--------|-------------|----------|
| `110m` | ResNet-50 | ~107M | ~3x faster | Hyperparameter sweeps |
| `500m` | ResNet-101 | ~485M | Baseline | Final training |

### Data Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--train-labels` | `teacher_labels/train_labels.npz` | Training labels path |
| `--val-labels` | `teacher_labels/val_labels.npz` | Validation labels path |
| `--frames-dir` | `extracted_frames` | Pre-extracted frames directory |
| `--offline` | false | Use dummy images (for testing) |
| `--ego-history-dir` | `teacher_labels` | Ego history npz files (required) |

### Training Hyperparameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--max-epochs` | 150 | Maximum training epochs |
| `--patience` | 15 | Early stopping patience |
| `--batch-size` | 48 | Batch size per GPU |
| `--accumulation-steps` | 1 | Gradient accumulation |
| `--lr` | 1e-4 | Learning rate |
| `--weight-decay` | 0.02 | AdamW weight decay |
| `--warmup-epochs` | 5 | LR warmup epochs |
| `--lr-schedule` | `cosine` | LR schedule: `cosine` or `constant` |
| `--ema` | false | Enable EMA for model weights |
| `--ema-decay` | 0.999 | EMA decay rate |

### Loss Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--yaw-weight` | 0.3 | Heading loss weight |
| `--reasoning-weight` | 0.25 | Reasoning loss weight (λ) |
| `--reasoning-warmup-epochs` | 20 | λ warmup epochs |
| `--temperature` | 0.07 | Contrastive temperature |
| `--action-weight` | 0.0 | Weight for action supervision |
| `--grad-clip` | 0.0 | Gradient clipping max norm (0=disabled) |

### Discrete Action Mode (unicycle only)

| Argument | Default | Description |
|----------|---------|-------------|
| `--discrete-actions` | false | Use discrete action tokens |
| `--num-action-bins` | 256 | Number of bins per action dimension |
| `--label-smoothing` | 0.1 | Label smoothing for cross-entropy loss |

### Training Mode (Direct Trajectory vs Unicycle)

| Argument | Default | Description |
|----------|---------|-------------|
| `--no-unicycle` | false | Direct trajectory mode: output (x, y, yaw) directly |

**Direct trajectory mode** (`--no-unicycle`):
- Model outputs (x, y, sin_yaw, cos_yaw) directly
- Simpler but less physically constrained
- Ego history still used for velocity context
- Use for ablation studies

**Unicycle mode** (default): Unicycle kinematic integration
- Model outputs (acceleration, curvature) actions
- Actions integrated through unicycle dynamics for trajectory
- Ego history required for velocity context (16 steps at 10Hz)
- Produces kinematically feasible trajectories
- Supports discrete action tokens

### Diagnostic Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--freeze-bn` | false | Freeze BatchNorm stats (backbone in eval mode) |
| `--overfit-tiny` | 0 | Use only N samples for train/val (debugging) |
| `--max-train-samples` | 0 | Limit training samples (0=unlimited) |
| `--max-val-samples` | 0 | Limit validation samples (0=unlimited) |
| `--log-action-stats` | false | Log action statistics during validation |

### Output Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--output-dir` | `checkpoints` | Checkpoint directory |
| `--save-every` | 5 | Save checkpoint interval |
| `--save-final-only` | false | Only save final best model (no intermediate checkpoints) |
| `--seed` | 42 | Random seed |
| `--compile` | false | Use torch.compile for faster training |
| `--resume` | none | Path to checkpoint to resume from |

## Recommended Configuration

### Unicycle Mode (Default) - Stabilized

```bash
python -m src.distillation.train \
    --model baseline \
    --model-size 500m \
    --lr 1e-5 \
    --warmup-epochs 0 \
    --lr-schedule cosine \
    --grad-clip 1.0 \
    --ema --ema-decay 0.999 \
    --action-weight 0.1 \
    --batch-size 32 \
    --max-epochs 50 \
    --patience 15 \
    --seed 42
```

Key stabilizers:
- `--grad-clip 1.0`: Prevents gradient explosions
- `--ema --ema-decay 0.999`: Smooths weight updates
- `--action-weight 0.1`: Direct action supervision aids unicycle optimization

### Fast Iteration (110M)

```bash
python -m src.distillation.train \
    --model baseline \
    --model-size 110m \
    --lr 5e-6 \
    --warmup-epochs 0 \
    --lr-schedule cosine \
    --grad-clip 1.0 \
    --ema --ema-decay 0.999 \
    --action-weight 0.1 \
    --batch-size 32 \
    --max-epochs 50 \
    --patience 15
```

Note: 110M model needs lower LR than 500M (~5e-6 vs 1e-5).

### With Reasoning Supervision

```bash
python -m src.distillation.train \
    --model reasoning \
    --reasoning-weight 0.25 \
    --yaw-weight 0.3 \
    --batch-size 48 \
    --lr 1e-4 \
    --weight-decay 0.02 \
    --max-epochs 150 \
    --patience 20 \
    --seed 42
```

Note: The model uses gradient checkpointing automatically. Use `--accumulation-steps` if you need larger effective batch sizes.

### Direct Trajectory Mode

```bash
python -m src.distillation.train \
    --model reasoning \
    --no-unicycle \
    --reasoning-weight 0.25 \
    --yaw-weight 0.3 \
    --batch-size 48 \
    --lr 1e-4 \
    --max-epochs 150 \
    --seed 42
```

Note: Direct trajectory mode still uses ego history for velocity context. Use this for ablation studies comparing trajectory output modes.

### Benchmark Training (nuScenes / Argoverse 2)

Train on standard benchmarks for comparison with other methods:

```bash
# nuScenes (raster BEV input)
python -m src.distillation.train \
    --model baseline \
    --benchmark nuscenes \
    --nuscenes-root /path/to/nuscenes \
    --input-type raster \
    --num-modes 6 \
    --batch-size 32

# Argoverse 2 (vector input)
python -m src.distillation.train \
    --model baseline \
    --benchmark argoverse2 \
    --argoverse2-root /path/to/av2 \
    --input-type vector \
    --num-modes 6 \
    --batch-size 32
```

Benchmark CLI options:

| Argument | Default | Description |
|----------|---------|-------------|
| `--benchmark` | none | Benchmark format: `nuscenes` or `argoverse2` |
| `--input-type` | `camera` | Input modality: `camera`, `raster`, or `vector` |
| `--num-modes` | 1 | Number of prediction modes (1 or 6) |
| `--multimodal-loss` | `wta` | Multi-modal loss: `wta` or `nll` |
| `--nuscenes-root` | none | Path to nuScenes dataset root |
| `--argoverse2-root` | none | Path to Argoverse 2 dataset root |

## Training Tips

### Memory Optimization

- **Reduce batch size**: Start with 8, increase if memory allows
- **Gradient accumulation**: Use `--accumulation-steps 4` for effective batch 32
- **Gradient checkpointing**: Enabled by default for the 500M model

### Monitoring

Training logs to stdout with:
- Per-epoch loss breakdown (position, yaw, reasoning)
- Validation metrics (ADE, FDE, yaw error)
- Early stopping progress

Checkpoints saved to `checkpoints/`:
- `{model}_seed{seed}_best.pt` - Best validation loss
- `{model}_seed{seed}_latest.pt` - Most recent
- `{model}_seed{seed}_epoch{N}.pt` - Periodic saves

### Multi-Seed Training

```bash
for seed in 42 123 456; do
    python -m src.distillation.train \
        --model reasoning \
        --reasoning-weight 0.25 \
        --seed $seed
done
```

### Debugging

```bash
# Quick sanity check (2 epochs, dummy data)
python -m src.distillation.train --model baseline --offline --max-epochs 2

# Verify model size
python -c "
from src.distillation.models import get_model_config
from src.distillation.models.reasoning_student import ReasoningStudent

cfg = get_model_config('500m')
model = ReasoningStudent.from_model_config(cfg)
params = sum(p.numel() for p in model.parameters())
print(f'500m: {params:,} parameters')
"
```

### Resuming Training

```bash
# Resume from a checkpoint
python -m src.distillation.train \
    --model reasoning \
    --resume checkpoints/reasoning_seed42_latest.pt
```

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size and use accumulation
--batch-size 4 --accumulation-steps 8

# Or try
--batch-size 2 --accumulation-steps 16
```

### Slow Data Loading

```bash
# Increase workers
--num-workers 8

# Or pre-extract frames instead of streaming
python scripts/extract_frames.py --split train
```

### NaN Loss

- Check learning rate (try `--lr 5e-5`)
- Verify data preprocessing (trajectories should be normalized)
- Check for corrupted samples in teacher labels
