# Alpamayo Distilled

Smaller, faster models distilled from NVIDIA's [Alpamayo-R1](https://huggingface.co/nvidia/Alpamayo-R1-10B) vision-language-action model for autonomous driving.

## Overview

This repo contains:
- **Data pipeline** for training on [PhysicalAI-AV](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
- **Teacher label generation** from Alpamayo-R1
- **Student models** (~485M params) - compact trajectory predictors using unicycle kinematics

The student model predicts acceleration and steering curvature (rather than waypoints directly), then integrates through a unicycle kinematic model. This approach is more stable to train and produces physically plausible trajectories. See the [Alpamayo paper](https://arxiv.org/abs/2511.00088) for details on why action-based prediction outperforms direct waypoint regression.

## Setup

```bash
git clone https://github.com/mu-hashmi/alpamayo-r1-distilled.git
cd alpamayo-r1-distilled

# Clone external dependencies
mkdir -p external && cd external
git clone https://github.com/NVlabs/alpamayo.git
git clone https://github.com/NVlabs/physical_ai_av.git
cd ..

# Install (CPU-only, for training the student)
uv sync

# For teacher label generation (requires GPU), also install:
uv pip install torch sentence-transformers
uv pip install -e external/physical_ai_av
uv pip install -e external/alpamayo
```

## Data Preparation

### 1. Extract Frames

```bash
# Extract camera frames from HuggingFace dataset
uv run python scripts/extract_frames.py --split train --resolution 224 --num-samples 10000
uv run python scripts/extract_frames.py --split val --resolution 224 --num-samples 1000
```

### 2. Generate Teacher Labels (Requires 24GB+ VRAM)

```bash
uv run python scripts/generate_teacher_labels.py --split train
uv run python scripts/generate_teacher_labels.py --split val
```

Supports sharding for multi-GPU:
```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/generate_teacher_labels.py --split train --shard 0/4
CUDA_VISIBLE_DEVICES=1 uv run python scripts/generate_teacher_labels.py --split train --shard 1/4
# ...
uv run python scripts/merge_shards.py --split train --num-shards 4
```

### 3. Generate Ego History (Required)

Provides velocity context for unicycle integration:

```bash
uv run python scripts/generate_ego_history.py --split all
```

## Training

### Quick Test

```bash
uv run python -m src.distillation.train --model baseline --offline --max-epochs 2
```

### Recommended: Stabilized Unicycle Mode

This configuration includes the full stabilization stack for reliable training:

```bash
uv run python -m src.distillation.train \
    --model baseline \
    --model-size 500m \
    --lr 1e-5 \
    --grad-clip 1.0 \
    --ema --ema-decay 0.999 \
    --action-weight 0.1 \
    --freeze-bn \
    --batch-size 32 \
    --max-epochs 50
```

Key flags:
- `--lr 1e-5` - Low learning rate (sequential integration amplifies gradients)
- `--grad-clip 1.0` - Prevents gradient explosions from bad batches
- `--ema` - Smooths weight oscillations
- `--action-weight 0.1` - Auxiliary supervision on predicted actions
- `--freeze-bn` - Uses pretrained BatchNorm statistics (avoids train/eval mismatch)

### Direct Trajectory Mode (Ablation)

For comparison, you can train without unicycle integration:

```bash
uv run python -m src.distillation.train \
    --model baseline \
    --no-unicycle \
    --lr 1e-4 \
    --batch-size 32
```

This predicts (x, y, yaw) directly. Expect higher error and less stable training.

### Benchmark Training (nuScenes / Argoverse 2)

```bash
# nuScenes
uv run python -m src.distillation.train \
    --model baseline \
    --benchmark nuscenes \
    --nuscenes-root /path/to/nuscenes \
    --input-type raster \
    --num-modes 6

# Argoverse 2
uv run python -m src.distillation.train \
    --model baseline \
    --benchmark argoverse2 \
    --argoverse2-root /path/to/av2 \
    --input-type vector \
    --num-modes 6
```

### Resume Training

```bash
uv run python -m src.distillation.train \
    --model baseline \
    --resume checkpoints/baseline_seed42_latest.pt
```

See [docs/TRAINING.md](docs/TRAINING.md) for all configuration options.

## Related

- [Alpamayo-R1 Paper](https://arxiv.org/abs/2511.00088) - Architecture and training details
- [Alpamayo-R1 Model](https://huggingface.co/nvidia/Alpamayo-R1-10B) - Teacher model
- [PhysicalAI-AV Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) - Training data

## License

MIT
