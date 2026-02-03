# Architecture

## Overview

This project distills NVIDIA's Alpamayo-R1 (10B VLA) into a ~485M parameter student model for autonomous driving trajectory prediction. The student model learns to predict future vehicle trajectories from multi-camera images using the same unicycle kinematic model as the teacher.

## Model Sizes

| Config | Backbone | Transformer | Total Params | Notes |
|--------|----------|-------------|--------------|-------|
| 110m | ResNet-50 | 8 layers, d=768 | ~107M | Fast iteration, no checkpointing |
| 250m | ResNet-50 | 12 layers, d=1024 | ~250M | Balanced speed/capacity |
| 500m | ResNet-101 | 16 layers, d=1280 | ~485M | Uses gradient checkpointing |

## Data Flow

```
                            Input
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│           4 Cameras × 4 Frames × 224×224 RGB                │
│                    (B, 4, 12, 224, 224)                      │
│           + Ego History (B, 16, 6) past poses               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   ResNet-101 Encoder (shared)               │
│               12-channel input, 2048-dim output             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   BEV Projection Layer                       │
│            Ground-plane projection → 200×200 grid           │
│                  Channel reduction → 512                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 Transformer Decoder                          │
│        16 layers, d=1280, 20 heads                          │
│        64 learnable waypoint queries                        │
│        + ego history embedding (velocity context)           │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌──────────────────────┐        ┌──────────────────────┐
│   Action Head        │        │   Reasoning Head     │
│  (accel, curvature)  │        │  (384-dim embedding) │
│   64 timesteps       │        │ [ReasoningStudent]   │
└──────────────────────┘        └──────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│            Unicycle Kinematic Integrator                    │
│     (wraps Alpamayo's UnicycleAccelCurvatureActionSpace)    │
│     Estimates v0 from ego history, integrates actions       │
└─────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│              Output: 64 Waypoints @ 10Hz                    │
│                   6.4 seconds horizon                       │
│              (x, y, sin_yaw, cos_yaw) per step              │
└─────────────────────────────────────────────────────────────┘
```

## Components

### Encoder

**ResNet-101 Backbone** (`src/distillation/models/encoder.py`)
- Modified for 12-channel input (4 frames × 3 RGB)
- Pretrained on ImageNet, first conv reinitialized
- Shared weights across all 4 cameras
- Output: 2048-dimensional features at 1/32 resolution

**BEV Projection**
- Projects camera features to bird's eye view grid
- Ground-plane assumption (z=0)
- Pre-computed projection indices for efficiency
- Output: 200×200 grid at 0.5m/cell resolution

### Decoder

**Transformer Decoder** (`src/distillation/models/decoder.py`)
- DETR-style architecture with learnable queries
- 64 waypoint queries with temporal positional encoding
- 2D sinusoidal positional encoding for BEV features
- Pre-layer normalization for training stability

**Ego History Encoder**
- Encodes past 16 ego poses (1.6s at 10Hz, matching Alpamayo)
- Input: (B, 16, 6) where 6 = xyz position + roll/pitch/yaw angles
- Provides velocity/motion context for unicycle integration
- Required for v0 (initial velocity) estimation

### Output Heads

**Action Head**
- MLP: d_model → 512 → 2
- Outputs normalized (acceleration, curvature) per timestep
- These are then integrated via the unicycle kinematic model

**Reasoning Head** (ReasoningStudent only)
- Attention-weighted pooling over waypoints (preserves temporal structure)
- MLP: d_model → 1024 → 384
- Predicts teacher's Chain-of-Cognition embedding
- Trained with contrastive loss
- Discarded at inference (zero overhead)

### Unicycle Kinematic Layer

**UnicycleIntegrator** (`src/distillation/models/unicycle.py`)
- Wraps Alpamayo's `UnicycleAccelCurvatureActionSpace.action_to_traj()`
- Ensures exact mathematical compatibility with teacher model
- Denormalizes actions, estimates v0 from ego history via optimization
- Integrates using second-order heading terms and trapezoidal position integration
- Outputs kinematically feasible trajectories (no impossible motions)

## Model Configurations

### 500M (Default)

```python
ModelConfig(
    backbone="resnet101",
    d_model=1280,
    n_heads=20,
    n_layers=16,
    ffn_dim=5120,
    bev_channels=512,
    gradient_checkpointing=True,
)
```

- ~485M total parameters
- Requires gradient checkpointing
- Fits on 40GB+ GPU with batch size 8
- Use `--accumulation-steps 4` for effective batch size 32

### 250M (Balanced)

```python
ModelConfig(
    backbone="resnet50",
    d_model=1024,
    n_heads=16,
    n_layers=12,
    ffn_dim=4096,
    bev_channels=384,
    gradient_checkpointing=False,
)
```

- ~250M total parameters
- Balanced between speed and capacity
- Good default for most experiments

### 110M (Fast Iteration)

```python
ModelConfig(
    backbone="resnet50",
    d_model=768,
    n_heads=12,
    n_layers=8,
    ffn_dim=3072,
    bev_channels=256,
    gradient_checkpointing=False,
)
```

- ~107M total parameters
- No gradient checkpointing needed
- ~3x faster training than 500M
- Use for hyperparameter sweeps before scaling to 500M

## Benchmark Encoders

For nuScenes and Argoverse 2 benchmarks, alternative encoders replace the camera+BEV pipeline:

### RasterEncoder (nuScenes)

```
Input: BEV raster (B, C, H, W)
├── HD map layers (drivable area, lanes, etc.)
├── Agent history (rendered as polylines)
└── Ego history

Output: (B, N, d_model) tokens for transformer
```

- Used with `--input-type raster --benchmark nuscenes`
- CNN backbone processes multi-channel BEV raster
- Compatible with nuScenes prediction challenge format

### VectorEncoder (Argoverse 2)

```
Input: Vectorized scene representation
├── Agent tracks: (B, N_agents, T, 4) - positions + velocities
├── Map polylines: (B, N_polylines, P, 2) - lane centerlines, boundaries
└── Agent masks: (B, N_agents, T) - valid timestep indicators

Output: (B, N, d_model) tokens for transformer
```

- Used with `--input-type vector --benchmark argoverse2`
- PointNet-style encoding for tracks and polylines
- Attention-based aggregation
- Compatible with Argoverse 2 motion forecasting format

## Design Decisions

### Why ResNet backbone?
- Well-understood, pretrained weights available
- Efficient inference
- Easy to scale (50 → 101 → 152)

### Why BEV representation?
- Natural for trajectory prediction (bird's eye view)
- Enables geometric reasoning about scene layout
- Common in autonomous driving literature

### Why DETR-style decoder?
- Parallel waypoint prediction (vs autoregressive)
- Set prediction aligns with trajectory output
- Proven effective for detection → adapted for trajectory

### Why contrastive loss for reasoning?
- No text generation overhead at inference
- Captures semantic structure of teacher's reasoning
- Forces student to learn similar intermediate representations

## File Structure

```
src/distillation/
├── models/
│   ├── config.py              # ModelConfig dataclass + presets
│   ├── encoder.py             # ResNet backbone + BEV projection
│   ├── decoder.py             # Transformer decoder + heads
│   ├── unicycle.py            # Unicycle kinematic integrator
│   ├── baseline_student.py    # Trajectory-only model
│   └── reasoning_student.py   # + Reasoning head
├── encoders/
│   ├── raster_encoder.py      # BEV raster encoder (nuScenes)
│   └── vector_encoder.py      # Vectorized encoder (Argoverse 2)
├── datasets/
│   ├── nuscenes.py            # nuScenes dataset adapter
│   └── argoverse2.py          # Argoverse 2 dataset adapter
├── training/
│   ├── state.py               # TrainingState management
│   ├── loops.py               # Train/validation epoch loops
│   └── benchmark_loops.py     # Benchmark-specific training
├── dataset.py                 # PhysicalAI-AV data loading
├── losses.py                  # Trajectory + contrastive losses
├── calibration.py             # Camera math + BEV projector
├── cli.py                     # Argument parsing
└── train.py                   # Training entry point
```
