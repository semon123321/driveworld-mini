# DriveWorld Mini

An action-conditioned, autoregressive bird's-eye-view (BEV) occupancy prediction baseline for **nuScenes mini**. It converts scene annotations into ego-frame occupancy grids, predicts the next four keyframes from three historical keyframes and a proposed future ego-motion sequence, and reports horizon-wise IoU against two simple baselines.

This is a compact research and engineering project, **not** a production driving stack or a photorealistic video world model. It is designed to make data alignment, action conditioning, multi-step rollout, and evaluation inspectable in a code review.

## What it demonstrates

- **Perception to representation:** Oriented 3D annotation boxes become two BEV channels: vehicles and vulnerable road users.
- **Action-conditioned dynamics:** A geometry warp handles the known ego-motion coordinate change; a convolutional recurrent model receives motion increments `[forward_m, lateral_m, yaw_rad]` and predicts residual occupancy dynamics autoregressively.
- **Explicit reference baselines:** Compare with raw persistence and a static-world warp that accounts for ego motion but assumes other agents do not move.
- **Honest evaluation:** Report IoU at each 0.5 s horizon on held-out scenes. All windows from a scene stay in one split.

The model's binary occupancy thresholds are calibrated **only on training scenes**; validation labels are never used to select them. The baselines produce binary occupancy directly.

The future ego motion used for offline evaluation comes from recorded poses. It is an **oracle conditioning input**, not an action inferred by this model. Improving a real policy would require candidate plans available at inference and closed-loop evaluation.

## Setup

Python 3.10–3.12 is recommended. Install a PyTorch build appropriate for your CPU/GPU using the [official selector](https://docs.pytorch.org/get-started/locally/), then install the project:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[test]"
pytest -q
```

The code has been developed for 64x64 grids and 8 GB GPUs; start with `--batch-size 8` and reduce it if needed.

## Data

Download the [official nuScenes mini archive](https://www.nuscenes.org/data/v1.0-mini.tgz) and extract it to `data/raw`, such that `data/raw/v1.0-mini/scene.json` exists. Only the JSON metadata is read by this baseline; raw sensor files are not used. Keep the dataset out of the Git repository and follow the [nuScenes terms](https://www.nuscenes.org/terms-of-use).

```powershell
driveworld convert --dataroot data/raw --output data/prepared
driveworld train --data data/prepared --run runs/baseline --epochs 20 --batch-size 8
driveworld eval --data data/prepared --checkpoint runs/baseline/best.pt --output runs/baseline/metrics.json
driveworld visualize --data data/prepared --checkpoint runs/baseline/best.pt --output runs/baseline/rollout.png
```

`manifest.json` records scene names, split membership, grid scale, horizon, and counts. The converter uses only successive keyframes with approximately 0.5 s gaps; it never creates a window across a missing interval or scene boundary. The first 80% of scene names in sorted order become training scenes, with the remainder reserved for validation. `--max-scenes` can speed up debugging while retaining scene-disjoint splits.

## Results on nuScenes mini

The [experiment record](results/README.md) contains the exact setup, [scene manifest](results/manifest.json), [training history](results/training-history.json), and unrounded [metrics JSON](results/nuscenes-mini.json). We used eight scenes for training and two for validation, 248/68 windows, three history steps and four future steps. On an RTX 5070 Laptop GPU, the best checkpoint was epoch 5; early stopping terminated at epoch 10.

| Method | +0.5 s IoU | +1.0 s IoU | +1.5 s IoU | +2.0 s IoU | Mean |
|---|---:|---:|---:|---:|---:|
| Persistence | 0.289 | 0.222 | 0.187 | 0.181 | 0.220 |
| Static-world ego warp | **0.503** | **0.373** | 0.321 | **0.287** | **0.371** |
| Action-conditioned ConvGRU | 0.499 | 0.371 | 0.321 | 0.285 | 0.369 |

The learned residual **does not beat the geometry baseline** on this split. The static-world warp accounts for most of the measurable change, while the small training set encourages overfitting. This is an experimental finding, not a state-of-the-art claim. A qualitative [validation rollout](assets/rollout.png) shows ground truth, model, and geometry baseline over two seconds.

On small sparse grids, IoU can be unstable; examine all horizons and failure examples. This proof of concept uses box occupancy rather than pixel video, and nuScenes mini is too small to support broad generalization claims.

## Suggested interview extensions

1. Run an **action ablation** by zeroing or shuffling future actions, then compare horizon-wise metrics. Separate the contribution of the geometry warp from the learned residual.
2. Add agent motion features or a rasterized drivable-area map, with an explicit map-availability and leakage check.
3. Expand to the full dataset and a fixed official scene split, then test longer rollouts and collision-relevant metrics.
4. Sample several candidate ego trajectories and use predicted occupancy to score collision risk, followed by a simulator evaluation. Offline labels alone are insufficient to claim policy improvement.

## Project structure

`src/driveworld/geometry.py` handles transforms and rasterization; `data.py` builds disjoint windows; `model.py` defines the recurrent predictor; `metrics.py` provides loss and baselines; `runner.py` trains and evaluates; `visualize.py` writes a rollout panel. Tests cover geometry, ego warp, gradients, and conversion on a tiny nuScenes-shaped fixture.

## Attribution

This repository contains original baseline code and no nuScenes data. nuScenes dataset and schema are provided by Motional; see the [official dataset tutorial](https://www.nuscenes.org/tutorials/nuscenes_tutorial.html) and [schema](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md). Project source code is MIT licensed; dataset rights are separate.
