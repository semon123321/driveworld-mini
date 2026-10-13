# nuScenes mini experiment

Date: 2026-10-13. Device: NVIDIA GeForce RTX 5070 Laptop GPU (8 GB). Software: Python 3.12, PyTorch 2.9.1+cu128. Commands:

```powershell
driveworld convert --dataroot data/raw --output data/prepared
driveworld train --data data/prepared --run runs/final --epochs 20 --batch-size 8 --device cuda
driveworld visualize --data data/prepared --checkpoint runs/final/best.pt --output assets/rollout.png --index 0 --device cuda
driveworld eval --data data/prepared --checkpoint runs/final/best.pt --output results/nuscenes-mini.json --device cuda
```

The scenes are split by sorted scene name: eight train, two validation. The [manifest](manifest.json) lists exact scene IDs. Fixed seed: 7. [Training history](training-history.json) shows early stopping after ten epochs because validation loss did not improve for five epochs; epoch 5 had the lowest validation loss (0.9661). Binary thresholds for the learned model were selected only on training windows, with 0.65 for vehicles and 0.30 for vulnerable road users. The baselines are binary.

The model reached **0.3688 mean horizon IoU**; the static-world warp reached **0.3712** and unwarped persistence **0.2196**. The learned residual was slightly worse than the geometry baseline on this split. This points to limited value from fitting dynamics on 248 small training windows. The plotted first validation window also shows drift and missed objects at longer horizons. Do not describe this as a successful policy or video generation model.

The experiment uses future ego pose deltas derived from recorded data to condition both the model and the warp. This is oracle motion input and cannot be taken as closed-loop planning performance. A stronger follow-up would evaluate additional scenes, agent motion priors, and candidate plans generated without future labels.
