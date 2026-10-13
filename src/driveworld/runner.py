"""Training and evaluation with scene-disjoint validation."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import WindowDataset
from .metrics import intersection_union, occupancy_loss, static_world_baseline
from .model import ActionRollout


def device_for(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(value)


def _loader(path: Path, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(WindowDataset(path), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _calibrate(model, loader, device) -> list[float]:
    """Choose one threshold per class on training scenes only."""
    model.eval()
    predictions, targets = [], []
    with torch.inference_mode():
        for history, actions, future in loader:
            predictions.append(model(history.to(device), actions.to(device)).sigmoid().cpu())
            targets.append(future)
    prediction = torch.cat(predictions)
    target = torch.cat(targets).bool()
    thresholds = []
    for channel in range(2):
        scores = []
        for threshold in torch.arange(0.3, 0.81, 0.05):
            binary = prediction[:, :, channel] >= threshold
            truth = target[:, :, channel]
            intersection = (binary & truth).sum().item()
            union = (binary | truth).sum().item()
            scores.append((intersection / union if union else 0.0, float(threshold)))
        thresholds.append(max(scores)[1])
    return thresholds


def _measure(model, loader, device, resolution: float, thresholds: list[float]) -> dict:
    model.eval()
    counts = {}
    with torch.inference_mode():
        for history, actions, future in loader:
            history, actions, future = history.to(device), actions.to(device), future.to(device)
            probability = model(history, actions).sigmoid()
            cutoff = probability.new_tensor(thresholds).view(1, 1, 2, 1, 1)
            predictions = {
                "model": (probability >= cutoff).float(),
                "static_world": static_world_baseline(history[:, -1], actions, resolution),
                "persistence": history[:, -1:].expand(-1, future.shape[1], -1, -1, -1),
            }
            for name, prediction in predictions.items():
                intersection, union = intersection_union(prediction, future)
                if name not in counts:
                    counts[name] = [intersection.cpu(), union.cpu()]
                else:
                    counts[name][0] += intersection.cpu()
                    counts[name][1] += union.cpu()
    result = {}
    for name, (intersection, union) in counts.items():
        values = []
        for step in range(union.shape[0]):
            present = union[step] > 0
            values.append(float((intersection[step, present] / union[step, present]).mean()) if present.any() else None)
        valid = [x for x in values if x is not None]
        channel_scores = {}
        for channel, label in enumerate(("vehicle", "vulnerable_road_user")):
            total_union = union[:, channel].sum()
            channel_scores[label] = float(intersection[:, channel].sum() / total_union) if total_union else None
        result[name] = {"iou_by_horizon": values, "mean_iou": float(np.mean(valid)) if valid else None,
                        "global_iou_by_channel": channel_scores}
    return result


def train(data_dir: Path, run_dir: Path, *, epochs: int = 20, batch_size: int = 8,
          learning_rate: float = 1e-3, device: str = "auto", seed: int = 7,
          channels: int = 32, patience: int = 5) -> dict:
    if epochs < 1 or batch_size < 1:
        raise ValueError("Epochs and batch size must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    selected_device = device_for(device)
    with (data_dir / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    training = _loader(data_dir / "train.npz", batch_size, True)
    validation = _loader(data_dir / "val.npz", batch_size, False)
    model = ActionRollout(channels, manifest["resolution_m"]).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    run_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    stale_epochs = 0
    records = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for history, actions, future in training:
            history, actions, future = history.to(selected_device), actions.to(selected_device), future.to(selected_device)
            optimizer.zero_grad(set_to_none=True)
            loss = occupancy_loss(model(history, actions), future)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        model.eval()
        val_losses = []
        with torch.inference_mode():
            for history, actions, future in validation:
                history, actions, future = history.to(selected_device), actions.to(selected_device), future.to(selected_device)
                val_losses.append(float(occupancy_loss(model(history, actions), future)))
        record = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": float(np.mean(val_losses))}
        records.append(record)
        print(json.dumps(record), flush=True)
        if record["val_loss"] < best:
            best = record["val_loss"]
            stale_epochs = 0
            torch.save({"model": model.state_dict(), "channels": channels, "manifest": manifest,
                        "best_val_loss": best, "seed": seed}, run_dir / "best.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping after {epoch} epochs", flush=True)
                break
    (run_dir / "history.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    return evaluate(data_dir, run_dir / "best.pt", run_dir / "metrics.json", batch_size=batch_size, device=device)


def evaluate(data_dir: Path, checkpoint: Path, output: Path | None = None, *,
             batch_size: int = 8, device: str = "auto") -> dict:
    selected_device = device_for(device)
    saved = torch.load(checkpoint, map_location=selected_device, weights_only=True)
    with (data_dir / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    for key in ("history", "horizon", "grid_size", "resolution_m", "channels"):
        if saved["manifest"][key] != manifest[key]:
            raise ValueError(f"Checkpoint and data disagree on {key}")
    model = ActionRollout(saved["channels"], manifest["resolution_m"]).to(selected_device)
    model.load_state_dict(saved["model"])
    thresholds = _calibrate(model, _loader(data_dir / "train.npz", batch_size, False), selected_device)
    result = {
        "dataset": manifest["dataset"],
        "split_scenes": manifest["split_scenes"]["val"],
        "windows": manifest["windows"]["val"],
        "history_steps": manifest["history"],
        "future_steps": manifest["horizon"],
        "notes": "Ground-truth future ego motion is an oracle conditioning signal; this is offline prediction, not policy evaluation.",
        "model_thresholds_calibrated_on_train": thresholds,
        "methods": _measure(model, _loader(data_dir / "val.npz", batch_size, False), selected_device,
                            manifest["resolution_m"], thresholds),
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result
