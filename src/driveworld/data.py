"""Build scene-disjoint BEV windows from official nuScenes mini metadata."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from .geometry import Grid, ego_step, rasterize


def _table(folder: Path, name: str) -> dict[str, dict]:
    with (folder / f"{name}.json").open(encoding="utf-8") as stream:
        return {row["token"]: row for row in json.load(stream)}


def _windows(frames: list[tuple[np.ndarray, dict, int]], history: int, horizon: int):
    width = history + horizon
    for start in range(len(frames) - width + 1):
        chunk = frames[start:start + width]
        # nuScenes keyframes are approximately 2 Hz; never bridge a missing interval.
        if any(not 400_000 <= chunk[i + 1][2] - chunk[i][2] <= 650_000 for i in range(width - 1)):
            continue
        history_maps = np.stack([frame[0] for frame in chunk[:history]])
        target_maps = np.stack([frame[0] for frame in chunk[history:]])
        actions = np.stack([
            ego_step(chunk[i - 1][1], chunk[i][1])
            for i in range(history, width)
        ])
        yield history_maps, actions, target_maps


def convert_nuscenes(
    dataroot: Path,
    output: Path,
    *,
    history: int = 3,
    horizon: int = 4,
    grid: Grid = Grid(),
    max_scenes: int | None = None,
) -> dict:
    """Read JSON tables only; raw camera and LiDAR files are not needed."""
    if history < 1 or horizon < 1:
        raise ValueError("History and horizon must be positive")
    metadata = dataroot / "v1.0-mini"
    if not metadata.is_dir():
        raise FileNotFoundError(f"Missing nuScenes mini metadata: {metadata}")
    scenes = _table(metadata, "scene")
    samples = _table(metadata, "sample")
    sample_data = _table(metadata, "sample_data")
    ego_poses = _table(metadata, "ego_pose")
    annotations = _table(metadata, "sample_annotation")
    instances = _table(metadata, "instance")
    categories = _table(metadata, "category")
    sensors = _table(metadata, "sensor")
    calibrated_sensors = _table(metadata, "calibrated_sensor")
    lidar_by_sample = {}
    for row in sample_data.values():
        channel = sensors[calibrated_sensors[row["calibrated_sensor_token"]]["sensor_token"]]["channel"]
        if channel == "LIDAR_TOP" and row["is_key_frame"]:
            lidar_by_sample[row["sample_token"]] = row
    annotations_by_sample: dict[str, list[dict]] = {}
    for row in annotations.values():
        annotations_by_sample.setdefault(row["sample_token"], []).append(row)
    category_by_instance = {
        token: categories[row["category_token"]]["name"]
        for token, row in instances.items()
    }
    names = sorted(scenes, key=lambda token: scenes[token]["name"])
    if max_scenes is not None:
        if max_scenes < 2:
            raise ValueError("At least two scenes are needed for scene-disjoint splits")
        names = names[:max_scenes]
    if len(names) < 2:
        raise ValueError("At least two scenes are needed for scene-disjoint splits")
    split_at = max(1, int(len(names) * 0.8))
    split_at = min(split_at, len(names) - 1)
    val_scene_tokens = set(names[split_at:])
    output.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {"train": [], "val": []}
    scene_names: dict[str, list[str]] = {"train": [], "val": []}

    for token in names:
        scene = scenes[token]
        split = "val" if token in val_scene_tokens else "train"
        scene_names[split].append(scene["name"])
        frames = []
        sample_token = scene["first_sample_token"]
        while sample_token:
            sample = samples[sample_token]
            lidar = lidar_by_sample[sample_token]
            pose = ego_poses[lidar["ego_pose_token"]]
            boxes = []
            for ann in annotations_by_sample.get(sample_token, []):
                boxes.append({
                    "category_name": category_by_instance[ann["instance_token"]],
                    "translation": ann["translation"],
                    "size": ann["size"],
                    "rotation": ann["rotation"],
                })
            frames.append((rasterize(boxes, pose, grid), pose, sample["timestamp"]))
            sample_token = sample["next"]
        groups[split].extend(_windows(frames, history, horizon))

    counts = {}
    for split, windows in groups.items():
        if not windows:
            raise ValueError(f"No {split} windows; try fewer history/future steps")
        history_maps, actions, target_maps = (np.stack(parts) for parts in zip(*windows))
        np.savez_compressed(
            output / f"{split}.npz",
            history=history_maps,
            actions=actions,
            future=target_maps,
        )
        counts[split] = len(windows)
    manifest = {
        "dataset": "nuScenes v1.0-mini annotations",
        "history": history,
        "horizon": horizon,
        "grid_size": grid.size,
        "resolution_m": grid.resolution,
        "sampling_hz": 2,
        "channels": ["vehicle", "vulnerable_road_user"],
        "split_scenes": scene_names,
        "windows": counts,
        "action_definition": "Future ego delta x/y/yaw in previous ego frame; oracle at offline evaluation",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class WindowDataset(Dataset):
    def __init__(self, path: Path):
        with np.load(path) as dataset:
            self.history = dataset["history"].astype(np.float32)
            self.actions = dataset["actions"].astype(np.float32)
            self.future = dataset["future"].astype(np.float32)
        if len(self.history) != len(self.actions) or len(self.actions) != len(self.future):
            raise ValueError("Mismatched window counts")
        if self.history.ndim != 5 or self.future.ndim != 5 or self.actions.ndim != 3:
            raise ValueError("Expected history/future [N,T,C,H,W] and actions [N,T,3]")
        if self.history.shape[2] != 2 or self.future.shape[2] != 2 or self.actions.shape[2] != 3:
            raise ValueError("Expected 2 occupancy channels and 3 action values")
        if self.actions.shape[1] != self.future.shape[1]:
            raise ValueError("Actions and future have different horizons")

    def __len__(self) -> int:
        return len(self.history)

    def __getitem__(self, index: int):
        return self.history[index], self.actions[index], self.future[index]
