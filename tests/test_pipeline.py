import json
from pathlib import Path

import numpy as np
import pytest
import torch

from driveworld.data import WindowDataset, convert_nuscenes
from driveworld.geometry import Grid, ego_step, rasterize
from driveworld.metrics import intersection_union, static_world_baseline
from driveworld.model import ActionRollout


Q = [1.0, 0.0, 0.0, 0.0]


def test_ego_transform_and_occupancy_geometry():
    pose = {"translation": [0, 0, 0], "rotation": Q}
    next_pose = {"translation": [1, 0, 0], "rotation": Q}
    np.testing.assert_allclose(ego_step(pose, next_pose), [1, 0, 0])
    ann = {"category_name": "vehicle.car", "translation": [4, 0, 0],
           "size": [2, 4, 2], "rotation": Q}
    grid = rasterize([ann], pose, Grid(32, 0.5))
    assert grid.shape == (2, 32, 32)
    assert grid[0, 8, 16] == 1
    assert grid[1].sum() == 0


def test_static_world_warp_follows_ego_motion():
    image = torch.zeros(1, 2, 32, 32)
    image[0, 0, 12, 16] = 1  # object four pixels ahead
    actions = torch.tensor([[[1.0, 0.0, 0.0]]])
    shifted = static_world_baseline(image, actions, resolution=0.5)
    assert shifted[0, 0, 0, 14, 16] == 1  # ego advanced two pixels
    assert shifted[0, 0, 0, 12, 16] == 0


def test_model_rollout_has_gradients_and_horizon_metrics():
    model = ActionRollout(16)
    history = torch.rand(2, 3, 2, 32, 32)
    actions = torch.randn(2, 4, 3)
    prediction = model(history, actions)
    assert prediction.shape == (2, 4, 2, 32, 32)
    prediction.mean().backward()
    assert model.action[0].weight.grad.abs().sum() > 0
    intersection, union = intersection_union(prediction.sigmoid(), torch.zeros_like(prediction))
    assert intersection.shape == union.shape == (4, 2)


def _write_table(folder: Path, name: str, rows: list[dict]):
    (folder / f"{name}.json").write_text(json.dumps(rows), encoding="utf-8")


def test_conversion_respects_scene_split_and_future_actions(tmp_path: Path):
    metadata = tmp_path / "raw" / "v1.0-mini"
    metadata.mkdir(parents=True)
    scenes, samples, sample_data, poses, annotations, instances = [], [], [], [], [], []
    for scene_id in range(2):
        scenes.append({"token": f"scene{scene_id}", "name": f"scene-{scene_id:04d}",
                       "first_sample_token": f"s{scene_id}_0"})
        instances.append({"token": f"instance{scene_id}", "category_token": "car"})
        for t in range(8):
            sample_token = f"s{scene_id}_{t}"
            annotations.append({"token": f"a{scene_id}_{t}", "sample_token": sample_token,
                                "instance_token": f"instance{scene_id}",
                                "translation": [float(t + 4), 1.0, 0.0], "size": [2, 4, 2], "rotation": Q})
            samples.append({"token": sample_token, "timestamp": t * 500_000,
                            "next": f"s{scene_id}_{t + 1}" if t < 7 else ""})
            sample_data.append({"token": f"d{scene_id}_{t}", "sample_token": sample_token,
                                "ego_pose_token": f"p{scene_id}_{t}", "calibrated_sensor_token": "calib",
                                "is_key_frame": True})
            poses.append({"token": f"p{scene_id}_{t}", "translation": [float(t), 0, 0], "rotation": Q})
    for name, rows in {
        "scene": scenes, "sample": samples, "sample_data": sample_data, "ego_pose": poses,
        "sample_annotation": annotations, "instance": instances,
        "category": [{"token": "car", "name": "vehicle.car"}],
        "sensor": [{"token": "sensor", "channel": "LIDAR_TOP"}],
        "calibrated_sensor": [{"token": "calib", "sensor_token": "sensor"}],
    }.items():
        _write_table(metadata, name, rows)
    output = tmp_path / "prepared"
    manifest = convert_nuscenes(tmp_path / "raw", output, history=2, horizon=2, grid=Grid(32, 0.5))
    assert manifest["windows"] == {"train": 5, "val": 5}
    assert set(manifest["split_scenes"]["train"]).isdisjoint(manifest["split_scenes"]["val"])
    dataset = WindowDataset(output / "val.npz")
    history, actions, future = dataset[0]
    assert history.shape == future.shape == (2, 2, 32, 32)
    np.testing.assert_allclose(actions, [[1, 0, 0], [1, 0, 0]])
    with pytest.raises(ValueError):
        convert_nuscenes(tmp_path / "raw", tmp_path / "bad", max_scenes=1)
