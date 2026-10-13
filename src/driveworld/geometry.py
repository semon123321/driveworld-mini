"""Coordinate transforms and annotation rasterization for nuScenes keyframes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class Grid:
    size: int = 64
    resolution: float = 0.5

    def __post_init__(self) -> None:
        if self.size < 8 or self.size % 4 != 0 or self.resolution <= 0:
            raise ValueError("Grid size must be divisible by 4 and resolution positive")


def yaw_from_quaternion(q: list[float]) -> float:
    """nuScenes stores rotations as [w, x, y, z]."""
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap_angle(value: float) -> float:
    return (value + math.pi) % (2 * math.pi) - math.pi


def local_xy(global_xy: np.ndarray, origin_xy: np.ndarray, yaw: float) -> np.ndarray:
    dx, dy = np.asarray(global_xy, dtype=float) - np.asarray(origin_xy, dtype=float)
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * dx + s * dy, -s * dx + c * dy], dtype=float)


def ego_step(current_pose: dict, next_pose: dict) -> np.ndarray:
    heading = yaw_from_quaternion(current_pose["rotation"])
    delta = local_xy(
        np.asarray(next_pose["translation"][:2]),
        np.asarray(current_pose["translation"][:2]),
        heading,
    )
    turn = wrap_angle(yaw_from_quaternion(next_pose["rotation"]) - heading)
    return np.array([delta[0], delta[1], turn], dtype=np.float32)


def category_channel(name: str) -> int | None:
    if name.startswith("vehicle."):
        return 0
    if name.startswith("human.pedestrian.") or name.startswith("cycle."):
        return 1
    return None


def rasterize(annotations: list[dict], pose: dict, grid: Grid) -> np.ndarray:
    """Two channels: vehicles and vulnerable road users; ego vehicle excluded."""
    image = [Image.new("L", (grid.size, grid.size)) for _ in range(2)]
    drawers = [ImageDraw.Draw(channel) for channel in image]
    ego_origin = np.asarray(pose["translation"][:2], dtype=float)
    ego_yaw = yaw_from_quaternion(pose["rotation"])
    for ann in annotations:
        channel = category_channel(ann["category_name"])
        if channel is None:
            continue
        center = local_xy(np.asarray(ann["translation"][:2]), ego_origin, ego_yaw)
        heading = yaw_from_quaternion(ann["rotation"]) - ego_yaw
        width, length = ann["size"][:2]
        footprint = np.array([
            [length / 2, width / 2], [length / 2, -width / 2],
            [-length / 2, -width / 2], [-length / 2, width / 2],
        ])
        c, s = math.cos(heading), math.sin(heading)
        points = footprint @ np.array([[c, s], [-s, c]]) + center
        pixels = [
            (grid.size / 2 - y / grid.resolution, grid.size / 2 - x / grid.resolution)
            for x, y in points
        ]
        drawers[channel].polygon(pixels, fill=1)
    return np.stack([np.asarray(channel, dtype=np.uint8) for channel in image])
