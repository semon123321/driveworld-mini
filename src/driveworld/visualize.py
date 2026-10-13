"""Save an easy-to-read rollout comparison for a validation window."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .data import WindowDataset
from .metrics import static_world_baseline
from .model import ActionRollout
from .runner import device_for


def _render(occupancy: np.ndarray, scale: int = 4) -> Image.Image:
    height, width = occupancy.shape[-2:]
    pixels = np.full((height, width, 3), 248, dtype=np.uint8)
    pixels[occupancy[0] >= 0.5] = (208, 60, 68)
    pixels[occupancy[1] >= 0.5] = (38, 137, 151)
    center = (height // 2, width // 2)
    pixels[center[0] - 1:center[0] + 2, center[1] - 1:center[1] + 2] = (25, 33, 39)
    return Image.fromarray(pixels).resize((width * scale, height * scale), Image.Resampling.NEAREST)


def render_comparison(data_dir: Path, checkpoint: Path, output: Path, index: int = 0,
                      device: str = "auto") -> None:
    selected = device_for(device)
    saved = torch.load(checkpoint, map_location=selected, weights_only=True)
    dataset = WindowDataset(data_dir / "val.npz")
    if not 0 <= index < len(dataset):
        raise IndexError(f"Index must be in [0, {len(dataset) - 1}]")
    history_np, actions_np, target_np = dataset[index]
    history = torch.from_numpy(history_np).unsqueeze(0).to(selected)
    actions = torch.from_numpy(actions_np).unsqueeze(0).to(selected)
    model = ActionRollout(saved["channels"], saved["manifest"]["resolution_m"]).to(selected)
    model.load_state_dict(saved["model"])
    model.eval()
    with torch.inference_mode():
        predicted = model(history, actions).sigmoid()[0].cpu().numpy()
        baseline = static_world_baseline(history[:, -1], actions, saved["manifest"]["resolution_m"])[0].cpu().numpy()
    rows = [("Ground truth", target_np), ("Model", predicted), ("Static world", baseline)]
    size = history_np.shape[-1] * 4
    gap, label_height = 12, 32
    canvas = Image.new("RGB", (gap + (size + gap) * len(target_np),
                               gap + (size + label_height + gap) * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, (label, sequence) in enumerate(rows):
        top = gap + row_index * (size + label_height + gap)
        for step, occupancy in enumerate(sequence):
            left = gap + step * (size + gap)
            draw.text((left, top), f"{label} | +{(step + 1) * 0.5:.1f}s", fill=(20, 28, 33))
            canvas.paste(_render(occupancy), (left, top + label_height))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
