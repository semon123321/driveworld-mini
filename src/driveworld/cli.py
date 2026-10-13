"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import convert_nuscenes
from .geometry import Grid
from .runner import evaluate, train
from .visualize import render_comparison


def main() -> None:
    parser = argparse.ArgumentParser(prog="driveworld")
    command = parser.add_subparsers(dest="command", required=True)
    convert = command.add_parser("convert", help="Convert nuScenes mini metadata to occupancy windows")
    convert.add_argument("--dataroot", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--history", type=int, default=3)
    convert.add_argument("--horizon", type=int, default=4)
    convert.add_argument("--grid-size", type=int, default=64)
    convert.add_argument("--resolution", type=float, default=0.5)
    convert.add_argument("--max-scenes", type=int)
    fit = command.add_parser("train", help="Fit the recurrent predictor and evaluate best checkpoint")
    fit.add_argument("--data", type=Path, required=True)
    fit.add_argument("--run", type=Path, required=True)
    fit.add_argument("--epochs", type=int, default=20)
    fit.add_argument("--batch-size", type=int, default=8)
    fit.add_argument("--learning-rate", type=float, default=1e-3)
    fit.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    fit.add_argument("--seed", type=int, default=7)
    fit.add_argument("--patience", type=int, default=5)
    score = command.add_parser("eval", help="Report horizon-wise IoU versus two baselines")
    score.add_argument("--data", type=Path, required=True)
    score.add_argument("--checkpoint", type=Path, required=True)
    score.add_argument("--output", type=Path)
    score.add_argument("--batch-size", type=int, default=8)
    score.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    visualize = command.add_parser("visualize", help="Save a rollout PNG for one validation window")
    visualize.add_argument("--data", type=Path, required=True)
    visualize.add_argument("--checkpoint", type=Path, required=True)
    visualize.add_argument("--output", type=Path, required=True)
    visualize.add_argument("--index", type=int, default=0)
    visualize.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.command == "convert":
        result = convert_nuscenes(args.dataroot, args.output, history=args.history,
                                  horizon=args.horizon, grid=Grid(args.grid_size, args.resolution),
                                  max_scenes=args.max_scenes)
        print(json.dumps(result, indent=2))
    elif args.command == "train":
        train(args.data, args.run, epochs=args.epochs, batch_size=args.batch_size,
              learning_rate=args.learning_rate, device=args.device, seed=args.seed,
              patience=args.patience)
    elif args.command == "eval":
        evaluate(args.data, args.checkpoint, args.output, batch_size=args.batch_size, device=args.device)
    else:
        render_comparison(args.data, args.checkpoint, args.output, index=args.index, device=args.device)


if __name__ == "__main__":
    main()
