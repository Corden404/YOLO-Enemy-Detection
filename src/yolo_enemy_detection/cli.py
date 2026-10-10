"""Command-line interface for the gameplay detection pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__
from .engine import DEFAULT_WEIGHTS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLOv8 gameplay soldier detection")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    extract = commands.add_parser("extract", help="Extract frames from a local video")
    extract.add_argument("--source", type=Path, required=True)
    extract.add_argument("--output", type=Path, default=Path("data/frames"))
    extract.add_argument("--fps", type=float, default=2.0)
    extract.add_argument("--segment", action="append", help="START-END in seconds or HH:MM:SS; repeat to select multiple segments")

    prepare = commands.add_parser("prepare", help="Validate labels and build a YOLO dataset")
    prepare.add_argument("--frames", type=Path, required=True)
    prepare.add_argument("--labels", type=Path, required=True, help="YOLO label directory or ZIP archive")
    prepare.add_argument("--output", type=Path, default=Path("data/dataset"))
    prepare.add_argument("--classes", nargs="+", default=["soldier"])
    prepare.add_argument("--train-ratio", type=float, default=0.8)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--split-by", choices=["video", "image"], default="video", help="Keep all frames from the same video together by default")
    prepare.add_argument("--allow-missing-labels", action="store_true", help="Explicitly treat unlabelled images as backgrounds")

    train = commands.add_parser("train", help="Fine-tune a YOLOv8 detector")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--weights", default="yolov8n.pt")
    train.add_argument("--output", type=Path, default=Path("runs/train"))
    train.add_argument("--name", default="enemies")
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--batch", type=int, default=8)
    train.add_argument("--workers", type=int, default=1)
    train.add_argument("--seed", type=int, default=42)

    predict = commands.add_parser("predict", help="Detect soldiers and export bounding boxes to CSV")
    predict.add_argument("--source", required=True, help="Image, video, directory, camera index or stream URL")
    predict.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    predict.add_argument("--output", type=Path, default=Path("runs/predict"))
    predict.add_argument("--name", default="enemies")
    predict.add_argument("--conf", type=float, default=0.25)
    predict.add_argument("--iou", type=float, default=0.7)
    predict.add_argument("--no-save-images", action="store_true")

    validate = commands.add_parser("validate", help="Measure model accuracy on a labelled dataset")
    validate.add_argument("--data", type=Path, required=True)
    validate.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)

    for command in (train, predict, validate):
        command.add_argument("--imgsz", type=int, default=640)
        command.add_argument("--device", default="cpu", help="cpu or CUDA device index such as 0")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        if command == "extract":
            from .frames import extract_frames, parse_segment

            requested = args.pop("segment")
            segments = [parse_segment(value) for value in requested] if requested else None
            result = {"saved_frames": extract_frames(**args, segments=segments)}
        elif command == "prepare":
            from .dataset import prepare_dataset

            args["labels_source"] = args.pop("labels")
            result = prepare_dataset(**args)
        elif command == "train":
            from .engine import train

            result = {"best_weights": train(**args)}
        elif command == "predict":
            from .engine import predict

            args["save_images"] = not args.pop("no_save_images")
            result = predict(**args)
        else:
            from .engine import validate

            result = validate(**args)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
