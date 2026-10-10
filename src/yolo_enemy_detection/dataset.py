"""Validate YOLO labels and create deterministic train/validation splits."""

from __future__ import annotations

import math
import random
import shutil
import zipfile
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_labels(source: Path) -> dict[str, str]:
    labels: dict[str, str] = {}

    def add(name: str, content: str) -> None:
        filename = name.replace("\\", "/").rsplit("/", 1)[-1]
        if filename.lower() in {"classes.txt", "train.txt", "val.txt", "test.txt"}:
            return
        stem = Path(filename).stem.casefold()
        if stem in labels:
            raise ValueError(f"Duplicate label name: {filename}")
        labels[stem] = content.strip()

    if source.is_dir():
        for path in sorted(source.rglob("*.txt")):
            add(path.name, path.read_text(encoding="utf-8-sig"))
    elif source.is_file() and zipfile.is_zipfile(source):
        # Read entries directly; never extract untrusted archive paths.
        with zipfile.ZipFile(source) as archive:
            for entry in archive.infolist():
                if not entry.is_dir() and entry.filename.lower().endswith(".txt"):
                    add(entry.filename, archive.read(entry).decode("utf-8-sig"))
    else:
        raise ValueError(f"Labels must be a directory or ZIP archive: {source}")
    return labels


def validate_label(content: str, name: str, class_count: int) -> int:
    objects = 0
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        location = f"{name}:{line_number}"
        if len(fields) != 5:
            raise ValueError(f"{location}: expected class_id x_center y_center width height")
        class_id = int(fields[0])
        if not 0 <= class_id < class_count:
            raise ValueError(f"{location}: class ID outside configured classes")
        x, y, width, height = map(float, fields[1:])
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in (x, y, width, height)):
            raise ValueError(f"{location}: coordinates must be finite and normalized to [0, 1]")
        if width <= 0 or height <= 0:
            raise ValueError(f"{location}: width and height must be positive")
        objects += 1
    return objects


def split_images(
    images: list[Path], train_ratio: float, seed: int, split_by: str
) -> dict[str, list[Path]]:
    if not 0 < train_ratio < 1:
        raise ValueError("Train ratio must be between 0 and 1")
    if split_by not in {"image", "video"}:
        raise ValueError("Split mode must be image or video")
    groups: dict[str, list[Path]] = {}
    for image in sorted(images):
        if split_by == "video":
            if "_s" not in image.stem:
                raise ValueError("Video splitting requires extracted frame names containing '_s'")
            group = image.stem.rsplit("_s", 1)[0]
        else:
            group = image.stem
        groups.setdefault(group, []).append(image)
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError(f"Need at least two distinct {split_by} groups for train/validation")
    random.Random(seed).shuffle(keys)
    split_at = max(1, min(len(keys) - 1, round(len(keys) * train_ratio)))
    return {
        "train": sorted(image for key in keys[:split_at] for image in groups[key]),
        "val": sorted(image for key in keys[split_at:] for image in groups[key]),
    }


def prepare_dataset(
    frames: Path,
    labels_source: Path,
    output: Path,
    classes: list[str],
    train_ratio: float = 0.8,
    seed: int = 42,
    split_by: str = "video",
    allow_missing_labels: bool = False,
) -> dict[str, int | str]:
    if not frames.is_dir():
        raise FileNotFoundError(frames)
    if not classes or len(set(classes)) != len(classes) or any(not name.strip() for name in classes):
        raise ValueError("Class names must be nonempty and unique")
    output = output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output must be a new or empty directory: {output}")
    images = sorted(path for path in frames.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    stems = [image.stem.casefold() for image in images]
    if len(stems) != len(set(stems)):
        raise ValueError("Duplicate image stems would overwrite images or labels")
    labels = read_labels(labels_source)
    missing = sorted(set(stems) - set(labels))
    extra = sorted(set(labels) - set(stems))
    if extra:
        raise ValueError(f"Labels without images: {', '.join(extra[:10])}")
    if missing and not allow_missing_labels:
        raise ValueError(f"Missing labels for {len(missing)} images; add empty labels for backgrounds or use --allow-missing-labels")
    objects = sum(validate_label(content, stem, len(classes)) for stem, content in labels.items())
    splits = split_images(images, train_ratio, seed, split_by)
    for split, split_paths in splits.items():
        images_dir = output / "images" / split
        labels_dir = output / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        for image in split_paths:
            shutil.copy2(image, images_dir / image.name)
            text = labels.get(image.stem.casefold(), "")
            (labels_dir / f"{image.stem}.txt").write_text(text + ("\n" if text else ""), encoding="utf-8")
    config = {
        "path": output.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "nc": len(classes),
        "names": dict(enumerate(classes)),
    }
    config_path = output / "dataset.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return {
        "images": len(images),
        "objects": objects,
        "train_images": len(splits["train"]),
        "val_images": len(splits["val"]),
        "background_images": sum(not labels.get(stem, "").strip() for stem in stems),
        "dataset": str(config_path),
    }
