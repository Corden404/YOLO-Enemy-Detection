"""Training, validation and streaming inference through Ultralytics."""

from __future__ import annotations

import csv
from pathlib import Path

DEFAULT_WEIGHTS = Path(__file__).with_name("weights") / "enemy_yolov8n.pt"
CSV_FIELDS = ["frame_index", "source", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "image_width", "image_height"]


def predict(
    source: str,
    weights: Path = DEFAULT_WEIGHTS,
    output: Path = Path("runs/predict"),
    name: str = "enemies",
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 640,
    device: str = "cpu",
    save_images: bool = True,
) -> dict[str, int | str]:
    from ultralytics import YOLO

    if not weights.is_file():
        raise FileNotFoundError(f"Model not found: {weights}")
    if not 0 <= conf <= 1 or not 0 <= iou <= 1 or imgsz <= 0:
        raise ValueError("Confidence/IoU must be in [0, 1] and image size must be positive")
    # A numeric source is a camera index; image/video paths and stream URLs stay strings.
    input_source = int(source) if source.isdecimal() else source
    model = YOLO(str(weights))
    results = model.predict(
        source=input_source, stream=True, conf=conf, iou=iou, imgsz=imgsz, device=device,
        save=save_images, save_txt=True, save_conf=True,
        project=str(output.resolve()), name=name, exist_ok=False, verbose=False,
    )
    frames = boxes = detected_frames = 0
    csv_file = None
    csv_path = None
    writer = None
    try:
        for frames, result in enumerate(results, start=1):
            if csv_file is None:
                save_dir = Path(result.save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                csv_path = save_dir / "detections.csv"
                csv_file = csv_path.open("w", encoding="utf-8-sig", newline="")
                writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
                writer.writeheader()
            if result.boxes is None:
                continue
            detected_frames += int(len(result.boxes) > 0)
            # Transfer boxes once per frame; keep memory bounded for long videos.
            for x1, y1, x2, y2, confidence, class_id in result.boxes.data.cpu().tolist():
                class_id = int(class_id)
                writer.writerow({
                    "frame_index": frames, "source": Path(result.path).name,
                    "class_id": class_id, "class_name": result.names[class_id],
                    "confidence": round(confidence, 6),
                    "x1": round(x1, 2), "y1": round(y1, 2),
                    "x2": round(x2, 2), "y2": round(y2, 2),
                    "image_width": result.orig_shape[1], "image_height": result.orig_shape[0],
                })
                boxes += 1
    finally:
        if csv_file is not None:
            csv_file.close()
        results.close()
    if csv_path is None:
        raise ValueError("Source produced no frames")
    return {"frames": frames, "detected_frames": detected_frames, "boxes": boxes, "csv": str(csv_path)}


def train(
    data: Path, weights: str = "yolov8n.pt", output: Path = Path("runs/train"),
    name: str = "enemies", epochs: int = 50, batch: int = 8, imgsz: int = 640,
    device: str = "cpu", workers: int = 1, seed: int = 42,
) -> str:
    from ultralytics import YOLO

    if not data.is_file():
        raise FileNotFoundError(data)
    if min(epochs, batch, imgsz) <= 0 or workers < 0:
        raise ValueError("Epochs, batch and image size must be positive; workers must be nonnegative")
    model = YOLO(weights)
    model.train(
        task="detect", data=str(data.resolve()), epochs=epochs, batch=batch, imgsz=imgsz,
        workers=workers, device=device, seed=seed, project=str(output.resolve()), name=name, exist_ok=False,
    )
    return str(model.trainer.best)


def validate(data: Path, weights: Path = DEFAULT_WEIGHTS, imgsz: int = 640, device: str = "cpu") -> dict[str, float]:
    from ultralytics import YOLO

    if not weights.is_file():
        raise FileNotFoundError(weights)
    if not data.is_file():
        raise FileNotFoundError(data)
    if imgsz <= 0:
        raise ValueError("Image size must be positive")
    metrics = YOLO(str(weights)).val(data=str(data.resolve()), imgsz=imgsz, device=device)
    return {key: float(value) for key, value in metrics.results_dict.items()}
