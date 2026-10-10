"""Extract timestamped frames without overwriting existing images."""

from __future__ import annotations

import math
from pathlib import Path

import cv2


def parse_time(value: str) -> float:
    parts = value.split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"Invalid timestamp: {value}")
    numbers = [float(part) for part in parts]
    if any(not math.isfinite(number) or number < 0 for number in numbers):
        raise ValueError(f"Invalid timestamp: {value}")
    if len(numbers) > 1 and any(number >= 60 for number in numbers[1:]):
        raise ValueError(f"Invalid timestamp: {value}")
    return sum(number * 60**index for index, number in enumerate(reversed(numbers)))


def parse_segment(value: str) -> tuple[float, float]:
    parts = value.split("-")
    if len(parts) != 2:
        raise ValueError("Segments must use START-END, e.g. 00:07-00:20")
    start, end = (parse_time(part) for part in parts)
    if end <= start:
        raise ValueError("Segment end must be after its start")
    return start, end


def extract_frames(
    source: Path,
    output: Path,
    fps: float = 2.0,
    segments: list[tuple[float, float]] | None = None,
) -> int:
    if not source.is_file():
        raise FileNotFoundError(source)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Sampling FPS must be finite and positive")
    capture = cv2.VideoCapture(str(source.resolve()))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {source}")
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(source_fps) or source_fps <= 0 or frame_count <= 0:
            raise ValueError("Video must expose a valid frame count and FPS")
        ranges = segments if segments is not None else [(0.0, frame_count / source_fps)]
        for start, end in ranges:
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                raise ValueError("Invalid segment range")
            if start >= frame_count / source_fps:
                raise ValueError("Segment starts beyond the end of the video")
        output.mkdir(parents=True, exist_ok=True)
        saved = 0
        for segment_id, (start, end) in enumerate(ranges, start=1):
            start_frame = round(start * source_fps)
            end_frame = min(math.ceil(end * source_fps), frame_count)
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            last_sample = -1
            for frame_id in range(start_frame, end_frame):
                ok, frame = capture.read()
                if not ok:
                    break
                sample = math.floor((frame_id - start_frame) * min(fps, source_fps) / source_fps + 1e-9)
                if sample == last_sample:
                    continue
                last_sample = sample
                timestamp = frame_id / source_fps
                target = output / f"{source.stem}_s{segment_id:02d}_t{timestamp:08.3f}_f{frame_id:06d}.jpg"
                if target.exists():
                    continue
                ok, encoded = cv2.imencode(".jpg", frame)
                if not ok:
                    raise RuntimeError(f"Cannot encode frame: {target}")
                # NumPy's file writer supports Windows paths containing non-ASCII characters.
                encoded.tofile(str(target))
                saved += 1
        return saved
    finally:
        capture.release()
