from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

import cv2
import numpy as np
import yaml

from yolo_enemy_detection.dataset import prepare_dataset, read_labels, split_images, validate_label
from yolo_enemy_detection.frames import extract_frames, parse_segment, parse_time


class DatasetTests(unittest.TestCase):
    def setUp(self):
        # Keep generated test data in the project, available for inspection after failures.
        Path(".trash").mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="dataset-", dir=".trash"))
        self.frames = self.root / "frames"
        self.labels = self.root / "labels"
        self.frames.mkdir()
        self.labels.mkdir()
        for video in ("alpha", "beta"):
            for frame in range(3):
                stem = f"{video}_s01_t000{frame}.000_f{frame:06d}"
                cv2.imencode(".jpg", np.zeros((24, 32, 3), dtype=np.uint8))[1].tofile(str(self.frames / f"{stem}.jpg"))
                (self.labels / f"{stem}.txt").write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
        self.output = self.root / "dataset"

    def test_build_keeps_videos_separate_and_yaml_resolves_from_other_directory(self):
        stats = prepare_dataset(self.frames, self.labels, self.output, ["soldier"])
        self.assertEqual(stats["images"], 6)
        self.assertEqual(stats["objects"], 6)
        train = {path.stem.split("_s")[0] for path in (self.output / "images/train").iterdir()}
        val = {path.stem.split("_s")[0] for path in (self.output / "images/val").iterdir()}
        self.assertFalse(train & val)
        config = yaml.safe_load((self.output / "dataset.yaml").read_text(encoding="utf-8"))
        self.assertTrue(Path(config["path"]).is_absolute())
        self.assertEqual(config["names"], {0: "soldier"})
        self.assertEqual(len(list(self.output.glob("labels/*/*.txt"))), 6)

    def test_missing_labels_require_explicit_background_opt_in(self):
        omitted = next(self.labels.glob("*.txt"))
        omitted.rename(self.root / omitted.name)
        with self.assertRaisesRegex(ValueError, "Missing labels"):
            prepare_dataset(self.frames, self.labels, self.output, ["soldier"])
        self.assertFalse(self.output.exists())
        stats = prepare_dataset(self.frames, self.labels, self.output, ["soldier"], allow_missing_labels=True)
        self.assertEqual(stats["background_images"], 1)

    def test_duplicate_image_stems_are_rejected_before_writing(self):
        duplicate_dir = self.frames / "other"
        duplicate_dir.mkdir()
        first = next(self.frames.glob("*.jpg"))
        (duplicate_dir / first.name).write_bytes(first.read_bytes())
        with self.assertRaisesRegex(ValueError, "Duplicate image"):
            prepare_dataset(self.frames, self.labels, self.output, ["soldier"])
        self.assertFalse(self.output.exists())

    def test_existing_dataset_is_preserved(self):
        self.output.mkdir()
        existing = self.output / "keep.bin"
        existing.write_bytes(b"original")
        with self.assertRaisesRegex(ValueError, "new or empty"):
            prepare_dataset(self.frames, self.labels, self.output, ["soldier"])
        self.assertEqual(existing.read_bytes(), b"original")

    def test_orphan_labels_are_rejected(self):
        (self.labels / "orphan.txt").write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Labels without images"):
            prepare_dataset(self.frames, self.labels, self.output, ["soldier"])

    def test_zip_duplicate_labels_are_rejected(self):
        archive_path = self.root / "labels.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("one/frame.txt", "")
            archive.writestr("two/frame.txt", "0 0.5 0.5 0.2 0.2")
        with self.assertRaisesRegex(ValueError, "Duplicate label"):
            read_labels(archive_path)

    def test_zip_is_read_without_extracting_paths(self):
        archive_path = self.root / "safe.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("../../outside.txt", "0 0.5 0.5 0.2 0.2")
            archive.writestr("classes.txt", "soldier")
        self.assertEqual(set(read_labels(archive_path)), {"outside"})
        self.assertFalse((self.root / "outside.txt").exists())

    def test_invalid_boxes_and_class_ids_are_rejected(self):
        for line in ("1 0.5 0.5 0.2 0.2", "0 nan 0.5 0.2 0.2", "0 0.5 0.5 0 0.2", "0 1.1 0.5 0.2 0.2", "0 0.5 0.5"):
            with self.subTest(line=line), self.assertRaises(ValueError):
                validate_label(line, "frame.txt", 1)

    def test_split_is_deterministic_and_nonempty(self):
        images = sorted(self.frames.glob("*.jpg"))
        first = split_images(images, 0.99, 42, "image")
        self.assertEqual(first, split_images(images, 0.99, 42, "image"))
        self.assertTrue(first["train"] and first["val"])
        self.assertFalse(set(first["train"]) & set(first["val"]))


class FrameTests(unittest.TestCase):
    def test_timestamps_and_ranges(self):
        self.assertEqual(parse_time("1:02:03.5"), 3723.5)
        self.assertEqual(parse_segment("00:07-00:20"), (7.0, 20.0))
        for value in ("nan", "-1", "1:60", "1:2:3:4"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_time(value)
        with self.assertRaises(ValueError):
            parse_segment("20-7")

    def test_noninteger_sampling_and_repeat_run_preserve_frames(self):
        Path(".trash").mkdir(exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="frames-", dir=".trash"))
        video = root / "sample.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24))
        self.assertTrue(writer.isOpened())
        try:
            for value in range(20):
                writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
        finally:
            writer.release()
        output = root / "图像"
        self.assertEqual(extract_frames(video, output, fps=3), 6)
        before = {path.name: path.read_bytes() for path in output.iterdir()}
        self.assertEqual(extract_frames(video, output, fps=3), 0)
        self.assertEqual(before, {path.name: path.read_bytes() for path in output.iterdir()})
        with self.assertRaisesRegex(ValueError, "beyond"):
            extract_frames(video, root / "invalid", segments=[(10, 11)])
        self.assertFalse((root / "invalid").exists())


if __name__ == "__main__":
    Path(".trash").mkdir(exist_ok=True)
    unittest.main()
