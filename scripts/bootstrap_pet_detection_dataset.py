from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _resolve_device(requested: str) -> str | None:
    normalized = requested.strip().lower()
    if normalized == "auto":
        try:
            import torch
        except ModuleNotFoundError:
            return None
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return normalized


def _iter_images(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> str:
    box_w = max(x2 - x1, 1.0)
    box_h = max(y2 - y1, 1.0)
    center_x = x1 + box_w / 2.0
    center_y = y1 + box_h / 2.0
    return (
        f"{center_x / width:.6f} "
        f"{center_y / height:.6f} "
        f"{box_w / width:.6f} "
        f"{box_h / height:.6f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bootstrap a YOLO detection dataset from cat/dog classification folders."
    )
    parser.add_argument(
        "--cat-dir",
        action="append",
        default=[],
        help="Folder containing cat images. Can be passed multiple times.",
    )
    parser.add_argument(
        "--dog-dir",
        action="append",
        default=[],
        help="Folder containing dog images. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output root for the generated YOLO detection dataset.",
    )
    parser.add_argument("--model", default="yolo26m.pt", help="Detection model used for pseudo-labeling.")
    parser.add_argument("--device", default="mps", help="Inference device for pseudo-labeling.")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.20, help="Minimum confidence for keeping a box.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val splitting.")
    parser.add_argument("--max-per-class", type=int, default=0, help="Optional cap per class. 0 means no cap.")
    parser.add_argument("--cat-class-id", type=int, default=1, help="YOLO class id to use for cats.")
    parser.add_argument("--dog-class-id", type=int, default=2, help="YOLO class id to use for dogs.")
    return parser


def _pick_best_box(result, target_class_id: int):
    if result.boxes is None:
        return None

    best = None
    best_conf = -1.0
    for box in result.boxes:
        class_id = int(box.cls.item())
        if class_id != target_class_id:
            continue
        confidence = float(box.conf.item())
        if confidence > best_conf:
            best_conf = confidence
            best = box
    return best


def _write_label(label_path: Path, class_id: int, box, width: int, height: int) -> None:
    x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
    yolo_box = _xyxy_to_yolo(x1, y1, x2, y2, width, height)
    label_path.write_text(f"{class_id} {yolo_box}\n", encoding="utf-8")


def _copy_example(image_path: Path, image_out: Path) -> None:
    image_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, image_out)


def _ensure_dataset_dirs(root: Path) -> None:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def _write_dataset_yaml(root: Path) -> Path:
    dataset_yaml = root / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {root.resolve()}",
                "train: images/train",
                "val: images/val",
                "",
                "names:",
                "  0: person",
                "  1: cat",
                "  2: dog",
                "  3: car",
                "  4: car_plate",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return dataset_yaml


def _prepare_samples(paths: list[str], limit: int, seed: int) -> list[Path]:
    images: list[Path] = []
    for raw in paths:
        folder = Path(raw).expanduser().resolve()
        if not folder.exists():
            raise SystemExit(f"Input folder not found: {folder}")
        images.extend(_iter_images(folder))

    if not images:
        return []

    rng = random.Random(seed)
    rng.shuffle(images)
    if limit > 0:
        return images[:limit]
    return images


def main() -> int:
    args = build_parser().parse_args()
    if not args.cat_dir and not args.dog_dir:
        raise SystemExit("Provide at least one --cat-dir or --dog-dir.")

    output_root = Path(args.output).expanduser().resolve()
    _ensure_dataset_dirs(output_root)

    model = YOLO(args.model)
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    class_name_to_id = {str(name).lower(): int(class_id) for class_id, name in names.items()}
    if "cat" not in class_name_to_id or "dog" not in class_name_to_id:
        raise SystemExit("The selected model does not expose both `cat` and `dog` classes.")

    detector_class_ids = {"cat": class_name_to_id["cat"], "dog": class_name_to_id["dog"]}
    device = _resolve_device(args.device)

    samples_by_class = {
        "cat": _prepare_samples(args.cat_dir, args.max_per_class, args.seed),
        "dog": _prepare_samples(args.dog_dir, args.max_per_class, args.seed + 1),
    }
    yolo_class_ids = {"cat": args.cat_class_id, "dog": args.dog_class_id}
    kept_counts = {"cat": 0, "dog": 0}
    skipped_counts = {"cat": 0, "dog": 0}

    for class_name, samples in samples_by_class.items():
        if not samples:
            continue

        split_index = max(int(len(samples) * (1.0 - args.val_ratio)), 1)
        for idx, image_path in enumerate(samples):
            split = "train" if idx < split_index else "val"
            results = model.predict(
                source=str(image_path),
                conf=args.conf,
                imgsz=args.imgsz,
                device=device,
                classes=[detector_class_ids[class_name]],
                verbose=False,
            )
            result = results[0]
            box = _pick_best_box(result, detector_class_ids[class_name])
            if box is None:
                skipped_counts[class_name] += 1
                continue

            height, width = result.orig_shape
            stem = f"{class_name}_{kept_counts[class_name]:06d}"
            image_out = output_root / "images" / split / f"{stem}{image_path.suffix.lower()}"
            label_out = output_root / "labels" / split / f"{stem}.txt"
            _copy_example(image_path, image_out)
            _write_label(label_out, yolo_class_ids[class_name], box, width, height)
            kept_counts[class_name] += 1

    dataset_yaml = _write_dataset_yaml(output_root)
    print("Bootstrap dataset created.")
    print(f"  dataset_root: {output_root}")
    print(f"  dataset_yaml: {dataset_yaml}")
    print(f"  kept cats: {kept_counts['cat']}, skipped cats: {skipped_counts['cat']}")
    print(f"  kept dogs: {kept_counts['dog']}, skipped dogs: {skipped_counts['dog']}")
    print("Review the generated labels before using them for final training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
