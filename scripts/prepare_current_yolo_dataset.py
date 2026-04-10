from __future__ import annotations

import argparse
import csv
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO


CLASS_IDS = {
    "person": 0,
    "cat": 1,
    "dog": 2,
    "car": 3,
    "car_plate": 4,
}
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


def _iter_images(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _ensure_clean_output(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise SystemExit(
                f"Output dataset already exists: {output_root}. "
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(output_root)

    for split in ("train", "val"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def _write_dataset_yaml(output_root: Path) -> Path:
    dataset_yaml = output_root / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {output_root.resolve()}",
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


def _write_example(
    output_root: Path,
    split: str,
    stem: str,
    image_path: Path,
    label_lines: list[str],
) -> None:
    image_out = output_root / "images" / split / f"{stem}{image_path.suffix.lower()}"
    label_out = output_root / "labels" / split / f"{stem}.txt"
    image_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, image_out)
    label_out.write_text("\n".join(label_lines) + "\n", encoding="utf-8")


def _split_items(items: list[Path], val_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) <= 1:
        return shuffled, []
    val_count = max(int(len(shuffled) * val_ratio), 1)
    train_count = max(len(shuffled) - val_count, 1)
    return shuffled[:train_count], shuffled[train_count:]


def _load_dt_car_annotations(csv_path: Path) -> dict[str, list[tuple[float, float, float, float]]]:
    annotations: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_name = row["image"].strip()
            annotations[image_name].append(
                (
                    float(row["xmin"]),
                    float(row["ymin"]),
                    float(row["xmax"]),
                    float(row["ymax"]),
                )
            )
    return annotations


def _prepare_car_dataset(output_root: Path, val_ratio: float, seed: int) -> dict[str, int]:
    images_dir = Path("data/dt_car/training_images").resolve()
    csv_path = Path("data/dt_car/train_solution_bounding_boxes (1).csv").resolve()
    if not images_dir.exists() or not csv_path.exists():
        return {"train": 0, "val": 0}

    annotations = _load_dt_car_annotations(csv_path)
    image_paths = [images_dir / name for name in sorted(annotations)]
    image_paths = [path for path in image_paths if path.exists()]
    train_items, val_items = _split_items(image_paths, val_ratio=val_ratio, seed=seed)

    counts = {"train": 0, "val": 0}
    for split, items in (("train", train_items), ("val", val_items)):
        for index, image_path in enumerate(items):
            from PIL import Image

            with Image.open(image_path) as image:
                width, height = image.size
            label_lines = [
                f"{CLASS_IDS['car']} {_xyxy_to_yolo(x1, y1, x2, y2, width, height)}"
                for x1, y1, x2, y2 in annotations[image_path.name]
            ]
            _write_example(
                output_root=output_root,
                split=split,
                stem=f"car_{index:06d}",
                image_path=image_path,
                label_lines=label_lines,
            )
            counts[split] += 1
    return counts


def _parse_voc_label(xml_path: Path) -> tuple[str, list[str]] | None:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.findtext("filename")
    width = int(root.findtext("size/width", "0"))
    height = int(root.findtext("size/height", "0"))
    if not filename or width <= 0 or height <= 0:
        return None

    label_lines: list[str] = []
    for obj in root.findall("object"):
        class_name = (obj.findtext("name", "") or "").strip().lower()
        if class_name not in {"licence", "license", "license_plate", "car_plate"}:
            continue
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        x1 = float(bbox.findtext("xmin", "0"))
        y1 = float(bbox.findtext("ymin", "0"))
        x2 = float(bbox.findtext("xmax", "0"))
        y2 = float(bbox.findtext("ymax", "0"))
        label_lines.append(
            f"{CLASS_IDS['car_plate']} {_xyxy_to_yolo(x1, y1, x2, y2, width, height)}"
        )
    if not label_lines:
        return None
    return filename, label_lines


def _prepare_car_plate_dataset(output_root: Path, val_ratio: float, seed: int) -> dict[str, int]:
    images_dir = Path("data/dt_car_plates/images").resolve()
    xml_dir = Path("data/dt_car_plates/annotations").resolve()
    if not images_dir.exists() or not xml_dir.exists():
        return {"train": 0, "val": 0}

    xml_paths = sorted(xml_dir.glob("*.xml"))
    train_items, val_items = _split_items(xml_paths, val_ratio=val_ratio, seed=seed)

    counts = {"train": 0, "val": 0}
    for split, items in (("train", train_items), ("val", val_items)):
        for index, xml_path in enumerate(items):
            parsed = _parse_voc_label(xml_path)
            if parsed is None:
                continue
            filename, label_lines = parsed
            image_path = images_dir / filename
            if not image_path.exists():
                continue
            _write_example(
                output_root=output_root,
                split=split,
                stem=f"car_plate_{index:06d}",
                image_path=image_path,
                label_lines=label_lines,
            )
            counts[split] += 1
    return counts


def _load_dt_cat_dog_from_csv() -> dict[str, list[Path]]:
    csv_path = Path("data/dt_cat_dog/cat_dog.csv").resolve()
    images_dir = Path("data/dt_cat_dog/cat_dog").resolve()
    samples = {"cat": [], "dog": []}
    if not csv_path.exists() or not images_dir.exists():
        return samples

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = str(row.get("labels", "")).strip()
            image_name = str(row.get("image", "")).strip()
            image_path = images_dir / image_name
            if not image_path.exists():
                continue
            if label == "0":
                samples["cat"].append(image_path)
            elif label == "1":
                samples["dog"].append(image_path)
    return samples


def _load_pet_sources() -> dict[str, list[Path]]:
    samples = {"cat": [], "dog": []}
    csv_samples = _load_dt_cat_dog_from_csv()
    samples["cat"].extend(csv_samples["cat"])
    samples["dog"].extend(csv_samples["dog"])

    samples["cat"].extend(_iter_images(Path("data/dt_cat_dog_2/Cat").resolve()))
    samples["dog"].extend(_iter_images(Path("data/dt_cat_dog_2/Dog").resolve()))
    samples["cat"].extend(_iter_images(Path("data/dt_cat_dog_3/training_set/training_set/cats").resolve()))
    samples["dog"].extend(_iter_images(Path("data/dt_cat_dog_3/training_set/training_set/dogs").resolve()))
    samples["cat"].extend(_iter_images(Path("data/dt_cat_dog_3/test_set/test_set/cats").resolve()))
    samples["dog"].extend(_iter_images(Path("data/dt_cat_dog_3/test_set/test_set/dogs").resolve()))
    return samples


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


def _prepare_pet_dataset(
    output_root: Path,
    model_name: str,
    device: str | None,
    imgsz: int,
    conf: float,
    val_ratio: float,
    seed: int,
    max_per_class: int,
) -> dict[str, dict[str, int]]:
    model = YOLO(model_name)
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    name_to_id = {str(name).lower(): int(class_id) for class_id, name in names.items()}
    if "cat" not in name_to_id or "dog" not in name_to_id:
        raise SystemExit(f"{model_name} does not expose both cat and dog classes.")

    sources = _load_pet_sources()
    summary = {
        "cat": {"kept_train": 0, "kept_val": 0, "skipped": 0},
        "dog": {"kept_train": 0, "kept_val": 0, "skipped": 0},
    }

    for class_name in ("cat", "dog"):
        samples = list(dict.fromkeys(sources[class_name]))
        random.Random(seed + (1 if class_name == "dog" else 0)).shuffle(samples)
        if max_per_class > 0:
            samples = samples[:max_per_class]
        train_items, val_items = _split_items(samples, val_ratio=val_ratio, seed=seed)

        for split, items in (("train", train_items), ("val", val_items)):
            kept_key = "kept_train" if split == "train" else "kept_val"
            for index, image_path in enumerate(items):
                results = model.predict(
                    source=str(image_path),
                    conf=conf,
                    imgsz=imgsz,
                    device=device,
                    classes=[name_to_id[class_name]],
                    verbose=False,
                )
                result = results[0]
                box = _pick_best_box(result, name_to_id[class_name])
                if box is None:
                    summary[class_name]["skipped"] += 1
                    continue
                height, width = result.orig_shape
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                label_lines = [
                    f"{CLASS_IDS[class_name]} {_xyxy_to_yolo(x1, y1, x2, y2, width, height)}"
                ]
                _write_example(
                    output_root=output_root,
                    split=split,
                    stem=f"{class_name}_{index:06d}",
                    image_path=image_path,
                    label_lines=label_lines,
                )
                summary[class_name][kept_key] += 1
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a trainable YOLO dataset from the current local Kaggle datasets."
    )
    parser.add_argument(
        "--output",
        default="datasets/current_detect_dataset",
        help="Output YOLO dataset root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output dataset if it already exists.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio for converted datasets.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--pet-model",
        default="yolo26m.pt",
        help="Model used to pseudo-label pet datasets.",
    )
    parser.add_argument(
        "--pet-device",
        default="mps",
        help="Device used for pet pseudo-labeling.",
    )
    parser.add_argument("--pet-imgsz", type=int, default=960, help="Pseudo-labeling image size.")
    parser.add_argument("--pet-conf", type=float, default=0.20, help="Pseudo-labeling confidence threshold.")
    parser.add_argument(
        "--pet-max-per-class",
        type=int,
        default=3000,
        help="Limit the number of cat and dog images used for pseudo-labeling. 0 means no cap.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_root = Path(args.output).expanduser().resolve()
    _ensure_clean_output(output_root, overwrite=args.overwrite)

    car_counts = _prepare_car_dataset(output_root, val_ratio=args.val_ratio, seed=args.seed)
    plate_counts = _prepare_car_plate_dataset(output_root, val_ratio=args.val_ratio, seed=args.seed + 17)
    pet_summary = _prepare_pet_dataset(
        output_root=output_root,
        model_name=args.pet_model,
        device=_resolve_device(args.pet_device),
        imgsz=args.pet_imgsz,
        conf=args.pet_conf,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_per_class=args.pet_max_per_class,
    )
    dataset_yaml = _write_dataset_yaml(output_root)

    print("Prepared trainable YOLO dataset.")
    print(f"  dataset_root: {output_root}")
    print(f"  dataset_yaml: {dataset_yaml}")
    print(f"  cars train/val: {car_counts['train']}/{car_counts['val']}")
    print(f"  car_plates train/val: {plate_counts['train']}/{plate_counts['val']}")
    print(
        "  pets kept train/val/skipped: "
        f"cat={pet_summary['cat']['kept_train']}/{pet_summary['cat']['kept_val']}/{pet_summary['cat']['skipped']}, "
        f"dog={pet_summary['dog']['kept_train']}/{pet_summary['dog']['kept_val']}/{pet_summary['dog']['skipped']}"
    )
    print("  person labels: 0 (not available in current local datasets)")
    print("Review pseudo-labeled pet boxes before long training runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
