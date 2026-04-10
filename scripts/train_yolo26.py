from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def _parse_batch(raw: str) -> int | float:
    value = raw.strip()
    if "." in value:
        return float(value)
    return int(value)


def _parse_cache(raw: str) -> bool | str:
    normalized = raw.strip().lower()
    if normalized in {"false", "off", "none", "0"}:
        return False
    if normalized in {"true", "ram", "1", "on"}:
        return True
    if normalized == "disk":
        return "disk"
    raise argparse.ArgumentTypeError("cache must be one of: false, true, ram, disk")


def _default_workers() -> int:
    # On macOS, Ultralytics + MPS is often much more reliable with a single-process
    # dataloader startup path. This avoids the common "looks frozen at launch" issue.
    return 0 if sys.platform == "darwin" else 4


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

    if normalized == "mps":
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PyTorch is not installed in the active environment, so MPS training is unavailable."
            ) from exc
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS was requested but is not available in this runtime. "
                "Use the local macOS .venv on Apple Silicon, or set --device cpu."
            )
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune a YOLO26 detection model for home surveillance."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to a YOLO dataset YAML file.",
    )
    parser.add_argument(
        "--model",
        default="yolo26m.pt",
        help="Pretrained YOLO26 weights to fine-tune.",
    )
    parser.add_argument(
        "--device",
        default="mps",
        help="Training device. Use mps on Apple Silicon, cpu if needed, or auto.",
    )
    parser.add_argument("--epochs", type=int, default=300, help="Maximum training epochs.")
    parser.add_argument("--imgsz", type=int, default=960, help="Training image size.")
    parser.add_argument(
        "--batch",
        type=_parse_batch,
        default=8,
        help="Batch size. Can be an int or a float for auto-batch fractions.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early-stopping patience in epochs.",
    )
    parser.add_argument(
        "--cache",
        type=_parse_cache,
        default="disk",
        help="Dataset cache mode: false, true/ram, or disk.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of the dataset to use. Helpful for quick experiments.",
    )
    parser.add_argument(
        "--project",
        default="runs",
        help="Directory where Ultralytics stores training runs.",
    )
    parser.add_argument(
        "--name",
        default="home-surveillance-yolo26m",
        help="Run name inside the project directory.",
    )
    parser.add_argument(
        "--optimizer",
        default="auto",
        help="Optimizer choice, e.g. auto, SGD, AdamW.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=_default_workers(),
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=0,
        help="Freeze the first N layers for transfer learning experiments.",
    )
    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=10,
        help="Disable mosaic in the last N epochs.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="Image scale augmentation strength.",
    )
    parser.add_argument(
        "--multi-scale",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable multiscale training for more robust size handling.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable automatic mixed precision.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the most recent compatible Ultralytics run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for repeatable experiments.",
    )
    return parser


def _load_dataset_config(data_path: Path) -> dict[str, object]:
    try:
        raw = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"Could not read dataset YAML: {data_path}") from exc
    if not isinstance(raw, dict):
        raise SystemExit(f"Dataset YAML must contain a mapping: {data_path}")
    return raw


def _resolve_dataset_root(data_path: Path, config: dict[str, object]) -> Path:
    raw_path = str(config.get("path", "")).strip()
    if not raw_path:
        raise SystemExit(
            "Dataset YAML is missing `path:`. Point it to the dataset root that contains "
            "`images/train`, `images/val`, `labels/train`, and `labels/val`."
        )
    if raw_path.startswith("/absolute/path/to/"):
        raise SystemExit(
            "Dataset YAML still contains the placeholder `path:` value. "
            "Edit the YAML and set `path:` to your real dataset root."
        )

    dataset_root = Path(raw_path).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (data_path.parent / dataset_root).resolve()
    if not dataset_root.exists():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")
    return dataset_root


def _validate_dataset_split(dataset_root: Path, config: dict[str, object], split: str) -> None:
    relative = str(config.get(split, "")).strip()
    if not relative:
        raise SystemExit(f"Dataset YAML is missing `{split}:`.")
    split_dir = dataset_root / relative
    if not split_dir.exists():
        raise SystemExit(f"Missing dataset directory for `{split}`: {split_dir}")

    labels_relative = relative.replace("images/", "labels/", 1)
    labels_dir = dataset_root / labels_relative
    if not labels_dir.exists():
        raise SystemExit(
            f"Missing labels directory for `{split}`: {labels_dir}. "
            "YOLO detection training requires both images and labels."
        )


def main() -> int:
    args = build_parser().parse_args()
    print("Validating dataset configuration...", flush=True)
    data_path = Path(args.data).expanduser().resolve()
    if not data_path.exists():
        raise SystemExit(f"Dataset YAML not found: {data_path}")

    dataset_config = _load_dataset_config(data_path)
    dataset_root = _resolve_dataset_root(data_path, dataset_config)
    _validate_dataset_split(dataset_root, dataset_config, "train")
    _validate_dataset_split(dataset_root, dataset_config, "val")

    print("Importing Ultralytics...", flush=True)
    from ultralytics import YOLO

    print("Resolving training device...", flush=True)
    device = _resolve_device(args.device)
    print("Starting YOLO26 fine-tuning with:")
    print(f"  data: {data_path}")
    print(f"  dataset_root: {dataset_root}")
    print(f"  model: {args.model}")
    print(f"  device: {device or 'default'}")
    print(f"  epochs: {args.epochs}")
    print(f"  imgsz: {args.imgsz}")
    print(f"  batch: {args.batch}")
    print(f"  cache: {args.cache}")
    print(f"  fraction: {args.fraction}")
    print(f"  workers: {args.workers}")
    print(f"  project/name: {args.project}/{args.name}")
    if sys.platform == "darwin" and args.workers > 0:
        print(
            "  note: macOS training is usually more stable with --workers 0. "
            "If startup appears stalled, retry with --workers 0.",
            flush=True,
        )

    print("Loading model weights...", flush=True)
    model = YOLO(args.model)
    print("Starting Ultralytics training loop...", flush=True)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=device,
        batch=args.batch,
        patience=args.patience,
        cache=args.cache,
        fraction=args.fraction,
        project=args.project,
        name=args.name,
        optimizer=args.optimizer,
        workers=args.workers,
        freeze=args.freeze,
        close_mosaic=args.close_mosaic,
        scale=args.scale,
        multi_scale=args.multi_scale,
        amp=args.amp,
        seed=args.seed,
        pretrained=True,
        exist_ok=True,
        resume=args.resume,
        val=True,
        plots=True,
        verbose=True,
    )

    save_dir = getattr(results, "save_dir", None)
    if save_dir is not None:
        print(f"Training complete. Run artifacts saved to: {save_dir}")
        print(f"Best weights should be in: {Path(save_dir) / 'weights' / 'best.pt'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
