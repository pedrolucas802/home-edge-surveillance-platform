# Home Edge Surveillance Training Runbook

## Purpose

This runbook is the operating guide for local dataset preparation, YOLO fine-tuning, and model iteration.

Use it for:

- preparing a trainable YOLO dataset from the local `data/` folder
- running quick training experiments on the MacBook
- running a full `yolo26m` fine-tuning pass
- understanding what the current datasets can and cannot improve
- promoting the best checkpoint back into the live surveillance app

This runbook is intentionally separate from the main operations guide in [docs/RUNBOOK.md](RUNBOOK.md).

## Current Baseline

The current promoted checkpoint is:

```text
models/home-surveillance-yolo26m-best.pt
```

It was promoted from the raw training run at:

```text
runs/detect/runs/home-surveillance-yolo26m/
```

Baseline summary from April 9, 2026:

- early stopped at epoch `117` with `patience=30`
- best epoch: `87`
- total wall time: `28.956` hours
- best validation metrics: `Precision=0.883`, `Recall=0.882`, `mAP50=0.914`, `mAP50-95=0.608`

Per-class `mAP50-95` for the promoted checkpoint:

- `cat`: `0.576`
- `dog`: `0.630`
- `car`: `0.700`
- `car_plate`: `0.524`

Current next-loop priorities:

- improve `car_plate` quality first
- improve `cat` quality next
- add real home-camera `person` labels before expecting person gains

## Current Training Goal

The current training priority is:

1. pets
2. car-related objects
3. people data later

With the datasets currently available in `data/`, the realistic near-term improvement target is:

- `cat`
- `dog`
- `car`
- `car_plate`

`person` remains part of the class schema, but current local training data does not yet include labeled person detections. Until that changes, person quality will still depend mostly on the pretrained YOLO weights.

## Class Map

All datasets and labels should follow this class order:

```text
0: person
1: cat
2: dog
3: car
4: car_plate
```

Keep this mapping stable across all future dataset conversion scripts and manual labeling work.

## Current Local Dataset Inventory

- `data/dt_cat_dog`
  - classification dataset
  - used for pseudo-labeling `cat` and `dog`
- `data/dt_cat_dog_2`
  - classification dataset
  - used for pseudo-labeling `cat` and `dog`
- `data/dt_cat_dog_3`
  - classification dataset with train/test split
  - used for pseudo-labeling `cat` and `dog`
- `data/dt_car`
  - detection-ready after CSV conversion
  - contributes labeled `car` boxes
- `data/dt_car_plates`
  - detection-ready after Pascal VOC XML conversion
  - contributes labeled `car_plate` boxes
- `data/dt_racd`
  - not part of the current detection training path
  - useful later for event mining, evaluation, or future data collection

## Training Workflow

### 1. Prepare the Python environment

```bash
cd /Users/patriciarego/Desktop/dev/home-edge-surveillance-platform
source .venv/bin/activate
pip install -r requirements.txt
```

Use the local `.venv` on the MacBook so PyTorch can use Apple `mps`.

The current successful run used a local Python 3.12 environment (`.venv312`). If your
local environment name differs, substitute it in the example commands below.

On macOS, training startup can look stalled if Ultralytics spends time importing, scanning the dataset, or launching dataloaders. Start with `--workers 0` and `--cache false` for the most predictable first run.

### 2. Build the current unified YOLO dataset

This step creates one trainable dataset from the current local `data/` folders.

```bash
.venv/bin/python scripts/prepare_current_yolo_dataset.py \
  --output datasets/current_detect_dataset \
  --overwrite \
  --pet-model yolo26m.pt \
  --pet-device mps \
  --pet-max-per-class 1500
```

What this does:

- converts `dt_car` CSV annotations into YOLO labels for `car`
- converts `dt_car_plates` XML annotations into YOLO labels for `car_plate`
- pseudo-labels pet images from `dt_cat_dog`, `dt_cat_dog_2`, and `dt_cat_dog_3`
- writes a merged YOLO dataset under `datasets/current_detect_dataset`

The generated dataset YAML will be:

```text
datasets/current_detect_dataset/dataset.yaml
```

### 3. Inspect the prepared dataset

Quick sanity checks:

```bash
find datasets/current_detect_dataset/images/train -type f | wc -l
find datasets/current_detect_dataset/images/val -type f | wc -l
find datasets/current_detect_dataset/labels/train -type f | wc -l
find datasets/current_detect_dataset/labels/val -type f | wc -l
```

Open a few labels manually before training if you want to spot-check pseudo-labeled pets.

### 4. Run a quick experiment first

Use a shorter run before committing to a full 300-epoch job.

```bash
.venv/bin/python scripts/train_yolo26.py \
  --data datasets/current_detect_dataset/dataset.yaml \
  --model yolo26m.pt \
  --device mps \
  --workers 0 \
  --cache false \
  --imgsz 960 \
  --batch 4 \
  --epochs 80 \
  --patience 15
```

This is the best first pass when testing whether the prepared dataset is healthy.

### 5. Run the main fine-tuning job

Once the quick experiment looks healthy, run the heavier training pass:

```bash
.venv/bin/python scripts/train_yolo26.py \
  --data datasets/current_detect_dataset/dataset.yaml \
  --model yolo26m.pt \
  --device mps \
  --workers 0 \
  --cache false \
  --imgsz 960 \
  --batch 8 \
  --epochs 300 \
  --patience 30
```

If MPS memory becomes unstable, lower `--batch` to `4`.

For smaller targets like pets and plates, a stronger variant is:

```bash
.venv/bin/python scripts/train_yolo26.py \
  --data datasets/current_detect_dataset/dataset.yaml \
  --model yolo26m.pt \
  --device mps \
  --workers 0 \
  --cache false \
  --imgsz 1280 \
  --batch 4 \
  --epochs 300 \
  --patience 30 \
  --multi-scale
```

## Outputs

Ultralytics writes training runs under `runs/`.

Raw output location for the current baseline:

```text
runs/detect/runs/home-surveillance-yolo26m/
```

Most important files:

- `runs/detect/runs/home-surveillance-yolo26m/weights/best.pt`
- `runs/detect/runs/home-surveillance-yolo26m/weights/last.pt`
- `runs/detect/runs/home-surveillance-yolo26m/results.csv`
- `runs/detect/runs/home-surveillance-yolo26m/results.png`
- `runs/detect/runs/home-surveillance-yolo26m/confusion_matrix.png`

Promoted repo-tracked artifacts:

- `models/home-surveillance-yolo26m-best.pt`
- `docs/assets/yolo26m-960-training-results.png`

## Use the Fine-Tuned Model in the App

After training, test the best checkpoint in the live pipeline:

```bash
.venv/bin/python -m app.main \
  --camera cam1 \
  --enable-yolo \
  --yolo-model models/home-surveillance-yolo26m-best.pt \
  --yolo-device mps \
  --yolo-classes person cat dog car car_plate
```

If you later add `car_plate` rendering or event logic to the live app, keep the same trained checkpoint and extend the runtime class filter there as needed.

## Current Limitations

- `person` is in the schema, but current prepared training data does not contain real person labels.
- pet labels are currently pseudo-labeled, so some cat and dog boxes will need manual review.
- `dt_racd` is not yet part of the training set.
- car-plate quality depends heavily on image resolution and annotation consistency.

## Troubleshooting

### MPS runs out of memory

Reduce load:

- lower `--batch` from `8` to `4`
- lower `--imgsz` from `1280` to `960`
- keep the quick experiment path for early validation

### Training appears stuck before the first epoch

This usually means the job is still in startup, not that it is dead.

Use this safer smoke test:

```bash
.venv/bin/python -u scripts/train_yolo26.py \
  --data datasets/current_detect_dataset/dataset.yaml \
  --model yolo26m.pt \
  --device mps \
  --workers 0 \
  --cache false \
  --imgsz 640 \
  --batch 2 \
  --epochs 1 \
  --patience 1
```

If that starts cleanly, move up to the normal quick experiment.

### Training quality is weak on cats or dogs

Common causes:

- pseudo-labeled pet boxes are noisy
- source images are too varied or too low quality
- the dataset is imbalanced between cats and dogs

Practical response:

- inspect a sample of generated pet labels
- rerun dataset preparation with a smaller, cleaner pet subset
- add home-camera pet frames later

### Training starts but `person` does not improve

That is expected with the current data. Person quality will not materially improve until you add real person-labeled examples.

## Recommended Next Improvements

- add home-camera person detections to the training data
- add home-camera pet frames from the actual surveillance cameras
- curate a small holdout set from your own environment for final validation
- clean up pseudo-labeled pet boxes before longer training runs
- fold in future car and plate datasets only after class-map consistency is confirmed

## Dataset Backlog

Already included locally in `data/`:

- `dt_car`: car object detection dataset
- `dt_car_plates`: car plate detection dataset
- `dt_cat_dog`: cat/dog classification dataset
- `dt_cat_dog_2`: cat/dog classification dataset
- `dt_cat_dog_3`: local cat/dog train/test folder dataset

Potential future additions:

- `dt_racd`: Residential Activity Capture Dataset, useful later for event logic, activity understanding, and richer household behavior modeling
- Home Fire Dataset: future hazard-detection extension, not part of the current object-detection target
