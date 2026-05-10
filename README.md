# CUB-200-2011 Transfer Learning Baseline

Sprint 0 — linear-probe baseline using ResNet-50 pretrained on ImageNet.
Project for the Signal, Image and Video Processing course at UAB.

## Project Layout

```
birds/
├── configs/
│   └── config.yaml          # all hyperparameters and paths
├── data/
│   ├── __init__.py
│   ├── dataset.py           # CUBDataset (reads official splits)
│   ├── transforms.py        # train / val / test augmentation pipelines
│   └── loaders.py           # DataLoader factory
├── models/
│   ├── __init__.py
│   └── backbone.py          # ResNet-50 head replacement + freeze helper
├── reports/                 # generated artifacts land here
├── checkpoints/             # best.pt saved here during training
├── train.py                 # training loop
├── evaluate.py              # test-set evaluation + plots
├── sanity_check.py          # quick dataset / loader verification
└── requirements.txt
```

## Setup

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Dataset

The CUB-200-2011 dataset should be placed (or already exists) at:

```
CUB_200_2011/CUB_200_2011/
```

Expected structure:
```
CUB_200_2011/CUB_200_2011/
├── images/
├── images.txt
├── image_class_labels.txt
├── train_test_split.txt
└── classes.txt
```

If not present, download from https://www.vision.caltech.edu/datasets/cub_200_2011/ and extract so
the above structure is satisfied. Total images: **11,788** across **200 classes**.

## Running

### 1. Sanity check (dataset loader)
```bash
python sanity_check.py
```
Expected output: split sizes, batch shape `[32, 3, 224, 224]`, and "Sanity check passed."

### 2. Smoke test (1 epoch — confirms pipeline runs end-to-end)
```bash
python train.py --smoke-test
```

### 3. Full training (10 epochs)
```bash
python train.py
```
Logs are written to `reports/train_log.csv`. Best checkpoint saved to `checkpoints/best.pt`.

### 4. Evaluation
```bash
python evaluate.py
```
Produces in `reports/`:
- `eval_summary.txt`       — top-1 and top-5 test accuracy
- `per_class_accuracy.csv` — per-class breakdown sorted by accuracy
- `confusion_matrix.png`   — 25 most-confused classes
- `sample_predictions.png` — 8 correct + 8 incorrect predictions

## Configuration

All settings live in `configs/config.yaml`. Key knobs:

| Key | Default | Notes |
|-----|---------|-------|
| `model.freeze_backbone` | `true` | Set `false` to fine-tune all layers (Week 1) |
| `training.epochs` | `10` | Reduced to 5 automatically on CPU |
| `training.batch_size` | `32` | Reduce if OOM |
| `training.lr` | `1e-3` | Adam learning rate for the head |
| `dataset.val_fraction` | `0.10` | Fraction of train split used for validation |

## Defaults and Design Decisions

- **Validation split**: 10 % of the official training images, stratified by class using
  `sklearn.train_test_split`. The official test split is never touched until `evaluate.py`.
- **No data leakage**: train/val/test splits are derived from the same `CUBDataset` with a fixed
  seed, so all three are consistent.
- **Windows compatibility**: `num_workers` is set to 0 on Windows (no POSIX fork).
- **Checkpoint**: only the best-validation-accuracy epoch is saved.
