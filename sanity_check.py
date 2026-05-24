"""
sanity_check.py -- Verifies the dataset pipeline before full training.

Prints split sizes, one batch shape, and label distribution stats.
"""

import os
import yaml
import torch
from data import get_loaders
from data.dataset import CUBDataset

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

if os.name == "nt":
    cfg["training"]["num_workers"] = 0

root         = cfg["dataset"]["root"]
val_fraction = cfg["dataset"]["val_fraction"]
seed         = cfg["dataset"]["seed"]

for split in ["train", "val", "test"]:
    ds = CUBDataset(root, split, val_fraction, seed)
    counts = ds.class_counts()
    print(f"{split:5s}  images={len(ds):5d}  "
          f"classes={counts.index.nunique()}  "
          f"min_per_class={counts.min()}  max_per_class={counts.max()}")

print()
train_loader, val_loader, test_loader = get_loaders(cfg)
images, labels = next(iter(train_loader))
print(f"Batch shape : {images.shape}   dtype={images.dtype}")
print(f"Label range : {labels.min().item()} - {labels.max().item()}")
print(f"Pixel range : {images.min():.3f} - {images.max():.3f}  (normalized)")

unique = labels.unique()
print(f"Unique labels in batch: {len(unique)}")
print("\nSanity check passed.")
