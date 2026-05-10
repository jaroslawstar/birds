"""
evaluate.py — Loads best checkpoint, evaluates on the official CUB-200 test set.

Usage:
    python evaluate.py
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
import yaml
from torchvision.utils import make_grid
from tqdm import tqdm

from data import get_loaders
from data.dataset import CUBDataset
from data.transforms import get_transforms
from models import build_model


# ── helpers ──────────────────────────────────────────────────────────────────

def topk_accuracy(outputs, labels, k):
    _, pred = outputs.topk(k, dim=1, largest=True, sorted=True)
    correct = pred.eq(labels.view(-1, 1).expand_as(pred))
    return correct.any(dim=1).float()


def denormalize(tensor, mean, std):
    """Invert ImageNet normalization for visualization."""
    t = tensor.clone()
    for c, (m, s) in enumerate(zip(mean, std)):
        t[c] = t[c] * s + m
    return t.clamp(0, 1)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if os.name == "nt":
        cfg["training"]["num_workers"] = 0

    results_dir = Path("reports/exp_baseline")
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path   = Path("checkpoints/best.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.2f}%)")

    model = build_model(cfg, technique).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # ── test loader ──
    _, _, test_loader = get_loaders(cfg)

    # ── forward pass ──
    all_top1 = []
    all_top5 = []
    all_labels = []
    all_preds  = []
    all_probs  = []

    # Store raw images + metadata for sample grid
    sample_images = []
    sample_meta   = []   # (label, pred, correct)

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="eval"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)

            top1_correct = topk_accuracy(outputs, labels, 1)
            top5_correct = topk_accuracy(outputs, labels, 5)
            preds = outputs.argmax(dim=1)

            all_top1.extend(top1_correct.cpu().tolist())
            all_top5.extend(top5_correct.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.append(probs.cpu())

            # Collect samples for the visual grid
            for i in range(images.size(0)):
                if len(sample_images) < 200:   # collect first 200, subsample later
                    sample_images.append(images[i].cpu())
                    sample_meta.append((labels[i].item(), preds[i].item(),
                                        top1_correct[i].item()))

    top1_acc = np.mean(all_top1) * 100
    top5_acc = np.mean(all_top5) * 100
    print(f"\nTest Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Test Top-5 Accuracy: {top5_acc:.2f}%")

    # ── per-class accuracy ──
    num_classes = cfg["dataset"]["num_classes"]
    class_correct = np.zeros(num_classes)
    class_total   = np.zeros(num_classes)
    for label, pred in zip(all_labels, all_preds):
        class_total[label]   += 1
        class_correct[label] += int(label == pred)

    # Load class names
    classes_df = pd.read_csv(
        Path(cfg["dataset"]["root"]) / "classes.txt",
        sep=" ", header=None, names=["class_id", "class_name"]
    )
    class_names = classes_df["class_name"].tolist()

    per_class_acc = pd.DataFrame({
        "class_id":   range(num_classes),
        "class_name": class_names,
        "correct":    class_correct.astype(int),
        "total":      class_total.astype(int),
        "accuracy":   class_correct / np.maximum(class_total, 1) * 100,
    }).sort_values("accuracy", ascending=False)

    per_class_csv = results_dir / "per_class_accuracy.csv"
    per_class_acc.to_csv(per_class_csv, index=False)
    print(f"Per-class accuracy saved to {per_class_csv}")

    # ── confusion matrix (top-25 most confused classes) ──
    # Full 200×200 matrix would be too dense; show top-25 by error rate
    from sklearn.metrics import confusion_matrix as sk_cm

    cm = sk_cm(all_labels, all_preds, labels=list(range(num_classes)))

    # Pick 25 classes with highest off-diagonal mass
    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)
    top25_idx = np.argsort(off_diag.sum(axis=1))[-25:][::-1]

    cm_sub = cm[np.ix_(top25_idx, top25_idx)]
    sub_names = [class_names[i].split(".")[-1][:20] for i in top25_idx]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_sub, aspect="auto", cmap="Blues")
    ax.set_xticks(range(25)); ax.set_xticklabels(sub_names, rotation=90, fontsize=7)
    ax.set_yticks(range(25)); ax.set_yticklabels(sub_names, fontsize=7)
    ax.set_title("Confusion Matrix — 25 most-confused classes", fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    cm_path = results_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)
    print(f"Confusion matrix saved to {cm_path}")

    # ── sample prediction grid (8 correct + 8 incorrect) ──
    mean = cfg["preprocessing"]["imagenet_mean"]
    std  = cfg["preprocessing"]["imagenet_std"]

    correct_samples   = [(img, m) for img, m in zip(sample_images, sample_meta) if m[2] == 1.0]
    incorrect_samples = [(img, m) for img, m in zip(sample_images, sample_meta) if m[2] == 0.0]

    n_correct   = min(8, len(correct_samples))
    n_incorrect = min(8, len(incorrect_samples))
    chosen = correct_samples[:n_correct] + incorrect_samples[:n_incorrect]

    fig = plt.figure(figsize=(16, 4.5))
    gs  = gridspec.GridSpec(2, 8, hspace=0.05, wspace=0.05)

    for col in range(8):
        for row, group_offset in enumerate([0, n_correct]):
            idx = group_offset + col
            if idx >= len(chosen):
                continue
            img_t, (lbl, pred, correct_flag) = chosen[idx]
            img_np = denormalize(img_t, mean, std).permute(1, 2, 0).numpy()

            ax = fig.add_subplot(gs[row, col])
            ax.imshow(img_np)
            ax.axis("off")
            short_true = class_names[lbl].split(".")[-1][:18]
            short_pred = class_names[pred].split(".")[-1][:18]
            color = "green" if correct_flag else "red"
            ax.set_title(f"T:{short_true}\nP:{short_pred}", fontsize=5, color=color)

    fig.suptitle("Top row: correct  |  Bottom row: incorrect", fontsize=10, y=1.01)
    grid_path = results_dir / "sample_predictions.png"
    fig.savefig(grid_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Sample predictions saved to {grid_path}")

    # ── summary ──
    summary_path = results_dir / "eval_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Technique: baseline\n")
        f.write(f"Top-1 Accuracy: {top1_acc:.2f}%\n")
        f.write(f"Top-5 Accuracy: {top5_acc:.2f}%\n")
        f.write(f"Checkpoint epoch: {ckpt['epoch']}\n")
    print(f"\nSummary written to {summary_path}")

    return top1_acc, top5_acc


if __name__ == "__main__":
    main()
