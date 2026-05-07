"""
train.py — Sprint 0 training loop for CUB-200 linear-probe baseline.

Usage:
    python train.py                         # uses configs/config.yaml
    python train.py --config configs/config.yaml --epochs 5
"""

import argparse
import csv
import os
import time
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from data import get_loaders
from models import build_model
from models.backbone import get_trainable_params


# ── helpers ──────────────────────────────────────────────────────────────────

def accuracy(outputs, labels, topk=(1,)):
    """Returns top-k accuracy values as percentages."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)
        _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))
        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum()
            results.append((correct_k / batch_size * 100).item())
        return results


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in tqdm(loader, desc="train" if train else "val  ", leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += preds.eq(labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    avg_acc  = total_correct / total_samples * 100
    return avg_loss, avg_acc


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run only 1 epoch for pipeline verification")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.smoke_test:
        cfg["training"]["epochs"] = 1
        print("[smoke-test] Overriding epochs=1")
    elif args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    # ── device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu" and cfg["training"]["epochs"] == 10:
        cfg["training"]["epochs"] = 5
        print("  No CUDA — reducing default epochs to 5 for CPU run.")

    # ── data ──
    # Reduce num_workers on Windows to avoid multiprocessing issues
    if os.name == "nt":
        cfg["training"]["num_workers"] = 0

    train_loader, val_loader, _ = get_loaders(cfg)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── model ──
    model = build_model(cfg).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        get_trainable_params(model),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    epochs = cfg["training"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── logging ──
    log_path = Path(cfg["training"]["log_csv"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    best_epoch   = 0

    with open(log_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_acc", "lr", "elapsed_s"])

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, train_acc = run_epoch(
                model, train_loader, criterion, optimizer, device, train=True
            )
            _, val_acc = run_epoch(
                model, val_loader, criterion, optimizer, device, train=False
            )
            scheduler.step()
            elapsed = time.time() - t0
            lr_now = scheduler.get_last_lr()[0]

            print(
                f"Epoch {epoch:02d}/{epochs}  "
                f"loss={train_loss:.4f}  train_acc={train_acc:.2f}%  "
                f"val_acc={val_acc:.2f}%  lr={lr_now:.2e}  "
                f"({elapsed:.0f}s)"
            )
            writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.2f}",
                             f"{val_acc:.2f}", f"{lr_now:.2e}", f"{elapsed:.0f}"])

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch   = epoch
                torch.save(
                    {"epoch": epoch, "state_dict": model.state_dict(),
                     "val_acc": val_acc, "cfg": cfg},
                    ckpt_dir / "best.pt"
                )
                print(f"  [best] Saved checkpoint (val_acc={val_acc:.2f}%)")

    print(f"\nTraining complete. Best val_acc={best_val_acc:.2f}% at epoch {best_epoch}.")
    print(f"Checkpoint: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
