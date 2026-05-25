"""
train_mlp.py — Train an MLP classifier on pre-extracted frozen embeddings.

Usage:
    python train_mlp.py --emb 512
    python train_mlp.py --emb 256
    python train_mlp.py --emb pca
"""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml
from tqdm import tqdm

from models import MLPClassifier


# helpers

def load_split(emb_dir: Path, split: str, tag: str):
    X = np.load(emb_dir / f"{split}_{tag}.npy").astype(np.float32)
    y = np.load(emb_dir / f"{split}_labels.npy").astype(np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)


def make_loader(X, y, batch_size, shuffle):
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X, y in tqdm(loader, desc="train" if train else "val  ", leave=False):
            X, y = X.to(device), y.to(device)
            out  = model(X)
            loss = criterion(out, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss    += loss.item() * X.size(0)
            total_correct += out.argmax(1).eq(y).sum().item()
            total_n       += X.size(0)

    return total_loss / total_n, total_correct / total_n * 100


# main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--emb", required=True,
                        help="Embedding type: 512, 256, or pca")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tag      = args.emb
    emb_dir  = Path(cfg["embeddings_dir"])
    mlp_cfg  = cfg["mlp_training"]
    n_cls    = cfg["dataset"]["num_classes"]
    epochs   = mlp_cfg["epochs"]
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mlp_{tag}] device={device}")

    # load embeddings
    X_train, y_train = load_split(emb_dir, "train", tag)
    X_val,   y_val   = load_split(emb_dir, "val",   tag)
    in_dim = X_train.shape[1]
    print(f"  Embedding dim: {in_dim}  |  Train: {len(y_train)}  Val: {len(y_val)}")

    batch_size  = mlp_cfg["batch_size"]
    train_loader = make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   batch_size, shuffle=False)

    # model
    model = MLPClassifier(
        in_dim=in_dim,
        num_classes=n_cls,
        hidden_dims=mlp_cfg["hidden_dims"],
        dropout=mlp_cfg["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=mlp_cfg["lr"],
        weight_decay=mlp_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # output paths
    ckpt_dir = Path(mlp_cfg["checkpoint_dir"]) / f"mlp_{tag}"
    log_dir  = Path(mlp_cfg["log_dir"]) / f"exp_mlp_{tag}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0

    with open(log_dir / "train_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_acc", "lr", "elapsed_s"])

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion,
                                        optimizer, device, train=True)
            _,       val_acc = run_epoch(model, val_loader, criterion,
                                         optimizer, device, train=False)
            scheduler.step()
            elapsed = time.time() - t0
            lr_now  = scheduler.get_last_lr()[0]

            print(f"[mlp_{tag}] Epoch {epoch:02d}/{epochs}  "
                  f"loss={tr_loss:.4f}  train_acc={tr_acc:.2f}%  "
                  f"val_acc={val_acc:.2f}%  ({elapsed:.0f}s)")
            writer.writerow([epoch, f"{tr_loss:.4f}", f"{tr_acc:.2f}",
                             f"{val_acc:.2f}", f"{lr_now:.2e}", f"{elapsed:.0f}"])

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {"epoch": epoch, "state_dict": model.state_dict(),
                     "val_acc": val_acc, "in_dim": in_dim, "tag": tag},
                    ckpt_dir / "best.pt",
                )
                print(f"  [best] val_acc={val_acc:.2f}%")

    print(f"\n[mlp_{tag}] Best val_acc={best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
