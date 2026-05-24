"""
train_ae.py -- Autoencoder training for CUB-200.

Joint loss: alpha * MSE(reconstruction) + (1 - alpha) * CrossEntropy(classification)

Usage:
    python train_ae.py --dim 512
    python train_ae.py --dim 256
    python train_ae.py --dim 512 --no-aug       # disable ColorJitter + Gaussian noise
    python train_ae.py --dim 512 --smoke-test   # 1 epoch sanity check
"""

import argparse
import csv
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from data import get_ae_loaders
from models import AutoEncoder, get_ae_param_groups


def run_epoch(model, loader, mse_crit, ce_crit, alpha, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, total_mse, total_ce, total_correct, total_n = 0.0, 0.0, 0.0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        desc = "train" if train else "val  "
        for (noisy, clean), labels in tqdm(loader, desc=desc, leave=False):
            noisy  = noisy.to(device)
            clean  = clean.to(device)
            labels = labels.to(device)

            recon, _, logits = model(noisy)
            mse  = mse_crit(recon, clean)
            ce   = ce_crit(logits, labels)
            loss = alpha * mse + (1.0 - alpha) * ce

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            n = noisy.size(0)
            total_loss    += loss.item() * n
            total_mse     += mse.item()  * n
            total_ce      += ce.item()   * n
            total_correct += logits.argmax(1).eq(labels).sum().item()
            total_n       += n

    return (total_loss / total_n,
            total_mse  / total_n,
            total_ce   / total_n,
            total_correct / total_n * 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dim",    type=int, required=True,
                        help="Embedding dimension (256 or 512)")
    parser.add_argument("--no-aug", action="store_true",
                        help="Disable ColorJitter and Gaussian noise")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if os.name == "nt":
        cfg["ae_training"]["num_workers"] = 0

    emb_dim     = args.dim
    augment     = not args.no_aug
    tag         = f"ae_{emb_dim}" if augment else f"ae_{emb_dim}_noaug"
    epochs      = 1 if args.smoke_test else cfg["ae_training"]["epochs"]
    alpha       = cfg["ae_training"]["alpha"]
    num_classes = cfg["dataset"]["num_classes"]

    if args.smoke_test:
        print(f"[smoke-test] dim={emb_dim}, epochs=1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{tag}] device={device}  emb_dim={emb_dim}  epochs={epochs}  "
          f"alpha={alpha}  augment={augment}")

    train_loader, val_loader = get_ae_loaders(cfg, augment=augment)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    freeze = cfg["model"]["freeze_backbone"]
    model  = AutoEncoder(emb_dim=emb_dim, pretrained=True,
                         freeze_backbone=freeze,
                         num_classes=num_classes).to(device)
    print(f"  freeze_backbone={freeze}  num_classes={num_classes}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    mse_crit = nn.MSELoss()
    ce_crit  = nn.CrossEntropyLoss()
    param_groups = get_ae_param_groups(
        model,
        lr_backbone=cfg["ae_training"]["lr_backbone"],
        lr_head=cfg["ae_training"]["lr_head"],
        weight_decay=cfg["ae_training"]["weight_decay"],
    )
    optimizer = torch.optim.Adam(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt_dir = Path(cfg["ae_training"]["checkpoint_dir"]) / tag
    log_dir  = Path("reports") / f"exp_{tag}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    with open(log_dir / "train_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_mse", "train_ce", "train_acc",
                         "val_loss",   "val_mse",   "val_ce",   "val_acc",
                         "lr", "elapsed_s"])

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            tr_loss, tr_mse, tr_ce, tr_acc = run_epoch(
                model, train_loader, mse_crit, ce_crit, alpha, optimizer, device, train=True)
            vl_loss, vl_mse, vl_ce, vl_acc = run_epoch(
                model, val_loader,   mse_crit, ce_crit, alpha, optimizer, device, train=False)
            scheduler.step()
            elapsed = time.time() - t0
            lr_now  = scheduler.get_last_lr()[0]

            print(f"[{tag}] Epoch {epoch:02d}/{epochs}  "
                  f"loss={vl_loss:.4f}  mse={vl_mse:.4f}  ce={vl_ce:.4f}  "
                  f"val_acc={vl_acc:.2f}%  lr={lr_now:.2e}  ({elapsed:.0f}s)")
            writer.writerow([epoch,
                             f"{tr_loss:.6f}", f"{tr_mse:.6f}", f"{tr_ce:.6f}", f"{tr_acc:.2f}",
                             f"{vl_loss:.6f}", f"{vl_mse:.6f}", f"{vl_ce:.6f}", f"{vl_acc:.2f}",
                             f"{lr_now:.2e}", f"{elapsed:.0f}"])

            if vl_loss < best_val_loss:
                best_val_loss = vl_loss
                torch.save(
                    {"epoch": epoch, "state_dict": model.state_dict(),
                     "val_loss": vl_loss, "val_acc": vl_acc,
                     "emb_dim": emb_dim, "num_classes": num_classes, "cfg": cfg},
                    ckpt_dir / "best.pt",
                )
                print(f"  [best] val_loss={vl_loss:.4f}  val_acc={vl_acc:.2f}%")

    print(f"\n[{tag}] Training complete. Best val_loss={best_val_loss:.4f}")
    print(f"  Checkpoint: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
