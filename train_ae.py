"""
train_ae.py — Denoising Autoencoder training for CUB-200.

Trains encoder (ResNet-50 fine-tuned) + projection head + decoder to
reconstruct clean images from noisy/augmented inputs.

Usage:
    python train_ae.py --dim 512
    python train_ae.py --dim 256
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


# ── helpers ──────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, total_n = 0.0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        desc = "train" if train else "val  "
        for (noisy, clean), _ in tqdm(loader, desc=desc, leave=False):
            noisy, clean = noisy.to(device), clean.to(device)
            recon, _ = model(noisy)
            loss = criterion(recon, clean)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * noisy.size(0)
            total_n    += noisy.size(0)

    return total_loss / total_n


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dim",    type=int, required=True,
                        help="Embedding dimension (256 or 512)")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if os.name == "nt":
        cfg["ae_training"]["num_workers"] = 0

    emb_dim = args.dim
    tag     = f"ae_{emb_dim}"
    epochs  = 1 if args.smoke_test else cfg["ae_training"]["epochs"]
    if args.smoke_test:
        print(f"[smoke-test] dim={emb_dim}, epochs=1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{tag}] device={device}  emb_dim={emb_dim}  epochs={epochs}")

    # ── data ──
    train_loader, val_loader = get_ae_loaders(cfg)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── model ──
    freeze = cfg["model"]["freeze_backbone"]
    model  = AutoEncoder(emb_dim=emb_dim, pretrained=True, freeze_backbone=freeze).to(device)
    print(f"  freeze_backbone={freeze}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    criterion = nn.MSELoss()
    param_groups = get_ae_param_groups(
        model,
        lr_backbone=cfg["ae_training"]["lr_backbone"],
        lr_head=cfg["ae_training"]["lr_head"],
        weight_decay=cfg["ae_training"]["weight_decay"],
    )
    optimizer = torch.optim.Adam(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── output paths ──
    ckpt_dir  = Path(cfg["ae_training"]["checkpoint_dir"]) / tag
    log_dir   = Path("reports") / f"exp_{tag}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    with open(log_dir / "train_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr", "elapsed_s"])

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = run_epoch(model, train_loader, criterion,
                                   optimizer, device, train=True)
            val_loss   = run_epoch(model, val_loader,   criterion,
                                   optimizer, device, train=False)
            scheduler.step()
            elapsed = time.time() - t0
            lr_now  = scheduler.get_last_lr()[0]

            print(f"[{tag}] Epoch {epoch:02d}/{epochs}  "
                  f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  "
                  f"lr={lr_now:.2e}  ({elapsed:.0f}s)")
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                             f"{lr_now:.2e}", f"{elapsed:.0f}"])

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {"epoch": epoch, "state_dict": model.state_dict(),
                     "val_loss": val_loss, "emb_dim": emb_dim, "cfg": cfg},
                    ckpt_dir / "best.pt",
                )
                print(f"  [best] val_loss={val_loss:.5f}")

    print(f"\n[{tag}] Training complete. Best val_loss={best_val_loss:.5f}")
    print(f"  Checkpoint: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
