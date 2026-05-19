"""
evaluate_mlp.py — Evaluate a trained MLP on the test-set embeddings.

Usage:
    python evaluate_mlp.py --emb 512
    python evaluate_mlp.py --emb 256
    python evaluate_mlp.py --emb pca
    python evaluate_mlp.py --all        # runs all three + writes comparison
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
from sklearn.metrics import confusion_matrix as sk_cm
from torch.utils.data import DataLoader, TensorDataset

from models import MLPClassifier


# ── helpers ──────────────────────────────────────────────────────────────────

def load_emb(emb_dir: Path, split: str, tag: str):
    X = np.load(emb_dir / f"{split}_{tag}.npy").astype(np.float32)
    y = np.load(emb_dir / f"{split}_labels.npy").astype(np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)


def topk(outputs, labels, k):
    _, pred = outputs.topk(k, dim=1)
    return pred.eq(labels.view(-1, 1).expand_as(pred)).any(dim=1).float()


def evaluate_one(tag: str, cfg: dict, device) -> dict:
    emb_dir  = Path(cfg["embeddings_dir"])
    mlp_cfg  = cfg["mlp_training"]
    n_cls    = cfg["dataset"]["num_classes"]

    X_test, y_test = load_emb(emb_dir, "test", tag)
    in_dim = X_test.shape[1]

    ckpt_path = Path(mlp_cfg["checkpoint_dir"]) / f"mlp_{tag}" / "best.pt"
    assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = MLPClassifier(in_dim=in_dim, num_classes=n_cls,
                          hidden_dims=mlp_cfg["hidden_dims"],
                          dropout=mlp_cfg["dropout"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    loader = DataLoader(TensorDataset(X_test, y_test), batch_size=512, shuffle=False)

    all_top1, all_top5, all_labels, all_preds = [], [], [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out  = model(X)
            all_top1.extend(topk(out, y, 1).cpu().tolist())
            all_top5.extend(topk(out, y, 5).cpu().tolist())
            all_labels.extend(y.cpu().tolist())
            all_preds.extend(out.argmax(1).cpu().tolist())

    top1 = np.mean(all_top1) * 100
    top5 = np.mean(all_top5) * 100
    print(f"[mlp_{tag}]  Top-1: {top1:.2f}%   Top-5: {top5:.2f}%")

    # ── per-class accuracy ──
    n_cls = cfg["dataset"]["num_classes"]
    class_correct = np.zeros(n_cls)
    class_total   = np.zeros(n_cls)
    for lbl, pred in zip(all_labels, all_preds):
        class_total[lbl]   += 1
        class_correct[lbl] += int(lbl == pred)

    classes_df  = pd.read_csv(
        Path(cfg["dataset"]["root"]) / "classes.txt",
        sep=" ", header=None, names=["class_id", "class_name"])
    class_names = classes_df["class_name"].tolist()

    per_class = pd.DataFrame({
        "class_name": class_names,
        "correct":    class_correct.astype(int),
        "total":      class_total.astype(int),
        "accuracy":   class_correct / np.maximum(class_total, 1) * 100,
    }).sort_values("accuracy", ascending=False)

    results_dir = Path("reports") / f"exp_mlp_{tag}"
    results_dir.mkdir(parents=True, exist_ok=True)
    per_class.to_csv(results_dir / "per_class_accuracy.csv", index=False)

    # ── confusion matrix (25 most-confused) ──
    cm = sk_cm(all_labels, all_preds, labels=list(range(n_cls)))
    off = cm.copy(); np.fill_diagonal(off, 0)
    idx25 = np.argsort(off.sum(axis=1))[-25:][::-1]
    sub_names = [class_names[i].split(".")[-1][:18] for i in idx25]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm[np.ix_(idx25, idx25)], aspect="auto", cmap="Blues")
    ax.set_xticks(range(25)); ax.set_xticklabels(sub_names, rotation=90, fontsize=7)
    ax.set_yticks(range(25)); ax.set_yticklabels(sub_names, fontsize=7)
    ax.set_title(f"Confusion Matrix (25 most-confused) — mlp_{tag}", fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.7); plt.tight_layout()
    fig.savefig(results_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    # ── summary ──
    with open(results_dir / "eval_summary.txt", "w") as f:
        f.write(f"Embedding: {tag}\n")
        f.write(f"Embedding dim: {in_dim}\n")
        f.write(f"Top-1 Accuracy: {top1:.2f}%\n")
        f.write(f"Top-5 Accuracy: {top5:.2f}%\n")
        f.write(f"Checkpoint epoch: {ckpt['epoch']}\n")

    return {"tag": tag, "in_dim": in_dim, "top1": top1, "top5": top5,
            "epoch": ckpt["epoch"]}


def write_comparison(results: list[dict], cfg: dict):
    """Write reports/ae_comparison.md with aug vs noaug comparison."""
    emb_dir = Path(cfg["embeddings_dir"])
    pca_note = ""
    pca_info_path = emb_dir / "pca_info.txt"
    if pca_info_path.exists():
        pca_note = pca_info_path.read_text()

    by_tag = {r["tag"]: r for r in results}

    lines = [
        "# Encoder-Decoder Embedding Comparison — CUB-200",
        "",
        "Joint loss: alpha*MSE + (1-alpha)*CrossEntropy  |  "
        "MLP classifier trained on frozen embeddings for 30 epochs.",
        "",
        "## Results",
        "",
        "| Embedding      | Augmented | Dim | Top-1 (%) | Top-5 (%) | Best Epoch |",
        "|----------------|-----------|-----|-----------|-----------|------------|",
    ]
    order = ["512", "512_noaug", "256", "256_noaug", "pca"]
    for tag in order:
        if tag not in by_tag:
            continue
        r   = by_tag[tag]
        aug = "No" if tag.endswith("_noaug") else ("—" if tag == "pca" else "Yes")
        lines.append(f"| {tag:<14} | {aug:<9} | {r['in_dim']:>3} | "
                     f"{r['top1']:>9.2f} | {r['top5']:>9.2f} | {r['epoch']:>10} |")

    # Aug vs noaug delta table
    lines += ["", "## Augmentation Effect (aug − noaug)","",
              "| Dim | Delta Top-1 | Delta Top-5 |",
              "|-----|-------------|-------------|"]
    for dim in ["512", "256"]:
        aug_r   = by_tag.get(dim)
        noaug_r = by_tag.get(f"{dim}_noaug")
        if aug_r and noaug_r:
            lines.append(f"| {dim} | {aug_r['top1']-noaug_r['top1']:+.2f}        "
                         f"| {aug_r['top5']-noaug_r['top5']:+.2f}        |")

    if pca_note:
        lines += ["", "## PCA Details", "", "```", pca_note.strip(), "```"]

    lines += [
        "", "## Per-experiment artifacts",
        "",
        "Each variant writes to `reports/exp_mlp_<tag>/`:",
        "- `train_log.csv`, `per_class_accuracy.csv`, `confusion_matrix.png`, `eval_summary.txt`",
    ]

    out = Path("reports/ae_comparison.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nComparison report: {out}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--emb",  default=None, help="512, 256, or pca")
    parser.add_argument("--all",  action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_tags = ["512", "256", "pca", "512_noaug", "256_noaug"]
    tags     = all_tags if args.all else [args.emb]

    results = [evaluate_one(tag, cfg, device) for tag in tags]
    if args.all:
        write_comparison(results, cfg)


if __name__ == "__main__":
    main()
