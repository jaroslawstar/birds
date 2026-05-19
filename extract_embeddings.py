"""
extract_embeddings.py — Extract and save embeddings for all three methods.

Produces (under embeddings/):
  {split}_labels.npy          labels for train / val / test (shared)
  {split}_512.npy             512-d learned embeddings from AE_512
  {split}_256.npy             256-d learned embeddings from AE_256
  {split}_pca.npy             PCA-reduced 2048-d raw features (90% variance)
  pca_model.pkl               fitted sklearn PCA object
  pca_info.txt                n_components, explained variance ratio

Usage:
    python extract_embeddings.py
"""

import os
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from data import get_embed_loaders
from models import AutoEncoder


# ── helpers ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract(model: AutoEncoder, loader, device, mode: str) -> np.ndarray:
    """
    mode: 'projected'  → emb_dim-d learned embedding
          'raw'        → 2048-d backbone features (for PCA)
    """
    model.eval()
    vecs, labs = [], []
    for images, labels in tqdm(loader, desc=f"extract ({mode})", leave=False):
        images = images.to(device)
        if mode == "projected":
            v = model.encode(images)
        else:
            v = model.get_raw_features(images)
        vecs.append(v.cpu().numpy())
        labs.append(labels.numpy())
    return np.concatenate(vecs), np.concatenate(labs)


def load_ae(cfg, dim: int, device, suffix: str = ""):
    tag       = f"ae_{dim}{suffix}"
    ckpt_path = Path(cfg["ae_training"]["checkpoint_dir"]) / tag / "best.pt"
    assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}. Run train_ae.py first."
    ckpt        = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_classes = ckpt.get("num_classes", None)
    model = AutoEncoder(emb_dim=dim, pretrained=False,
                        num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    val_acc_str = f"  val_acc={ckpt['val_acc']:.2f}%" if "val_acc" in ckpt else ""
    print(f"Loaded {tag} checkpoint (epoch {ckpt['epoch']}, "
          f"val_loss={ckpt['val_loss']:.5f}{val_acc_str})")
    return model


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    if os.name == "nt":
        cfg["ae_training"]["num_workers"] = 0

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_dir  = Path(cfg["embeddings_dir"])
    emb_dir.mkdir(exist_ok=True)

    splits = ["train", "val", "test"]
    train_loader, val_loader, test_loader = get_embed_loaders(cfg)
    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}

    # ── aug and noaug learned embeddings for each dim ──
    for dim in cfg["ae_training"]["emb_dims"]:
        for suffix in ["", "_noaug"]:
            ckpt_tag  = f"ae_{dim}{suffix}"
            emb_tag   = f"{dim}{suffix}"
            ckpt_path = Path(cfg["ae_training"]["checkpoint_dir"]) / ckpt_tag / "best.pt"
            if not ckpt_path.exists():
                print(f"  Skipping {ckpt_tag} (no checkpoint found)")
                continue
            print(f"\n=== Extracting {emb_tag} embeddings ===")
            model = load_ae(cfg, dim, device, suffix=suffix)
            for split, loader in loaders.items():
                vecs, labs = extract(model, loader, device, mode="projected")
                np.save(emb_dir / f"{split}_{emb_tag}.npy", vecs)
                lab_path = emb_dir / f"{split}_labels.npy"
                if not lab_path.exists():
                    np.save(lab_path, labs)
            print(f"  Saved {emb_tag} embeddings to {emb_dir}/")

    # ── 2048-d raw features for PCA (frozen backbone; aug/noaug gives identical features) ──
    src_dim = cfg["pca"]["source_dim"]
    print(f"\n=== Extracting 2048-d raw features for PCA (from ae_{src_dim}) ===")
    model = load_ae(cfg, src_dim, device, suffix="")
    raw = {}
    for split, loader in loaders.items():
        vecs, _ = extract(model, loader, device, mode="raw")
        raw[split] = vecs
    print(f"  Raw feature shape: {raw['train'].shape}")

    # ── Fit PCA on training set ──
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    print(f"\n=== Fitting PCA (target variance={cfg['pca']['variance_threshold']}) ===")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(raw["train"])

    pca = PCA(n_components=cfg["pca"]["variance_threshold"], svd_solver="full")
    pca.fit(X_train_scaled)
    n_components = pca.n_components_
    explained    = pca.explained_variance_ratio_.sum()
    print(f"  PCA: {n_components} components explain {explained*100:.2f}% variance")

    # Save PCA model + scaler together
    pca_bundle = {"pca": pca, "scaler": scaler}
    with open(emb_dir / "pca_model.pkl", "wb") as f:
        pickle.dump(pca_bundle, f)

    with open(emb_dir / "pca_info.txt", "w") as f:
        f.write(f"n_components: {n_components}\n")
        f.write(f"explained_variance: {explained*100:.4f}%\n")
        f.write(f"source_encoder: ae_{src_dim}\n")
        f.write(f"variance_threshold: {cfg['pca']['variance_threshold']}\n")

    # Transform all splits
    for split in splits:
        X_scaled = scaler.transform(raw[split])
        X_pca    = pca.transform(X_scaled)
        np.save(emb_dir / f"{split}_pca.npy", X_pca)
    print(f"  Saved PCA embeddings ({n_components}-d) to {emb_dir}/")

    print("\nAll embeddings saved:")
    for p in sorted(emb_dir.iterdir()):
        if p.suffix in (".npy", ".pkl", ".txt"):
            size = p.stat().st_size / 1024
            print(f"  {p.name:<30} {size:8.1f} KB")


if __name__ == "__main__":
    main()
