from torch.utils.data import DataLoader
from .dataset import CUBDataset
from .transforms import get_transforms, get_ae_transforms


def get_loaders(cfg):
    """Standard classification loaders returning (image, label)."""
    root         = cfg["dataset"]["root"]
    val_fraction = cfg["dataset"]["val_fraction"]
    seed         = cfg["dataset"]["seed"]
    batch_size   = cfg["training"]["batch_size"]
    num_workers  = cfg["training"]["num_workers"]

    train_ds = CUBDataset(root, "train", val_fraction, seed,
                          get_transforms(cfg, "train"))
    val_ds   = CUBDataset(root, "val",   val_fraction, seed,
                          get_transforms(cfg, "val"))
    test_ds  = CUBDataset(root, "test",  val_fraction, seed,
                          get_transforms(cfg, "test"))

    kw = dict(num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw),
    )


def get_ae_loaders(cfg):
    """
    AE loaders returning ((noisy, clean), label) per sample.
    DataLoader's default collate produces ((B_noisy, B_clean), B_labels).
    """
    root         = cfg["dataset"]["root"]
    val_fraction = cfg["dataset"]["val_fraction"]
    seed         = cfg["dataset"]["seed"]
    batch_size   = cfg["ae_training"]["batch_size"]
    num_workers  = cfg["ae_training"]["num_workers"]

    train_ds = CUBDataset(root, "train", val_fraction, seed,
                          get_ae_transforms(cfg, "train"))
    val_ds   = CUBDataset(root, "val",   val_fraction, seed,
                          get_ae_transforms(cfg, "val"))

    kw = dict(num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw),
    )


def get_embed_loaders(cfg):
    """
    Clean-image loaders for embedding extraction (all three splits).
    Uses val/test transforms (center crop, no noise) even for train.
    """
    root         = cfg["dataset"]["root"]
    val_fraction = cfg["dataset"]["val_fraction"]
    seed         = cfg["dataset"]["seed"]
    batch_size   = cfg["ae_training"]["batch_size"]
    num_workers  = cfg["ae_training"]["num_workers"]

    # Use 'val' transforms for all splits (deterministic, no augmentation)
    train_ds = CUBDataset(root, "train", val_fraction, seed,
                          get_transforms(cfg, "val"))
    val_ds   = CUBDataset(root, "val",   val_fraction, seed,
                          get_transforms(cfg, "val"))
    test_ds  = CUBDataset(root, "test",  val_fraction, seed,
                          get_transforms(cfg, "test"))

    kw = dict(num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=False, **kw),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw),
    )
