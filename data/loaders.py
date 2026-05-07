from torch.utils.data import DataLoader
from .dataset import CUBDataset
from .transforms import get_transforms


def get_loaders(cfg):
    """Returns (train_loader, val_loader, test_loader)."""
    root         = cfg["dataset"]["root"]
    val_fraction = cfg["dataset"]["val_fraction"]
    seed         = cfg["dataset"]["seed"]
    batch_size   = cfg["training"]["batch_size"]
    num_workers  = cfg["training"]["num_workers"]

    train_ds = CUBDataset(root, "train", val_fraction, seed, get_transforms(cfg, "train"))
    val_ds   = CUBDataset(root, "val",   val_fraction, seed, get_transforms(cfg, "val"))
    test_ds  = CUBDataset(root, "test",  val_fraction, seed, get_transforms(cfg, "test"))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
