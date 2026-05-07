import os
from pathlib import Path
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset


class CUBDataset(Dataset):
    """
    CUB-200-2011 dataset loader.

    Reads the official annotation files:
      images.txt            — image_id → relative path
      image_class_labels.txt — image_id → class_id (1-indexed)
      train_test_split.txt  — image_id → is_train (1=train, 0=test)

    split: 'train' | 'val' | 'test'

    For 'train' and 'val', the is_train=1 images are split by
    stratified sampling according to val_fraction.
    """

    def __init__(self, root: str, split: str, val_fraction: float = 0.10,
                 seed: int = 42, transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        # ---- load annotation tables ----
        images_df = pd.read_csv(
            self.root / "images.txt",
            sep=" ", header=None, names=["img_id", "filepath"]
        )
        labels_df = pd.read_csv(
            self.root / "image_class_labels.txt",
            sep=" ", header=None, names=["img_id", "class_id"]
        )
        split_df = pd.read_csv(
            self.root / "train_test_split.txt",
            sep=" ", header=None, names=["img_id", "is_train"]
        )

        df = images_df.merge(labels_df, on="img_id").merge(split_df, on="img_id")
        # Convert to 0-indexed labels
        df["label"] = df["class_id"] - 1

        # ---- carve val from train using stratified sampling ----
        official_train = df[df["is_train"] == 1].copy()
        official_test  = df[df["is_train"] == 0].copy()

        from sklearn.model_selection import train_test_split
        train_idx, val_idx = train_test_split(
            official_train.index,
            test_size=val_fraction,
            stratify=official_train["label"],
            random_state=seed,
        )
        train_df = official_train.loc[train_idx].reset_index(drop=True)
        val_df   = official_train.loc[val_idx].reset_index(drop=True)
        test_df  = official_test.reset_index(drop=True)

        if split == "train":
            self.df = train_df
        elif split == "val":
            self.df = val_df
        elif split == "test":
            self.df = test_df
        else:
            raise ValueError(f"Unknown split '{split}'. Choose train/val/test.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root / "images" / row["filepath"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["label"])

    def class_counts(self):
        """Returns a Series mapping label → count, for sanity checks."""
        return self.df["label"].value_counts().sort_index()
