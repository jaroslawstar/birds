import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF
from torchvision import transforms


# ── Standard classification transforms ───────────────────────────────────────

def get_transforms(cfg, split: str):
    """Standard single-image transforms for classification."""
    mean   = cfg["preprocessing"]["imagenet_mean"]
    std    = cfg["preprocessing"]["imagenet_std"]
    resize = cfg["preprocessing"]["resize_short_edge"]
    crop   = cfg["preprocessing"]["crop_size"]
    normalize = transforms.Normalize(mean=mean, std=std)

    if split == "train":
        return transforms.Compose([
            transforms.Resize(resize),
            transforms.RandomCrop(crop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            normalize,
        ])


# ── Denoising Autoencoder paired transforms ───────────────────────────────────

class AETransformPair:
    """
    Produces (noisy_input, clean_target) pairs for denoising AE training.

    Pipeline:
      1. Geometric transforms (resize, crop, flip, rotation) applied ONCE —
         the same result becomes the clean target.
      2. ColorJitter + Gaussian noise applied ONLY to the input copy.

    For val/test (split != 'train'): deterministic center crop, same noise.
    """

    def __init__(self, cfg, split: str):
        ae_aug  = cfg["ae_augmentation"]
        pre     = cfg["preprocessing"]
        self.mean      = pre["imagenet_mean"]
        self.std       = pre["imagenet_std"]
        self.resize    = pre["resize_short_edge"]
        self.crop_size = pre["crop_size"]
        self.noise_std = cfg["ae_training"]["noise_std"]
        self.split     = split

        self.jitter = transforms.ColorJitter(
            brightness=ae_aug["jitter_brightness"],
            contrast=ae_aug["jitter_contrast"],
            saturation=ae_aug["jitter_saturation"],
            hue=ae_aug["jitter_hue"],
        )
        self.rot_deg = ae_aug["rotation_degrees"]
        self._normalize = transforms.Normalize(mean=self.mean, std=self.std)

    def _to_tensor_normalized(self, img: Image.Image) -> torch.Tensor:
        return self._normalize(TF.to_tensor(img))

    def __call__(self, img: Image.Image):
        # ── Step 1: shared geometric transforms ──
        img = TF.resize(img, self.resize)

        if self.split == "train":
            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                img, (self.crop_size, self.crop_size))
            img = TF.crop(img, i, j, h, w)
            # Random horizontal flip
            if torch.rand(1).item() < 0.5:
                img = TF.hflip(img)
            # Random rotation
            angle = (torch.rand(1).item() * 2 - 1) * self.rot_deg
            img = TF.rotate(img, angle)
        else:
            img = TF.center_crop(img, self.crop_size)

        # ── Step 2: clean target (no corruption) ──
        clean = self._to_tensor_normalized(img)

        # ── Step 3: noisy input (ColorJitter + Gaussian noise) ──
        noisy_pil = self.jitter(img) if self.split == "train" else img
        noisy = self._to_tensor_normalized(noisy_pil)
        noisy = noisy + torch.randn_like(noisy) * self.noise_std

        return noisy, clean


def get_ae_transforms(cfg, split: str) -> AETransformPair:
    return AETransformPair(cfg, split)
