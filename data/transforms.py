from torchvision import transforms


def get_transforms(cfg, split: str):
    """
    Returns a torchvision transform pipeline for the given split.

    split: 'train' | 'val' | 'test'

    Train:  resize short edge → random crop 224 → random H-flip → normalize
    Val/Test: resize short edge → center crop 224 → normalize
    """
    mean = cfg["preprocessing"]["imagenet_mean"]
    std  = cfg["preprocessing"]["imagenet_std"]
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
