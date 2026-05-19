import torch
import torch.nn as nn
from torchvision import models


class Encoder(nn.Module):
    """
    ResNet-50 backbone (fine-tuned) + 2-layer projection head → emb_dim.

    get_raw_features() returns the 2048-d avgpool output before projection;
    used for PCA embedding extraction.
    """

    def __init__(self, emb_dim: int, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)
        # Drop the final FC; keep conv layers + avgpool → (B, 2048, 1, 1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.projection = nn.Sequential(
            nn.Linear(2048, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim),
        )
        self.emb_dim = emb_dim

    def get_raw_features(self, x: torch.Tensor) -> torch.Tensor:
        """2048-d features before projection (for PCA)."""
        return self.backbone(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.get_raw_features(x))


class Decoder(nn.Module):
    """
    Transposed-convolution decoder: emb_dim → 3×224×224.

    Upsampling path: 7 → 14 → 28 → 56 → 112 → 224
    Each ConvTranspose2d(kernel=4, stride=2, padding=1) doubles spatial size.
    """

    def __init__(self, emb_dim: int):
        super().__init__()
        self.fc = nn.Linear(emb_dim, 512 * 7 * 7)
        self.net = nn.Sequential(
            # (B, 512, 7,   7)  → (B, 256, 14,  14)
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            # (B, 256, 14,  14) → (B, 128, 28,  28)
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            # (B, 128, 28,  28) → (B,  64, 56,  56)
            nn.ConvTranspose2d(128,  64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            # (B,  64, 56,  56) → (B,  32, 112, 112)
            nn.ConvTranspose2d( 64,  32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            # (B,  32, 112, 112) → (B,   3, 224, 224)
            nn.ConvTranspose2d( 32,   3, 4, 2, 1),
            # No activation: MSE loss in normalized pixel space (unbounded)
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        x = self.fc(emb).view(-1, 512, 7, 7)
        return self.net(x)


class AutoEncoder(nn.Module):
    """
    Denoising AE with optional joint classification branch.

    When num_classes is given, forward() returns (recon, emb, logits) and
    the model can be trained with alpha*MSE + (1-alpha)*CrossEntropy.
    When num_classes is None the classifier is absent and forward() returns
    (recon, emb, None) for reconstruction-only use.
    """

    def __init__(self, emb_dim: int, pretrained: bool = True,
                 freeze_backbone: bool = True, num_classes: int = None):
        super().__init__()
        self.encoder    = Encoder(emb_dim, pretrained, freeze_backbone)
        self.decoder    = Decoder(emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes) if num_classes else None
        self.emb_dim    = emb_dim

    def forward(self, x: torch.Tensor):
        emb     = self.encoder(x)
        recon   = self.decoder(emb)
        logits  = self.classifier(emb) if self.classifier is not None else None
        return recon, emb, logits

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def get_raw_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.get_raw_features(x)


def get_ae_param_groups(model: AutoEncoder,
                        lr_backbone: float,
                        lr_head: float,
                        weight_decay: float):
    """
    Returns optimiser param groups.  When backbone is frozen only the
    projection head and decoder params (requires_grad=True) are returned.
    When unfrozen, backbone gets lr_backbone and the rest get lr_head.
    """
    backbone_ids    = {id(p) for p in model.encoder.backbone.parameters()}
    backbone_params = [p for p in model.parameters()
                       if id(p) in backbone_ids and p.requires_grad]
    head_params     = [p for p in model.parameters()
                       if id(p) not in backbone_ids and p.requires_grad]
    groups = [{"params": head_params, "lr": lr_head, "weight_decay": weight_decay}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone,
                       "weight_decay": weight_decay})
    return groups
