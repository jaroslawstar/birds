import torch.nn as nn


class MLPClassifier(nn.Module):
    """
    Small MLP classifier on top of frozen embeddings.
    Architecture: in_dim → [hidden_dims] → num_classes
    Each hidden layer uses BatchNorm + ReLU + Dropout.
    """

    def __init__(self, in_dim: int, num_classes: int,
                 hidden_dims: list = None, dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512]
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
