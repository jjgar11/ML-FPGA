import torch.nn as nn


class TinyCNN(nn.Module):
    """
    Tiny CNN for GTSRB (43 traffic sign classes).
    Architecture is hls4ml-compatible: Conv2d + ReLU + MaxPool + Linear only.
    Input: (batch, 3, 32, 32)
    """

    def __init__(self, num_classes: int = 43):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),   # → 16×32×32
            nn.ReLU(),
            nn.MaxPool2d(2),                               # → 16×16×16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # → 32×16×16
            nn.ReLU(),
            nn.MaxPool2d(2),                               # → 32×8×8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → 64×8×8
            nn.ReLU(),
            nn.MaxPool2d(2),                               # → 64×4×4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
