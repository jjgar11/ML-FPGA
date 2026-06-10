import torch.nn as nn


class NavCNN(nn.Module):
    """
    Navigation CNN for robot car (hls4ml-compatible, io_stream deployment).
    Input:  (batch, 1, 64, 64)  — grayscale
    Output: (batch, 4)          — forward=0, left=1, right=2, stop=3

    4 conv blocks halve spatial dims each time: 64 → 32 → 16 → 8 → 4 px.
    FC bottleneck is 1024×64 — manageable with ReuseFactor=16.
    io_parallel on 4096 inputs would be impractical; use io_stream + AXI DMA.
    Only uses Conv2d + ReLU + MaxPool2d + Flatten + Linear (all hls4ml-safe ops).
    """

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # → 16×64×64
            nn.ReLU(),
            nn.MaxPool2d(2),                               # → 16×32×32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # → 32×32×32
            nn.ReLU(),
            nn.MaxPool2d(2),                               # → 32×16×16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → 64×16×16
            nn.ReLU(),
            nn.MaxPool2d(2),                               # → 64×8×8
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # → 64×8×8
            nn.ReLU(),
            nn.MaxPool2d(2),                               # → 64×4×4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
