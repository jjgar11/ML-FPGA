import torch
import torch.nn as nn


class TinyCNN(nn.Module):
    """
    Tiny CNN for GTSRB (43 traffic sign classes).
    Architecture is hls4ml-compatible: Conv2d + ReLU + MaxPool + Linear only.
    Input: (batch, 3, 32, 32)
    ~160 K parameters. Use TinyCNN_GAP for FPGA synthesis (FC1 is 1024→128 here).
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


class TinyCNN_GAP(nn.Module):
    """
    FPGA-optimised GTSRB classifier.
    Conv×3 (3→32→64→128, BN+ReLU+MaxPool) → GlobalAveragePooling → Linear(128,256) → Linear(256,43).
    ~29 K parameters. GAP replaces Flatten+FC1 — resolution-agnostic and hls4ml-compatible.
    Trained at 64×64; synthesis uses input_shape (1,3,64,64).
    """

    def __init__(self, num_classes: int = 43):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AvgPool2d(kernel_size=8),  # GAP equivalente a AdaptiveAvgPool2d(1) para 64px input (8×8 spatial aquí)
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class TinyCNN_GAP_S(nn.Module):
    """
    AXI-Lite FPGA variant of TinyCNN_GAP for synthesis testing.
    32×32 input, reduced channels (3→16→32→64), no Dropout.
    GAP: AvgPool2d(4) on 4×4 spatial (after 3×MaxPool2d(2) from 32px).
    3×32×32 = 3072 inputs → feasible for AXI-Lite register interface.
    """

    def __init__(self, num_classes: int = 43):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 32→16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 16→8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 8→4
            nn.AvgPool2d(kernel_size=4),                   # GAP: 4×4→1×1
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
