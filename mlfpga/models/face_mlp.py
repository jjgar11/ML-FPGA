import torch.nn as nn


class FaceMLP(nn.Module):
    """
    Face recognition MLP — classifies a PCA embedding into N identities.
    Input:  (batch, n_components)  — PCA embedding from sklearn PCA(50)
    Output: (batch, num_classes)   — raw logits per person

    Same structure as WineNet (hls4ml s_axilite validated path).
    n_components=50 gives 50 AXI-Lite input registers — fully manageable.
    'Unknown' identity is handled by a confidence threshold in PS, not here.
    """

    def __init__(self, n_components: int = 50, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_components, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        return self.net(x)
