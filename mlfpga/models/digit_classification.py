import torch
import torch.nn as nn
import torch.quantization as quant
from .model import BaseNet

class DigitClassificationNN(BaseNet):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class QuantizableDigitClassificationNN(BaseNet):
    def __init__(self):
        super().__init__()
        self.quant = quant.QuantStub()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dequant = quant.DeQuantStub()

    def forward(self, x):
        # 🔹 Convert to int8
        x = self.quant(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        # 🔹 Convert back to float32
        x = self.dequant(x)
        return x