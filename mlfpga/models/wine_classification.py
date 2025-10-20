# mlfpga/models/wine_classification.py
import torch
import torch.nn as nn
import torch.quantization as quant

class WineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(13, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 3)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

class QuantizableWineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = quant.QuantStub()
        self.fc1 = nn.Linear(13, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 3)
        self.dequant = quant.DeQuantStub()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.quant(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.dequant(x)
        x = self.softmax(x)
        return x
