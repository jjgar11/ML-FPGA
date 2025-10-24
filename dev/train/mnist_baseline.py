import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from mlfpga.config import DATA_ROOT, MODELS_ROOT
from mlfpga.models import DigitClassificationNN as ModelNN

# 1. Dataset
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

model = ModelNN()

# 3. Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train_model(3, criterion, optimizer, trainloader)

# 4. Validation
model.test_model(testloader)

# 5. Save model
file_path = os.path.join(MODELS_ROOT, "mnist_baseline.pth")
torch.save(model.state_dict(), file_path)
print(f"Model saved at {file_path}")
