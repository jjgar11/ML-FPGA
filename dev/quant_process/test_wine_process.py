import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mlfpga.config import DATA_ROOT, MODELS_ROOT
from mlfpga.models import WineClassificationNN


model = WineClassificationNN()

data = load_wine()
X = data['data']
y = data['target']
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
trainloader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
testloader = DataLoader(TensorDataset(X_test, y_test), batch_size=16)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train_model(
    epochs=20,
    criterion=criterion,
    optimizer=optimizer,
    trainloader=trainloader,
    filename="wine.pth"
)

model.test_model(testloader)

for name, param in model.named_parameters():
    print(name, param.dtype)


q_model = model.quantize_post_training(
    calib_loader=testloader,
    backend="qnnpack", 
    max_batches=10,
    filename="wine_q.pth",
)

q_model.export_onnx(
    dummy_input = torch.randn(1, 13),
    filename = "wine_quantized.onnx"
)