# ptq_wine.py (asumiendo mlfpga.models.wine_classification.WineNet ya tiene QuantStub/DeQuantStub)
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.ao.quantization as quant

from mlfpga.models.wine_classification import QuantizableWineNet as model
from mlfpga.config import *

# Data
data = load_wine()
X = data['data']
y = data['target']
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16)

# Model
model = model()
model.load_state_dict(torch.load(os.path.join(MODELS_ROOT, "wine.pth"), map_location="cpu", weights_only=True))
model.eval()

# Configure backend
print("Supported quant engines:", torch.backends.quantized.supported_engines)
torch.backends.quantized.engine = "fbgemm"

# Prepare PTQ
model.qconfig = quant.get_default_qconfig(torch.backends.quantized.engine)
model_prep = quant.prepare(model, inplace=False)

# Calibration
with torch.no_grad():
    for i, (xb, _) in enumerate(train_loader):
        model_prep(xb)
        if i > 10:
            break

# Convert
model_q = quant.convert(model_prep, inplace=False)
print("Quantized model ready")
print(model)
print(model_q)

# Evaluate
model_q.eval()
correct = 0
total = 0
with torch.no_grad():
    for xb, yb in test_loader:
        preds = model_q(xb)
        correct += (preds.argmax(1) == yb).sum().item()
        total += yb.size(0)
print(f"Accuray in test (quantized): {100 * correct / total:.2f}%")


file_path = os.path.join(MODELS_ROOT, "wine_q.pth")
torch.save(model_q.state_dict(), file_path)
print(f"Model saved at {file_path}")