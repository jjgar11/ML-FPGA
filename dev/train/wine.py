import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mlfpga.config import MODELS_ROOT
from mlfpga.models import WineClassificationNN as ModelNN

# 1. Cargar dataset Wine
data = load_wine()
X = data['data']
y = data['target']

# 2. Escalar features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3. Convertir a tensores
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# 4. Dividir en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=16)

model = ModelNN()

# 6. Entrenamiento básico
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(3):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 7. Evaluación antes de cuantizar
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for xb, yb in test_loader:
        preds = model(xb)
        correct += (preds.argmax(1) == yb).sum().item()
        total += yb.size(0)
print("Accuracy before quantization:", correct/total*100)

# 8. Save model
file_path = os.path.join(MODELS_ROOT, "wine.pth")
torch.save(model.state_dict(), file_path)
print(f"Model saved at {file_path}")