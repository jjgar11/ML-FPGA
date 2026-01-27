import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mlfpga.config import MODELS_ROOT

from mlfpga.models.wine_brevitas import WineBrevitasMLP_8b8b, WineBrevitasMLP_4b8b

def main(bit_width: int = 8, epochs: int = 20):
    # Data
    data = load_wine()
    X = data["data"]
    y = data["target"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16)

    # Model
    model = WineBrevitasMLP_8b8b()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    model.train()
    for epoch in range(epochs):
        running = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, loss={running/len(train_loader):.4f}")

    # Eval
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb)
            pred = out.argmax(dim=1)
            total += yb.size(0)
            correct += (pred == yb).sum().item()
    acc = 100.0 * correct / total
    print(f"Accuracy: {acc:.2f}%")

    # Save
    fname = "wine_brevitas_8b8b.pth"
    fpath = os.path.join(MODELS_ROOT, fname)
    torch.save(model.state_dict(), fpath)
    print(f"Saved: {fpath}")


if __name__ == "__main__":
    main(bit_width=8, epochs=20)
