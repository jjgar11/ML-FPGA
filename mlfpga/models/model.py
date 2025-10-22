import torch
import torch.nn as nn
import torch.quantization as quant


class BaseNet(nn.Module):

    def train_model(self, epochs, criterion, optimizer, trainloader):
        self.train()
        print("Training...")
        for epoch in range(epochs):
            self.train()
            for images, labels in trainloader:
                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")  

    def test_model(self, testloader):
        self.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in testloader:
                outputs = self(X)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        print(f"Accuracy in test: {100 * correct / total:.2f}%")

    def quantize(self):
        pass