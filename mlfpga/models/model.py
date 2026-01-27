import torch
import torch.nn as nn
import torch.quantization as quant
import os
from mlfpga.config import MODELS_ROOT

class BaseNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_path = MODELS_ROOT
        os.makedirs(self.model_path, exist_ok=True)

    def save_model(self, filename):
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.quantized.Linear):
                weight = module.weight()
                bias = module.bias()
                print(f"{name}: weight dtype = {weight.dtype}, shape = {weight.shape}")
                if bias is not None:
                    print(f"{name}: bias dtype = {bias.dtype}, shape = {bias.shape}")

        file_path = os.path.join(self.model_path, filename)
        torch.save(self.state_dict(), file_path)
        print(f"Model saved at {file_path}")

    def train_model(self, epochs, criterion, optimizer, trainloader, filename=None):
        self.train()
        print("Training...")
        for epoch in range(epochs):
            for images, labels in trainloader:
                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

        if filename:
            self.save_model(filename)

    def test_model(self, testloader):
        self.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in testloader:
                outputs = self(X)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        acc = 100 * correct / total
        print(f"Accuracy in test: {acc:.2f}%")
        return acc

    def quantize_post_training(self, calib_loader, backend="qnnpack", max_batches=10, filename=None):
        print(f"Using quantization backend: {backend}")
        torch.backends.quantized.engine = backend

        self.qconfig = quant.get_default_qconfig(torch.backends.quantized.engine)
        prepared_model = quant.prepare(self, inplace=False)

        print("Calibrating model...")
        with torch.no_grad():
            for i, (X, _) in enumerate(calib_loader):
                prepared_model(X)
                if i >= max_batches:
                    break

        quantized_model = quant.convert(prepared_model, inplace=False)
        print("Quantization complete.")
        quantized_model.test_model(calib_loader)

        if filename:
            quantized_model.save_model(filename)

        return quantized_model

    def export_onnx(self, dummy_input, filename, opset_version=13,
                    input_names=("input",), output_names=("logits",), dynamic_batch=True):
        save_path = os.path.join(self.model_path, filename)
        self.eval()
        dyn = None
        if dynamic_batch:
            dyn = {input_names[0]: {0: "batch"}, output_names[0]: {0: "batch"}}

        torch.onnx.export(
            self, dummy_input, save_path,
            opset_version=opset_version,
            input_names=list(input_names),
            output_names=list(output_names),
            dynamic_axes=dyn,
        )
        print(f"Model exported to ONNX: {save_path}")
