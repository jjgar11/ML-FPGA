import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
from mlfpga.models import QuantizableDigitClassificationNN as QuantizableNN
from mlfpga.config import *

# 1. Test dataset
transform = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

# 2. Load quantized model
model_int8 = QuantizableNN()
model_int8.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
model_int8_fp = torch.quantization.prepare(model_int8, inplace=False)
model_int8 = torch.quantization.convert(model_int8_fp, inplace=False)

model_int8.load_state_dict(torch.load(os.path.join(MODELS_ROOT, "mnist_baseline_q.pth")))
model_int8.eval()

# 3. Evaluate in test
correct, total = 0, 0
with torch.no_grad():
    for images, labels in testloader:
        outputs = model_int8(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test accuracy (quantized model): {accuracy:.2f}%")

# 4. Export to ONNX
dummy_input = torch.randn(1, 1, 28, 28)
onnx_filename = MODELS_ROOT+"/mnist_quantized.onnx"
torch.onnx.export(model_int8, dummy_input, onnx_filename, opset_version=13)
print(f"Model saved at {onnx_filename}")

# 6. Compare sizes
pth_size = os.path.getsize(os.path.join(MODELS_ROOT, "mnist_baseline.pth")) / 1024
onnx_size = os.path.getsize(onnx_filename) / 1024
print(f"Size .pth: {pth_size:.1f} KB")
print(f"Size .onnx: {onnx_size:.1f} KB")
