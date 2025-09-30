import torch
import torch.nn as nn
import torch.ao.quantization as quant
from torchvision import datasets, transforms
from mlfpga.models import QuantizableDigitClassificationNN as QuantizableNN
from torch.utils.data import DataLoader
from mlfpga.config import *

# ============================
# 2. Load trained model
# ============================
model = QuantizableNN()
model.load_state_dict(torch.load(os.path.join(MODELS_ROOT, "mnist_baseline.pth"), map_location="cpu", weights_only=True))
model.eval()

# ============================
# 3. Configure quantization backend
# ============================
print("Motores disponibles:", torch.backends.quantized.supported_engines)
torch.backends.quantized.engine = "fbgemm"  # Cambia a "qnnpack" si estás en ARM/Raspberry

# ============================
# 4. Prepare calibration dataset
# ============================
transform = transforms.Compose([transforms.ToTensor()])
calib_dataset = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
calib_loader = DataLoader(calib_dataset, batch_size=64, shuffle=True)

# ============================
# 5. Post-Training Quantization
# ============================
model.qconfig = quant.get_default_qconfig(torch.backends.quantized.engine)

model_prep = quant.prepare(model, inplace=False)

print("Calibrando modelo...")
with torch.no_grad():
    for i, (images, _) in enumerate(calib_loader):
        model_prep(images)
        if i > 10:
            break

print(torch.backends.quantized.supported_engines)


# d) Convert to quantized model
model_q = quant.convert(model_prep, inplace=False)
print("Quantized model ready")
print(model_q)

# ============================
# 6. Test quantized model
# ============================
test_dataset = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model_q(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuray in test (quantized): {100 * correct / total:.2f}%")


# ============================
# 🔹 5. Save models
# ============================
torch.save(model_q.state_dict(), os.path.join(MODELS_ROOT, "mnist_baseline_q.pth"))
print("Model saved at mnist_baseline.pth")
