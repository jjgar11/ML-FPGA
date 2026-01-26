import os
import torch
import torch.nn as nn
import torch.ao.quantization as quant
from torchvision import datasets, transforms
from mlfpga.models import DigitClassificationNN as QuantizableNN
from torch.utils.data import DataLoader
from mlfpga.config import DATA_ROOT, MODELS_ROOT

# ============================
# 2. Load trained model
# ============================
model = QuantizableNN()
model.load_state_dict(torch.load(os.path.join(MODELS_ROOT, "mnist_baseline.pth"), map_location="cpu", weights_only=True))
model.eval()

# ============================
# 3. Configure quantization backend
# ============================
print("Supported engines:", torch.backends.quantized.supported_engines)
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

print("Calibrating model...")
with torch.no_grad():
    for i, (images, _) in enumerate(calib_loader):
        model_prep(images)
        if i > 10:
            break

# 5.1 Convert to quantized model
model_q = quant.convert(model_prep, inplace=False)
print("Quantized model ready")
print(model)
print(model_q)

# ============================
# 6. Test quantized model
# ============================
test_dataset = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

model_q.test_model()


# ============================
# 🔹 5. Save models
# ============================
file_path = os.path.join(MODELS_ROOT, "mnist_baseline_q.pth")
torch.save(model_q.state_dict(), file_path)
print(f"Model saved at {file_path}")
