# ptq_mnist.py (ejemplo)
import torch
from torchvision import datasets, transforms
from mnist_baseline import SimpleNN  # importa tu definición de modelo
import copy

# 1) cargar modelo
model = SimpleNN()
model.load_state_dict(torch.load("mnist_baseline.pth", map_location='cpu'))
model.eval()

# 2) preparar dataset de calibración (pequeño)
transform = transforms.Compose([transforms.ToTensor()])
calib_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
calib_loader = torch.utils.data.DataLoader(calib_ds, batch_size=64, shuffle=True)

# 3) preparar model for static quantization
model_fp32 = copy.deepcopy(model)
model_fp32.eval()
model_q = copy.deepcopy(model_fp32)

# Fusion (no hay conv/bn en MLP, pero es buena práctica)
# torch.quantization.fuse_modules(model_q, [['fc1','relu']])  # si tuvieras capas fusionables

# prepare
model_q.qconfig = torch.quantization.get_default_qconfig('fbgemm')  # cpu qconfig
torch.quantization.prepare(model_q, inplace=True)

# calibration: pasar algunos batches
num_calib_batches = 10
for i, (img, lbl) in enumerate(calib_loader):
    if i >= num_calib_batches: break
    model_q(img)  # forward para calibrar observadores

# convert to quantized
torch.quantization.convert(model_q, inplace=True)

# test accuracy quick check
test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1000, shuffle=False)
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model_q(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print("PTQ accuracy:", 100*correct/total)

# save quantized model (scripted)
model_q.cpu()
traced = torch.jit.script(model_q)
traced.save("mnist_int8_scripted.pt")
