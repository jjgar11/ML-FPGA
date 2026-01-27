import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from mlfpga.config import DATA_ROOT, MODELS_ROOT


from mlfpga.models.digit_classification import DigitClassificationNN, QuantizableDigitClassificationNN


model = QuantizableDigitClassificationNN()

transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train_model(
    epochs=2,
    criterion=criterion,
    optimizer=optimizer,
    trainloader=trainloader,
    filename="mnist.pth"
)

model.test_model(testloader)

q_model = model.quantize_post_training(
    calib_loader=testloader,
    backend="qnnpack", 
    max_batches=10,
    filename="mnist_quantized.pth",
)

q_model.export_onnx(
    dummy_input = torch.randn(1, 1, 28, 28),
    filename = "mnist_quantized.onnx"
)