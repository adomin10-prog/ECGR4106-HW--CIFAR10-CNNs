import torch

from models.alexnet_cifar import AlexNetCIFAR
from models.vgg_cifar import VGGCIFAR
from models.resnet_cifar import ResNet11CIFAR, ResNet18CIFAR
from utils.metrics import count_trainable_parameters


def test_model(model, name):
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 10), f"{name} output shape is wrong: {y.shape}"
    print(f"{name}: output shape {y.shape}, params {count_trainable_parameters(model):,}")


def main():
    test_model(AlexNetCIFAR(dropout=0.0), "AlexNetCIFAR")
    test_model(VGGCIFAR(dropout=0.0), "VGGCIFAR")
    test_model(ResNet11CIFAR(dropout=0.0), "ResNet11CIFAR")
    test_model(ResNet18CIFAR(dropout=0.0), "ResNet18CIFAR")


if __name__ == "__main__":
    main()
