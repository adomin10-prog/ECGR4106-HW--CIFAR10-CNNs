import torch
import torch.nn as nn


class AlexNetCIFAR(nn.Module):
    """
    Modified AlexNet for CIFAR-10.

    Main changes from original AlexNet:
    - Uses 3x3 convolution kernels instead of the original large first kernel.
    - Uses fewer channels because CIFAR-10 is smaller than ImageNet.
    - Uses less aggressive downsampling to avoid collapsing 32x32 images too quickly.
    - Uses much smaller fully connected layers.
    - Dropout is optional so baseline and dropout variants can be compared.
    """
    def __init__(self, num_classes: int = 10, dropout: float = 0.0):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),    # 32x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # 16x16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 16x16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # 8x8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # 8x8
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), # 8x8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # 4x4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_first_conv_filters(model):
    """
    Returns first conv filters for visualization.
    Shape: out_channels x in_channels x kernel_h x kernel_w
    """
    return model.features[0].weight.detach().cpu()
