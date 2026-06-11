import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out


class ResNetCIFAR(nn.Module):
    """
    CIFAR-style ResNet.

    This uses a 3x3 first convolution and no initial max-pooling layer.
    That is better for 32x32 CIFAR-10 images than the original ImageNet stem.
    """
    def __init__(self, block, layers, num_classes=10, dropout=0.0):
        super().__init__()

        self.in_channels = 64

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.stage1 = self._make_stage(block, 64, layers[0], stride=1)
        self.stage2 = self._make_stage(block, 128, layers[1], stride=2)
        self.stage3 = self._make_stage(block, 256, layers[2], stride=2)
        self.stage4 = self._make_stage(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_stage(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []

        for s in strides:
            blocks.append(block(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def ResNet11CIFAR(num_classes=10, dropout=0.0):
    """
    Smaller ResNet baseline. This is a practical ResNet-11-style model with
    one BasicBlock per stage.
    """
    return ResNetCIFAR(
        BasicBlock,
        layers=[1, 1, 1, 1],
        num_classes=num_classes,
        dropout=dropout
    )


def ResNet18CIFAR(num_classes=10, dropout=0.0):
    """
    ResNet-18 with two BasicBlocks per stage.
    """
    return ResNetCIFAR(
        BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=num_classes,
        dropout=dropout
    )
