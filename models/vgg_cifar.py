import torch.nn as nn


def make_vgg_layers(cfg):
    layers = []
    in_channels = 3

    for item in cfg:
        if item == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            out_channels = item
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ])
            in_channels = out_channels

    return nn.Sequential(*layers)


class VGGCIFAR(nn.Module):
    """
    Adapted VGG-11 style network for CIFAR-10.

    The channel count is capped at 256 to keep the parameter count near the
    modified AlexNet model. The classifier is also reduced compared with
    original VGG.
    """
    def __init__(self, num_classes: int = 10, dropout: float = 0.0):
        super().__init__()

        cfg = [64, "M", 128, "M", 256, 256, "M", 256, 256, "M", 256, 256, "M"]

        self.features = make_vgg_layers(cfg)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 1 * 1, 512),
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
