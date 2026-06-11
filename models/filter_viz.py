from pathlib import Path

import torch
import matplotlib.pyplot as plt

from models.alexnet_cifar import AlexNetCIFAR


def visualize_alexnet_first_layer(checkpoint_path: str, output_path: str):
    """
    Use this after training AlexNet.

    Example:
    python -c "from models.filter_viz import visualize_alexnet_first_layer; visualize_alexnet_first_layer('results/alexnet_dropout00/best_model.pt', 'results/alexnet_dropout00/first_layer_filters.png')"
    """
    model = AlexNetCIFAR(dropout=0.0)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)

    filters = model.features[0].weight.detach().cpu()

    # Normalize each filter to [0, 1] for display.
    filters = filters.clone()
    for i in range(filters.size(0)):
        f = filters[i]
        f_min = f.min()
        f_max = f.max()
        filters[i] = (f - f_min) / (f_max - f_min + 1e-8)

    n_filters = min(32, filters.size(0))
    cols = 8
    rows = (n_filters + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(10, 5))

    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n_filters:
            img = filters[i].permute(1, 2, 0)
            ax.imshow(img)

    plt.suptitle("First Convolutional Layer Filters")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
