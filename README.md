# ECGR 4106 Homework 1: CIFAR-10 CNN Comparison

**Student:** Andrew Dominguez Luna  
**Course:** ECGR 4106 Deep Learning  
**Homework:** Homework 1  
**Dataset:** CIFAR-10  
**Framework:** PyTorch  

This repository contains the source code for Homework 1. The assignment compares Modified AlexNet, Adapted VGGNet, ResNet-11, and ResNet-18 on CIFAR-10. Each model uses the same train/validation/test split and the same training pipeline.

## Repository Layout

```text
ECGR4106-HW1-CIFAR10-CNNs/
├── README.md
├── requirements.txt
├── .gitignore
├── HW1_Colab_Run_All.ipynb
├── train.py
├── plot_summary.py
├── run_all_experiments.py
├── quick_model_test.py
├── models/
│   ├── alexnet_cifar.py
│   ├── vgg_cifar.py
│   ├── resnet_cifar.py
│   └── filter_viz.py
├── utils/
│   ├── data.py
│   ├── metrics.py
│   ├── plotting.py
│   └── seed.py
├── results/
└── report/
```

## Dataset Split

The code automatically downloads CIFAR-10 using `torchvision.datasets.CIFAR10`.

| Split | Images |
|---|---:|
| Training | 45,000 |
| Validation | 5,000 |
| Test | 10,000 |

Random seed: `42`

## Common Training Setup

| Item | Value |
|---|---|
| Loss | CrossEntropyLoss |
| Optimizer | SGD |
| Momentum | 0.9 |
| Weight decay | 5e-4 |
| Scheduler | CosineAnnealingLR |
| Batch size | 128 |
| AlexNet/VGG epochs | 30 |
| ResNet epochs | 50 |
| Augmentation | RandomCrop(32, padding=4), RandomHorizontalFlip |
| Normalization | CIFAR-10 mean/std |

## Run in Google Colab

To run this repository in Google Colab:

1. Open Google Colab.
2. Go to `Runtime → Change runtime type`.
3. Set the hardware accelerator to `T4 GPU`.
4. Run the following cell:

```python
 !git clone https://github.com/adomin10-prog/ECGR4106-HW--CIFAR10-CNNs.git
%cd ECGR4106-HW--CIFAR10-CNNs
!pip install -r requirements.txt

# Run professional verification script
!python run_verification.py --mode verification

# Display verification results
import pandas as pd
from IPython.display import display, Image

display(pd.read_csv("verification_results/verification_summary.csv"))
display(Image(filename="verification_results/verification_accuracy_bar_chart.png"))
```

This cell downloads the GitHub repository, moves into the project folder, installs the required Python packages, runs the repository verification script, and displays the generated results.

The verification script trains each required model and dropout configuration for a short run to confirm that the repository is working correctly. After it finishes, the notebook displays a summary table and an accuracy bar chart from the `verification_results/` folder.

