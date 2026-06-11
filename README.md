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

1. Open Colab.
2. Go to `Runtime → Change runtime type → Hardware accelerator → T4 GPU`.
3. Run:

```python
!git clone https://github.com/adomin10-prog/ECGR4106-HW--CIFAR10-CNNs.git
%cd ECGR4106-HW--CIFAR10-CNNs
!pip install -r requirements.txt
```

Check GPU:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
```

Run one experiment:

```python
!python train.py --model alexnet --dropout 0.0 --epochs 30 --lr 0.01
```

## Main Experiment Commands

### Problem 1: Modified AlexNet

```bash
python train.py --model alexnet --dropout 0.0 --epochs 30 --lr 0.01
python train.py --model alexnet --dropout 0.3 --epochs 30 --lr 0.01
python train.py --model alexnet --dropout 0.5 --epochs 30 --lr 0.01
```

### Problem 2: Adapted VGGNet

```bash
python train.py --model vgg --dropout 0.0 --epochs 30 --lr 0.01
python train.py --model vgg --dropout 0.3 --epochs 30 --lr 0.01
python train.py --model vgg --dropout 0.5 --epochs 30 --lr 0.01
```

### Problem 3: ResNet-11 and ResNet-18

```bash
python train.py --model resnet11 --dropout 0.0 --epochs 50 --lr 0.1
python train.py --model resnet11 --dropout 0.3 --epochs 50 --lr 0.1
python train.py --model resnet11 --dropout 0.5 --epochs 50 --lr 0.1

python train.py --model resnet18 --dropout 0.0 --epochs 50 --lr 0.1
python train.py --model resnet18 --dropout 0.3 --epochs 50 --lr 0.1
python train.py --model resnet18 --dropout 0.5 --epochs 50 --lr 0.1
```

## Output Files

Each experiment creates a folder in `results/`, for example:

```text
results/alexnet_dropout00/
```

Each result folder contains:

| File | Purpose |
|---|---|
| `config.json` | Hyperparameters, seed, hardware, parameter count |
| `training_log.csv` | Training and validation metrics per epoch |
| `final_results.json` | Final test loss, test accuracy, best validation accuracy |
| `loss_curve.png` | Training loss vs validation loss |
| `val_accuracy_curve.png` | Validation accuracy curve |
| `confusion_matrix.png` | Test-set confusion matrix |
| `best_model.pt` | Best model checkpoint, ignored by Git by default |

The `.pt` files are ignored by Git because they can be large. Commit the CSV, JSON, and PNG files for the final submission.

## Summary Table and Bar Chart

After all experiments are complete:

```bash
python plot_summary.py
```

This creates:

```text
results/summary_table.csv
results/best_model_test_accuracy_bar_chart.png
```

## Quick Model Test

```bash
python quick_model_test.py
```

## Final Submission

The PDF report should include your name, student ID, homework number, and the public GitHub repository link.
