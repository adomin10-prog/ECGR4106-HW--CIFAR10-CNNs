# Homework 1 Report Outline

## Cover Page

Name: Andrew Dominguez Luna  
Student ID: [Add student ID]  
Homework: Homework 1  
GitHub Repository: [Add public GitHub link]  

---

## Common Experimental Setup

- Dataset: CIFAR-10
- Split: 45,000 training / 5,000 validation / 10,000 test
- Seed: 42
- Augmentation: random crop with padding 4 and random horizontal flip
- Normalization: CIFAR-10 mean and standard deviation
- Loss: cross-entropy
- Optimizer: SGD with momentum
- Scheduler: cosine annealing
- Batch size: 128
- Hardware: [Example: Google Colab Tesla T4]

---

## Problem 1: Modified AlexNet on CIFAR-10

### Part A: Simplified Architecture

Include:

- Explanation of original AlexNet
- Why original AlexNet is too large for CIFAR-10
- Architecture table for modified AlexNet
- Parameter count
- Comparison against original AlexNet parameter count
- Training loss curve
- Validation loss curve
- Validation accuracy curve
- Final test accuracy
- Confusion matrix
- First-layer filter visualization
- Discussion of learned filters

### Part B: Dropout

Include:

- Baseline vs dropout p = 0.3 vs dropout p = 0.5
- Loss and accuracy curve comparison
- Whether dropout reduced the train/validation gap
- Which dropout rate worked best

---

## Problem 2: Modified VGGNet on CIFAR-10

### Part A: Adapted VGGNet

Include:

- Review of VGG-11, VGG-13, VGG-16, VGG-19
- Justification for selected VGG configuration
- Architecture table
- Parameter count
- Training and validation curves
- Final test accuracy
- Confusion matrix
- AlexNet vs VGG comparison table

### Part B: Dropout and Cross-Model Comparison

Include:

- VGG baseline vs dropout p = 0.3 vs p = 0.5
- Comparison to AlexNet dropout results
- Discussion of whether the deeper model responds differently to dropout

---

## Problem 3: ResNet-11 vs ResNet-18 on CIFAR-10

### Part A: ResNet-18 Implementation

Include:

- BasicBlock explanation
- Skip connection explanation
- Downsampling shortcut explanation
- Why 3x3 conv and no max pool are used for CIFAR-10
- ResNet-11 vs ResNet-18 results
- Training/validation curves
- Confusion matrices
- Parameter, accuracy, and training time comparison

### Part B: Dropout and Final Comparison

Include:

- Dropout effect on ResNet-11 and ResNet-18
- Whether dropout helps even though ResNet already uses BatchNorm
- Final comparison table across best AlexNet, best VGGNet, best ResNet-11, and best ResNet-18
- Bar chart of final test accuracies
- Discussion of performance vs complexity
