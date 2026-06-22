# ECGR 4106 Homework 1 — CIFAR-10 CNN Comparison

**Student:** Andrew Dominguez Luna  
**Course:** ECGR 4106 Deep Learning  
**Homework:** Homework 1    

GitHub repository:

`https://github.com/adomin10-prog/ECGR4106-HW--CIFAR10-CNNs`


## 1. Set Google Colab to GPU

Before running:

1. Click `Runtime`
2. Click `Change runtime type`
3. Set `Hardware accelerator` to `T4 GPU`
4. Click `Save`

Running the full homework experiment on CPU is not recommended. CPU takes too long.

## 2. Clone the GitHub repository

This cell downloads the repository into Colab.


```python
import os
import shutil
from pathlib import Path

REPO_URL = "https://github.com/adomin10-prog/ECGR4106-HW--CIFAR10-CNNs.git"
REPO_DIR = Path("/content/ECGR4106-HW--CIFAR10-CNNs")

if REPO_DIR.exists():
    shutil.rmtree(REPO_DIR)

!git clone {REPO_URL} {REPO_DIR}

%cd {REPO_DIR}
print("Current folder:", Path.cwd())
```

    Cloning into '/content/ECGR4106-HW--CIFAR10-CNNs'...
    remote: Enumerating objects: 80, done.[K
    remote: Counting objects: 100% (80/80), done.[K
    remote: Compressing objects: 100% (75/75), done.[K
    remote: Total 80 (delta 24), reused 0 (delta 0), pack-reused 0 (from 0)[K
    Receiving objects: 100% (80/80), 40.94 KiB | 5.12 MiB/s, done.
    Resolving deltas: 100% (24/24), done.
    /content/ECGR4106-HW--CIFAR10-CNNs
    Current folder: /content/ECGR4106-HW--CIFAR10-CNNs


## 3. Install requirements

This installs the packages listed in the repo's `requirements.txt`.


```python
%pip install -q -r requirements.txt
```

## 4. Confirm GPU and PyTorch setup


```python
import sys
import platform
import torch

print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Selected device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: GPU is not enabled. Go to Runtime -> Change runtime type -> T4 GPU.")
```

    Python: 3.12.13
    PyTorch: 2.11.0+cu128
    CUDA available: True
    Selected device: cuda
    GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition


### 5. Import the code from the GitHub repository

This cell imports the existing model, data, metric, plotting, and seed functions from the repo.



```python
import csv
import json
import time
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from IPython.display import display, Image

# These imports come directly from the GitHub repository files.
from models.alexnet_cifar import AlexNetCIFAR
from models.vgg_cifar import VGGCIFAR
from models.resnet_cifar import ResNet11CIFAR, ResNet18CIFAR

from utils.data import get_cifar10_loaders, CIFAR10_CLASSES
from utils.metrics import evaluate, count_trainable_parameters
from utils.plotting import plot_curves, plot_confusion_matrix
from utils.seed import set_seed

# These functions are also already defined in train.py.
from train import build_model, train_one_epoch

print("Repository code imported successfully.")
```

    Repository code imported successfully.


## 7. Experiment settings

For final submission, leave:

```python
RUN_FULL_HOMEWORK = True
```


```python
RUN_FULL_HOMEWORK = True

SEED = 42
BATCH_SIZE = 128
NUM_WORKERS = 2

DATA_DIR = "./data"
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

set_seed(SEED)

print("RUN_FULL_HOMEWORK:", RUN_FULL_HOMEWORK)
print("Batch size:", BATCH_SIZE)
print("Results folder:", RESULTS_DIR.resolve())
```

    RUN_FULL_HOMEWORK: True
    Batch size: 128
    Results folder: /content/ECGR4106-HW--CIFAR10-CNNs/results


## 8. Load CIFAR-10 using the repo's data function



```python
train_loader, val_loader, test_loader = get_cifar10_loaders(
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    seed=SEED,
    num_workers=NUM_WORKERS
)

print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Test batches:", len(test_loader))

images, labels = next(iter(train_loader))
print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
```

    100%|██████████| 170M/170M [52:16<00:00, 54.4kB/s]


    Training batches: 352
    Validation batches: 40
    Test batches: 79
    Image batch shape: torch.Size([128, 3, 32, 32])
    Label batch shape: torch.Size([128])


## 9. Check model output shapes and parameter counts

The model classes come directly from the repo:

- `models/alexnet_cifar.py`
- `models/vgg_cifar.py`
- `models/resnet_cifar.py`

The `build_model()` function comes from `train.py`.


```python
model_names = ["alexnet", "vgg", "resnet11", "resnet18"]

for model_name in model_names:
    model = build_model(model_name, dropout=0.0)
    sample_output = model(torch.zeros(2, 3, 32, 32))
    params = count_trainable_parameters(model)

    print(
        f"{model_name:8s} | output shape: {tuple(sample_output.shape)} | "
        f"trainable parameters: {params:,}"
    )
```

    alexnet  | output shape: (2, 10) | trainable parameters: 3,192,458
    vgg      | output shape: (2, 10) | trainable parameters: 3,586,698
    resnet11 | output shape: (2, 10) | trainable parameters: 4,903,242
    resnet18 | output shape: (2, 10) | trainable parameters: 11,173,962


## 10. Experiment list

These are the full homework model/dropout configurations.

- AlexNet and VGG run for 30 epochs.
- ResNet-11 and ResNet-18 run for 50 epochs.
- Each model is tested with dropout values of `0.0`, `0.3`, and `0.5`.


```python
FULL_EXPERIMENTS = [
    # model, dropout, epochs, learning rate
    ("alexnet", 0.0, 30, 0.01),
    ("alexnet", 0.3, 30, 0.01),
    ("alexnet", 0.5, 30, 0.01),

    ("vgg", 0.0, 30, 0.01),
    ("vgg", 0.3, 30, 0.01),
    ("vgg", 0.5, 30, 0.01),

    ("resnet11", 0.0, 50, 0.1),
    ("resnet11", 0.3, 50, 0.1),
    ("resnet11", 0.5, 50, 0.1),

    ("resnet18", 0.0, 50, 0.1),
    ("resnet18", 0.3, 50, 0.1),
    ("resnet18", 0.5, 50, 0.1),
]

QUICK_TEST_EXPERIMENTS = [
    ("alexnet", 0.0, 1, 0.01),
    ("vgg", 0.0, 1, 0.01),
    ("resnet11", 0.0, 1, 0.1),
    ("resnet18", 0.0, 1, 0.1),
]

experiments = FULL_EXPERIMENTS if RUN_FULL_HOMEWORK else QUICK_TEST_EXPERIMENTS

experiment_df = pd.DataFrame(
    experiments,
    columns=["model", "dropout", "epochs", "learning_rate"]
)

display(experiment_df)
```



  <div id="df-961eb9e1-87bc-4eb4-9c2b-e6acd794f8f6" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>dropout</th>
      <th>epochs</th>
      <th>learning_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>alexnet</td>
      <td>0.0</td>
      <td>30</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>alexnet</td>
      <td>0.3</td>
      <td>30</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>alexnet</td>
      <td>0.5</td>
      <td>30</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>vgg</td>
      <td>0.0</td>
      <td>30</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>vgg</td>
      <td>0.3</td>
      <td>30</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>5</th>
      <td>vgg</td>
      <td>0.5</td>
      <td>30</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>6</th>
      <td>resnet11</td>
      <td>0.0</td>
      <td>50</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>7</th>
      <td>resnet11</td>
      <td>0.3</td>
      <td>50</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>8</th>
      <td>resnet11</td>
      <td>0.5</td>
      <td>50</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>9</th>
      <td>resnet18</td>
      <td>0.0</td>
      <td>50</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>10</th>
      <td>resnet18</td>
      <td>0.3</td>
      <td>50</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>11</th>
      <td>resnet18</td>
      <td>0.5</td>
      <td>50</td>
      <td>0.10</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-961eb9e1-87bc-4eb4-9c2b-e6acd794f8f6')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-961eb9e1-87bc-4eb4-9c2b-e6acd794f8f6 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-961eb9e1-87bc-4eb4-9c2b-e6acd794f8f6');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


  <div id="id_ce0b8a1d-764d-4f48-812b-91cea091de50">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('experiment_df')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_ce0b8a1d-764d-4f48-812b-91cea091de50 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('experiment_df');
      }
      })();
    </script>
  </div>

    </div>
  </div>



## 11. Notebook experiment runner using repo functions

This cell is the notebook version of the training process.

It uses the existing repo functions:

- `build_model()` from `train.py`
- `train_one_epoch()` from `train.py`
- `evaluate()` from `utils/metrics.py`
- `plot_curves()` from `utils/plotting.py`
- `plot_confusion_matrix()` from `utils/plotting.py`
- `count_trainable_parameters()` from `utils/metrics.py`




```python
def run_experiment_from_repo_code(
    model_name: str,
    dropout: float,
    epochs: int,
    learning_rate: float,
    skip_if_finished: bool = True,
):
    dropout_tag = str(dropout).replace(".", "")
    run_name = f"{model_name}_dropout{dropout_tag}"
    output_dir = RESULTS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.json"
    log_path = output_dir / "training_log.csv"
    best_model_path = output_dir / "best_model.pt"
    final_results_path = output_dir / "final_results.json"

    if skip_if_finished and config_path.exists() and log_path.exists() and final_results_path.exists():
        print(f"Skipping completed run: {run_name}")
        log_df = pd.read_csv(log_path)
        with open(final_results_path, "r") as f:
            final_results = json.load(f)

        return {
            "run": run_name,
            "model": model_name,
            "dropout": dropout,
            "epochs": epochs,
            "parameters": final_results["parameters"],
            "best_validation_accuracy": final_results["best_validation_accuracy"],
            "test_accuracy": final_results["test_accuracy"],
            "test_loss": final_results["test_loss"],
            "avg_epoch_time_sec": log_df["epoch_time_sec"].mean(),
            "device": str(device),
        }

    set_seed(SEED)

    model = build_model(model_name, dropout).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=5e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    param_count = count_trainable_parameters(model)

    config = {
        "model": model_name,
        "dropout": dropout,
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "learning_rate": learning_rate,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "scheduler": "CosineAnnealingLR",
        "seed": SEED,
        "train_size": 45000,
        "validation_size": 5000,
        "test_size": 10000,
        "parameters": param_count,
        "device": str(device),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    if torch.cuda.is_available():
        config["gpu_name"] = torch.cuda.get_device_name(0)
        config["cuda_version"] = torch.version.cuda

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "epoch_time_sec",
            "learning_rate"
        ])

    best_val_acc = 0.0

    print("=" * 90)
    print(f"Run: {run_name}")
    print(f"Device: {device}")
    print(f"Trainable parameters: {param_count:,}")
    print("=" * 90)

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc, _ = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step()

        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                epoch_time,
                current_lr
            ])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} | "
            f"val acc {val_acc:.4f} | "
            f"time {epoch_time:.1f}s"
        )

    model.load_state_dict(
        torch.load(best_model_path, map_location=device)
    )

    test_loss, test_acc, test_cm = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    final_results = {
        "best_validation_accuracy": best_val_acc,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "parameters": param_count
    }

    with open(final_results_path, "w") as f:
        json.dump(final_results, f, indent=4)

    plot_curves(
        log_csv_path=str(log_path),
        output_dir=str(output_dir),
        title=run_name
    )

    plot_confusion_matrix(
        cm=test_cm,
        class_names=CIFAR10_CLASSES,
        output_path=str(output_dir / "confusion_matrix.png"),
        title=f"{run_name}: Test Confusion Matrix"
    )

    print(f"Final test accuracy for {run_name}: {test_acc:.4f}")
    print(f"Results saved to: {output_dir}")

    log_df = pd.read_csv(log_path)

    return {
        "run": run_name,
        "model": model_name,
        "dropout": dropout,
        "epochs": epochs,
        "parameters": param_count,
        "best_validation_accuracy": best_val_acc,
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "avg_epoch_time_sec": log_df["epoch_time_sec"].mean(),
        "device": str(device),
    }
```

## 12. Run all experiments

This can take a long time because it trains all required models.


```python
all_results = []

for model_name, dropout, epochs, learning_rate in experiments:
    result = run_experiment_from_repo_code(
        model_name=model_name,
        dropout=dropout,
        epochs=epochs,
        learning_rate=learning_rate,
        skip_if_finished=True
    )

    all_results.append(result)

summary_df = pd.DataFrame(all_results)
display(summary_df)
```

    ==========================================================================================
    Run: alexnet_dropout00
    Device: cuda
    Trainable parameters: 3,192,458
    ==========================================================================================
    Epoch 001/30 | train loss 2.0834 | train acc 0.2198 | val loss 1.7531 | val acc 0.3454 | time 3.0s
    Epoch 002/30 | train loss 1.6180 | train acc 0.3989 | val loss 1.4097 | val acc 0.4840 | time 2.1s
    Epoch 003/30 | train loss 1.3926 | train acc 0.4855 | val loss 1.2565 | val acc 0.5458 | time 2.0s
    Epoch 004/30 | train loss 1.2320 | train acc 0.5511 | val loss 1.1117 | val acc 0.5906 | time 2.0s
    Epoch 005/30 | train loss 1.0838 | train acc 0.6127 | val loss 0.9294 | val acc 0.6690 | time 2.0s
    Epoch 006/30 | train loss 0.9557 | train acc 0.6607 | val loss 0.8685 | val acc 0.6944 | time 2.0s
    Epoch 007/30 | train loss 0.8603 | train acc 0.6961 | val loss 0.7653 | val acc 0.7244 | time 2.0s
    Epoch 008/30 | train loss 0.7899 | train acc 0.7210 | val loss 0.6751 | val acc 0.7560 | time 2.0s
    Epoch 009/30 | train loss 0.7236 | train acc 0.7465 | val loss 0.7300 | val acc 0.7402 | time 2.0s
    Epoch 010/30 | train loss 0.6847 | train acc 0.7592 | val loss 0.6584 | val acc 0.7598 | time 2.0s
    Epoch 011/30 | train loss 0.6433 | train acc 0.7743 | val loss 0.6185 | val acc 0.7714 | time 2.0s
    Epoch 012/30 | train loss 0.6034 | train acc 0.7888 | val loss 0.5802 | val acc 0.7966 | time 2.0s
    Epoch 013/30 | train loss 0.5675 | train acc 0.8027 | val loss 0.5350 | val acc 0.8086 | time 2.1s
    Epoch 014/30 | train loss 0.5356 | train acc 0.8131 | val loss 0.5209 | val acc 0.8140 | time 2.0s
    Epoch 015/30 | train loss 0.5095 | train acc 0.8226 | val loss 0.5161 | val acc 0.8182 | time 2.0s
    Epoch 016/30 | train loss 0.4816 | train acc 0.8322 | val loss 0.5002 | val acc 0.8246 | time 2.0s
    Epoch 017/30 | train loss 0.4550 | train acc 0.8394 | val loss 0.4914 | val acc 0.8270 | time 2.0s
    Epoch 018/30 | train loss 0.4369 | train acc 0.8481 | val loss 0.4929 | val acc 0.8272 | time 2.0s
    Epoch 019/30 | train loss 0.4151 | train acc 0.8542 | val loss 0.4914 | val acc 0.8300 | time 2.0s
    Epoch 020/30 | train loss 0.3927 | train acc 0.8623 | val loss 0.4535 | val acc 0.8398 | time 2.0s
    Epoch 021/30 | train loss 0.3761 | train acc 0.8684 | val loss 0.4631 | val acc 0.8416 | time 2.0s
    Epoch 022/30 | train loss 0.3580 | train acc 0.8745 | val loss 0.4468 | val acc 0.8416 | time 2.1s
    Epoch 023/30 | train loss 0.3416 | train acc 0.8806 | val loss 0.4375 | val acc 0.8482 | time 2.0s
    Epoch 024/30 | train loss 0.3281 | train acc 0.8863 | val loss 0.4394 | val acc 0.8472 | time 2.1s
    Epoch 025/30 | train loss 0.3156 | train acc 0.8914 | val loss 0.4243 | val acc 0.8544 | time 2.0s
    Epoch 026/30 | train loss 0.3073 | train acc 0.8928 | val loss 0.4237 | val acc 0.8550 | time 2.1s
    Epoch 027/30 | train loss 0.2998 | train acc 0.8959 | val loss 0.4216 | val acc 0.8550 | time 2.1s
    Epoch 028/30 | train loss 0.2932 | train acc 0.8994 | val loss 0.4252 | val acc 0.8542 | time 2.0s
    Epoch 029/30 | train loss 0.2903 | train acc 0.9003 | val loss 0.4172 | val acc 0.8584 | time 2.0s
    Epoch 030/30 | train loss 0.2876 | train acc 0.9013 | val loss 0.4159 | val acc 0.8580 | time 2.0s
    Final test accuracy for alexnet_dropout00: 0.8517
    Results saved to: results/alexnet_dropout00
    ==========================================================================================
    Run: alexnet_dropout03
    Device: cuda
    Trainable parameters: 3,192,458
    ==========================================================================================
    Epoch 001/30 | train loss 2.1155 | train acc 0.2013 | val loss 1.8041 | val acc 0.3190 | time 2.1s
    Epoch 002/30 | train loss 1.6956 | train acc 0.3647 | val loss 1.4973 | val acc 0.4514 | time 2.0s
    Epoch 003/30 | train loss 1.4736 | train acc 0.4562 | val loss 1.3459 | val acc 0.5078 | time 2.0s
    Epoch 004/30 | train loss 1.3259 | train acc 0.5163 | val loss 1.1465 | val acc 0.5860 | time 2.0s
    Epoch 005/30 | train loss 1.1838 | train acc 0.5727 | val loss 1.0300 | val acc 0.6220 | time 2.0s
    Epoch 006/30 | train loss 1.0546 | train acc 0.6212 | val loss 0.9140 | val acc 0.6730 | time 2.0s
    Epoch 007/30 | train loss 0.9519 | train acc 0.6641 | val loss 0.8071 | val acc 0.7070 | time 2.1s
    Epoch 008/30 | train loss 0.8757 | train acc 0.6934 | val loss 0.7440 | val acc 0.7316 | time 2.1s
    Epoch 009/30 | train loss 0.8060 | train acc 0.7164 | val loss 0.7335 | val acc 0.7336 | time 2.1s
    Epoch 010/30 | train loss 0.7599 | train acc 0.7356 | val loss 0.7053 | val acc 0.7438 | time 2.0s
    Epoch 011/30 | train loss 0.7178 | train acc 0.7506 | val loss 0.6388 | val acc 0.7696 | time 2.2s
    Epoch 012/30 | train loss 0.6685 | train acc 0.7686 | val loss 0.5908 | val acc 0.7944 | time 2.0s
    Epoch 013/30 | train loss 0.6337 | train acc 0.7803 | val loss 0.5683 | val acc 0.7992 | time 2.1s
    Epoch 014/30 | train loss 0.6014 | train acc 0.7916 | val loss 0.5477 | val acc 0.8074 | time 2.1s
    Epoch 015/30 | train loss 0.5717 | train acc 0.8006 | val loss 0.5306 | val acc 0.8160 | time 2.0s
    Epoch 016/30 | train loss 0.5454 | train acc 0.8104 | val loss 0.5327 | val acc 0.8114 | time 2.1s
    Epoch 017/30 | train loss 0.5224 | train acc 0.8184 | val loss 0.5194 | val acc 0.8166 | time 2.1s
    Epoch 018/30 | train loss 0.5002 | train acc 0.8286 | val loss 0.4977 | val acc 0.8268 | time 2.0s
    Epoch 019/30 | train loss 0.4768 | train acc 0.8354 | val loss 0.5002 | val acc 0.8234 | time 2.0s
    Epoch 020/30 | train loss 0.4563 | train acc 0.8428 | val loss 0.4664 | val acc 0.8382 | time 2.1s
    Epoch 021/30 | train loss 0.4423 | train acc 0.8461 | val loss 0.4648 | val acc 0.8406 | time 2.1s
    Epoch 022/30 | train loss 0.4210 | train acc 0.8532 | val loss 0.4623 | val acc 0.8396 | time 2.1s
    Epoch 023/30 | train loss 0.4050 | train acc 0.8579 | val loss 0.4533 | val acc 0.8406 | time 2.0s
    Epoch 024/30 | train loss 0.3949 | train acc 0.8628 | val loss 0.4523 | val acc 0.8434 | time 2.0s
    Epoch 025/30 | train loss 0.3806 | train acc 0.8686 | val loss 0.4366 | val acc 0.8512 | time 2.0s
    Epoch 026/30 | train loss 0.3700 | train acc 0.8731 | val loss 0.4297 | val acc 0.8514 | time 2.0s
    Epoch 027/30 | train loss 0.3637 | train acc 0.8747 | val loss 0.4277 | val acc 0.8530 | time 2.0s
    Epoch 028/30 | train loss 0.3599 | train acc 0.8745 | val loss 0.4282 | val acc 0.8512 | time 2.0s
    Epoch 029/30 | train loss 0.3574 | train acc 0.8766 | val loss 0.4248 | val acc 0.8554 | time 2.0s
    Epoch 030/30 | train loss 0.3552 | train acc 0.8793 | val loss 0.4244 | val acc 0.8554 | time 2.1s
    Final test accuracy for alexnet_dropout03: 0.8500
    Results saved to: results/alexnet_dropout03
    ==========================================================================================
    Run: alexnet_dropout05
    Device: cuda
    Trainable parameters: 3,192,458
    ==========================================================================================
    Epoch 001/30 | train loss 2.1425 | train acc 0.1841 | val loss 1.8310 | val acc 0.2926 | time 2.1s
    Epoch 002/30 | train loss 1.7521 | train acc 0.3356 | val loss 1.5668 | val acc 0.4266 | time 2.1s
    Epoch 003/30 | train loss 1.5421 | train acc 0.4227 | val loss 1.3555 | val acc 0.5020 | time 2.0s
    Epoch 004/30 | train loss 1.3998 | train acc 0.4882 | val loss 1.1842 | val acc 0.5746 | time 2.1s
    Epoch 005/30 | train loss 1.2524 | train acc 0.5444 | val loss 1.0567 | val acc 0.6218 | time 2.2s
    Epoch 006/30 | train loss 1.1325 | train acc 0.5941 | val loss 0.9710 | val acc 0.6578 | time 2.0s
    Epoch 007/30 | train loss 1.0262 | train acc 0.6361 | val loss 0.9063 | val acc 0.6746 | time 2.0s
    Epoch 008/30 | train loss 0.9507 | train acc 0.6667 | val loss 0.7978 | val acc 0.7136 | time 2.1s
    Epoch 009/30 | train loss 0.8711 | train acc 0.6946 | val loss 0.7378 | val acc 0.7362 | time 2.1s
    Epoch 010/30 | train loss 0.8246 | train acc 0.7144 | val loss 0.7159 | val acc 0.7394 | time 2.0s
    Epoch 011/30 | train loss 0.7732 | train acc 0.7340 | val loss 0.6816 | val acc 0.7518 | time 2.1s
    Epoch 012/30 | train loss 0.7292 | train acc 0.7490 | val loss 0.6391 | val acc 0.7774 | time 2.1s
    Epoch 013/30 | train loss 0.6849 | train acc 0.7660 | val loss 0.6175 | val acc 0.7846 | time 2.0s
    Epoch 014/30 | train loss 0.6565 | train acc 0.7757 | val loss 0.5680 | val acc 0.7988 | time 2.1s
    Epoch 015/30 | train loss 0.6189 | train acc 0.7902 | val loss 0.5707 | val acc 0.7998 | time 2.1s
    Epoch 016/30 | train loss 0.5905 | train acc 0.8004 | val loss 0.5512 | val acc 0.8082 | time 2.0s
    Epoch 017/30 | train loss 0.5707 | train acc 0.8052 | val loss 0.5323 | val acc 0.8090 | time 2.0s
    Epoch 018/30 | train loss 0.5425 | train acc 0.8149 | val loss 0.5189 | val acc 0.8192 | time 2.1s
    Epoch 019/30 | train loss 0.5185 | train acc 0.8240 | val loss 0.5123 | val acc 0.8214 | time 2.1s
    Epoch 020/30 | train loss 0.4951 | train acc 0.8313 | val loss 0.4868 | val acc 0.8314 | time 2.0s
    Epoch 021/30 | train loss 0.4797 | train acc 0.8352 | val loss 0.4789 | val acc 0.8328 | time 2.1s
    Epoch 022/30 | train loss 0.4531 | train acc 0.8462 | val loss 0.4792 | val acc 0.8306 | time 2.1s
    Epoch 023/30 | train loss 0.4444 | train acc 0.8497 | val loss 0.4616 | val acc 0.8340 | time 2.0s
    Epoch 024/30 | train loss 0.4330 | train acc 0.8528 | val loss 0.4700 | val acc 0.8332 | time 2.0s
    Epoch 025/30 | train loss 0.4176 | train acc 0.8593 | val loss 0.4508 | val acc 0.8424 | time 2.0s
    Epoch 026/30 | train loss 0.4093 | train acc 0.8618 | val loss 0.4414 | val acc 0.8444 | time 2.1s
    Epoch 027/30 | train loss 0.4018 | train acc 0.8644 | val loss 0.4398 | val acc 0.8468 | time 2.0s
    Epoch 028/30 | train loss 0.3954 | train acc 0.8665 | val loss 0.4406 | val acc 0.8452 | time 2.1s
    Epoch 029/30 | train loss 0.3934 | train acc 0.8669 | val loss 0.4369 | val acc 0.8486 | time 2.1s
    Epoch 030/30 | train loss 0.3938 | train acc 0.8680 | val loss 0.4344 | val acc 0.8464 | time 2.1s
    Final test accuracy for alexnet_dropout05: 0.8473
    Results saved to: results/alexnet_dropout05
    ==========================================================================================
    Run: vgg_dropout00
    Device: cuda
    Trainable parameters: 3,586,698
    ==========================================================================================
    Epoch 001/30 | train loss 2.3029 | train acc 0.1001 | val loss 2.3031 | val acc 0.1000 | time 2.0s
    Epoch 002/30 | train loss 2.3029 | train acc 0.0980 | val loss 2.3033 | val acc 0.0942 | time 2.0s
    Epoch 003/30 | train loss 2.3029 | train acc 0.0991 | val loss 2.3032 | val acc 0.0942 | time 2.0s
    Epoch 004/30 | train loss 2.3029 | train acc 0.1001 | val loss 2.3030 | val acc 0.0976 | time 2.0s
    Epoch 005/30 | train loss 2.3029 | train acc 0.0978 | val loss 2.3027 | val acc 0.1028 | time 2.0s
    Epoch 006/30 | train loss 2.3029 | train acc 0.0958 | val loss 2.3028 | val acc 0.0942 | time 2.1s
    Epoch 007/30 | train loss 2.3029 | train acc 0.1003 | val loss 2.3025 | val acc 0.1028 | time 2.1s
    Epoch 008/30 | train loss 2.3028 | train acc 0.1002 | val loss 2.3027 | val acc 0.1000 | time 2.0s
    Epoch 009/30 | train loss 2.3028 | train acc 0.0992 | val loss 2.3026 | val acc 0.1014 | time 2.0s
    Epoch 010/30 | train loss 2.3028 | train acc 0.1002 | val loss 2.3028 | val acc 0.1000 | time 2.0s
    Epoch 011/30 | train loss 2.3028 | train acc 0.1002 | val loss 2.3029 | val acc 0.0942 | time 2.0s
    Epoch 012/30 | train loss 2.3028 | train acc 0.0970 | val loss 2.3030 | val acc 0.0942 | time 2.0s
    Epoch 013/30 | train loss 2.3027 | train acc 0.0994 | val loss 2.3026 | val acc 0.1008 | time 2.1s
    Epoch 014/30 | train loss 2.3028 | train acc 0.0972 | val loss 2.3027 | val acc 0.1028 | time 2.0s
    Epoch 015/30 | train loss 2.3027 | train acc 0.1006 | val loss 2.3028 | val acc 0.0976 | time 2.0s
    Epoch 016/30 | train loss 2.3028 | train acc 0.0981 | val loss 2.3027 | val acc 0.0942 | time 2.0s
    Epoch 017/30 | train loss 2.3027 | train acc 0.0982 | val loss 2.3026 | val acc 0.1008 | time 2.1s
    Epoch 018/30 | train loss 2.3027 | train acc 0.0971 | val loss 2.3027 | val acc 0.0976 | time 2.0s
    Epoch 019/30 | train loss 2.3027 | train acc 0.0964 | val loss 2.3027 | val acc 0.0942 | time 2.0s
    Epoch 020/30 | train loss 2.3027 | train acc 0.0991 | val loss 2.3027 | val acc 0.0942 | time 2.0s
    Epoch 021/30 | train loss 2.3027 | train acc 0.0982 | val loss 2.3027 | val acc 0.0942 | time 2.1s
    Epoch 022/30 | train loss 2.3026 | train acc 0.0988 | val loss 2.3027 | val acc 0.0942 | time 2.0s
    Epoch 023/30 | train loss 2.3026 | train acc 0.0990 | val loss 2.3027 | val acc 0.0942 | time 2.0s
    Epoch 024/30 | train loss 2.3026 | train acc 0.0997 | val loss 2.3027 | val acc 0.0942 | time 2.0s
    Epoch 025/30 | train loss 2.3026 | train acc 0.0989 | val loss 2.3027 | val acc 0.0942 | time 2.0s
    Epoch 026/30 | train loss 2.3026 | train acc 0.1006 | val loss 2.3027 | val acc 0.0942 | time 2.1s
    Epoch 027/30 | train loss 2.3026 | train acc 0.1006 | val loss 2.3027 | val acc 0.0942 | time 2.0s
    Epoch 028/30 | train loss 2.3026 | train acc 0.0993 | val loss 2.3027 | val acc 0.0942 | time 2.0s
    Epoch 029/30 | train loss 2.3026 | train acc 0.1006 | val loss 2.3027 | val acc 0.0942 | time 2.0s
    Epoch 030/30 | train loss 2.3026 | train acc 0.1006 | val loss 2.3027 | val acc 0.0942 | time 2.0s
    Final test accuracy for vgg_dropout00: 0.1000
    Results saved to: results/vgg_dropout00
    ==========================================================================================
    Run: vgg_dropout03
    Device: cuda
    Trainable parameters: 3,586,698
    ==========================================================================================
    Epoch 001/30 | train loss 2.3029 | train acc 0.0995 | val loss 2.3031 | val acc 0.0942 | time 2.1s
    Epoch 002/30 | train loss 2.3029 | train acc 0.0972 | val loss 2.3032 | val acc 0.0942 | time 2.1s
    Epoch 003/30 | train loss 2.3029 | train acc 0.0992 | val loss 2.3031 | val acc 0.0942 | time 2.0s
    Epoch 004/30 | train loss 2.3029 | train acc 0.0996 | val loss 2.3030 | val acc 0.0976 | time 2.0s
    Epoch 005/30 | train loss 2.3029 | train acc 0.0988 | val loss 2.3027 | val acc 0.1028 | time 2.0s
    Epoch 006/30 | train loss 2.3029 | train acc 0.0974 | val loss 2.3028 | val acc 0.0942 | time 2.1s
    Epoch 007/30 | train loss 2.3029 | train acc 0.0991 | val loss 2.3025 | val acc 0.1028 | time 2.1s
    Epoch 008/30 | train loss 2.3028 | train acc 0.0995 | val loss 2.3027 | val acc 0.1000 | time 2.1s
    Epoch 009/30 | train loss 2.3028 | train acc 0.0975 | val loss 2.3026 | val acc 0.1014 | time 2.2s
    Epoch 010/30 | train loss 2.3028 | train acc 0.0992 | val loss 2.3028 | val acc 0.1000 | time 2.1s
    Epoch 011/30 | train loss 2.3028 | train acc 0.1015 | val loss 2.3029 | val acc 0.0942 | time 2.1s
    Epoch 012/30 | train loss 2.3028 | train acc 0.0985 | val loss 2.3030 | val acc 0.0942 | time 2.1s
    Epoch 013/30 | train loss 2.3027 | train acc 0.0995 | val loss 2.3027 | val acc 0.1008 | time 2.1s
    Epoch 014/30 | train loss 2.3028 | train acc 0.0977 | val loss 2.3027 | val acc 0.1028 | time 2.0s
    Epoch 015/30 | train loss 2.3028 | train acc 0.0992 | val loss 2.3028 | val acc 0.0976 | time 2.1s
    Epoch 016/30 | train loss 2.3027 | train acc 0.0998 | val loss 2.3027 | val acc 0.0942 | time 2.1s
    Epoch 017/30 | train loss 2.3027 | train acc 0.1000 | val loss 2.3026 | val acc 0.1008 | time 2.1s
    Epoch 018/30 | train loss 2.3027 | train acc 0.0966 | val loss 2.3027 | val acc 0.0976 | time 2.1s
    Epoch 019/30 | train loss 2.3027 | train acc 0.0976 | val loss 2.3027 | val acc 0.0942 | time 2.1s
    Epoch 020/30 | train loss 2.3027 | train acc 0.0978 | val loss 2.3028 | val acc 0.0942 | time 2.1s
    Epoch 021/30 | train loss 2.3027 | train acc 0.0973 | val loss 2.3027 | val acc 0.0942 | time 2.0s
    Epoch 022/30 | train loss 2.3027 | train acc 0.0992 | val loss 2.3027 | val acc 0.0942 | time 2.1s
    Epoch 023/30 | train loss 2.3026 | train acc 0.0970 | val loss 2.3027 | val acc 0.0942 | time 2.0s
    Epoch 024/30 | train loss 2.3026 | train acc 0.0987 | val loss 2.3027 | val acc 0.0942 | time 2.0s
    Epoch 025/30 | train loss 2.3026 | train acc 0.0998 | val loss 2.3027 | val acc 0.0942 | time 2.0s
    Epoch 026/30 | train loss 2.3026 | train acc 0.0994 | val loss 2.3027 | val acc 0.0942 | time 2.1s
    Epoch 027/30 | train loss 2.3026 | train acc 0.1005 | val loss 2.3027 | val acc 0.0942 | time 2.0s
    Epoch 028/30 | train loss 2.3026 | train acc 0.1006 | val loss 2.3027 | val acc 0.0942 | time 2.1s
    Epoch 029/30 | train loss 2.3026 | train acc 0.1026 | val loss 2.3027 | val acc 0.0942 | time 2.0s
    Epoch 030/30 | train loss 2.3026 | train acc 0.1002 | val loss 2.3027 | val acc 0.0942 | time 2.1s
    Final test accuracy for vgg_dropout03: 0.1000
    Results saved to: results/vgg_dropout03
    ==========================================================================================
    Run: vgg_dropout05
    Device: cuda
    Trainable parameters: 3,586,698
    ==========================================================================================
    Epoch 001/30 | train loss 2.3030 | train acc 0.1005 | val loss 2.3031 | val acc 0.1000 | time 2.1s
    Epoch 002/30 | train loss 2.3030 | train acc 0.0982 | val loss 2.3032 | val acc 0.0942 | time 2.0s
    Epoch 003/30 | train loss 2.3030 | train acc 0.0993 | val loss 2.3032 | val acc 0.0942 | time 2.0s
    Epoch 004/30 | train loss 2.3029 | train acc 0.0987 | val loss 2.3030 | val acc 0.0976 | time 2.1s
    Epoch 005/30 | train loss 2.3028 | train acc 0.0985 | val loss 2.3027 | val acc 0.1028 | time 2.1s
    Epoch 006/30 | train loss 2.3029 | train acc 0.0981 | val loss 2.3028 | val acc 0.0942 | time 2.1s
    Epoch 007/30 | train loss 2.3029 | train acc 0.0989 | val loss 2.3025 | val acc 0.1028 | time 2.1s
    Epoch 008/30 | train loss 2.3028 | train acc 0.1005 | val loss 2.3028 | val acc 0.1000 | time 2.0s
    Epoch 009/30 | train loss 2.3028 | train acc 0.0973 | val loss 2.3026 | val acc 0.1014 | time 2.1s
    Epoch 010/30 | train loss 2.3028 | train acc 0.1009 | val loss 2.3028 | val acc 0.1000 | time 2.0s
    Epoch 011/30 | train loss 2.3028 | train acc 0.0993 | val loss 2.3029 | val acc 0.0942 | time 2.1s
    Epoch 012/30 | train loss 2.3028 | train acc 0.0979 | val loss 2.3030 | val acc 0.0942 | time 2.0s
    Epoch 013/30 | train loss 2.3028 | train acc 0.0981 | val loss 2.3027 | val acc 0.1008 | time 2.1s
    Epoch 014/30 | train loss 2.3028 | train acc 0.0978 | val loss 2.3027 | val acc 0.1028 | time 2.0s
    Epoch 015/30 | train loss 2.3028 | train acc 0.0990 | val loss 2.3028 | val acc 0.0976 | time 2.0s
    Epoch 016/30 | train loss 2.3028 | train acc 0.0975 | val loss 2.3027 | val acc 0.0942 | time 2.0s
    Epoch 017/30 | train loss 2.3027 | train acc 0.1004 | val loss 2.3026 | val acc 0.1008 | time 2.1s
    Epoch 018/30 | train loss 2.3027 | train acc 0.0987 | val loss 2.3027 | val acc 0.0976 | time 2.1s
    Epoch 019/30 | train loss 2.3027 | train acc 0.0979 | val loss 2.3027 | val acc 0.0942 | time 2.1s
    Epoch 020/30 | train loss 2.3027 | train acc 0.0981 | val loss 2.3028 | val acc 0.0942 | time 2.1s
    Epoch 021/30 | train loss 2.3027 | train acc 0.0983 | val loss 2.3027 | val acc 0.0942 | time 2.0s
    Epoch 022/30 | train loss 2.3026 | train acc 0.0989 | val loss 2.3027 | val acc 0.0942 | time 2.1s
    Epoch 023/30 | train loss 2.3026 | train acc 0.0992 | val loss 2.3028 | val acc 0.0942 | time 2.0s
    Epoch 024/30 | train loss 2.3026 | train acc 0.1000 | val loss 2.3027 | val acc 0.0942 | time 2.0s
    Epoch 025/30 | train loss 2.3026 | train acc 0.1017 | val loss 2.3027 | val acc 0.0942 | time 2.0s
    Epoch 026/30 | train loss 2.3026 | train acc 0.1012 | val loss 2.3027 | val acc 0.0942 | time 2.0s
    Epoch 027/30 | train loss 2.3026 | train acc 0.1004 | val loss 2.3027 | val acc 0.0942 | time 2.0s
    Epoch 028/30 | train loss 2.3026 | train acc 0.1021 | val loss 2.3027 | val acc 0.0942 | time 2.0s
    Epoch 029/30 | train loss 2.3026 | train acc 0.1025 | val loss 2.3027 | val acc 0.0942 | time 2.1s
    Epoch 030/30 | train loss 2.3026 | train acc 0.1015 | val loss 2.3027 | val acc 0.0942 | time 2.0s
    Final test accuracy for vgg_dropout05: 0.1000
    Results saved to: results/vgg_dropout05
    ==========================================================================================
    Run: resnet11_dropout00
    Device: cuda
    Trainable parameters: 4,903,242
    ==========================================================================================
    Epoch 001/50 | train loss 1.5969 | train acc 0.4106 | val loss 1.3631 | val acc 0.5090 | time 2.2s
    Epoch 002/50 | train loss 1.1195 | train acc 0.5977 | val loss 1.2188 | val acc 0.5808 | time 2.0s
    Epoch 003/50 | train loss 0.8815 | train acc 0.6887 | val loss 0.9988 | val acc 0.6552 | time 2.0s
    Epoch 004/50 | train loss 0.7325 | train acc 0.7430 | val loss 0.9537 | val acc 0.6740 | time 2.0s
    Epoch 005/50 | train loss 0.6531 | train acc 0.7726 | val loss 0.7014 | val acc 0.7558 | time 2.0s
    Epoch 006/50 | train loss 0.5910 | train acc 0.7958 | val loss 0.7409 | val acc 0.7508 | time 2.0s
    Epoch 007/50 | train loss 0.5559 | train acc 0.8090 | val loss 0.6283 | val acc 0.7844 | time 2.0s
    Epoch 008/50 | train loss 0.5195 | train acc 0.8204 | val loss 0.6271 | val acc 0.7860 | time 2.0s
    Epoch 009/50 | train loss 0.4954 | train acc 0.8290 | val loss 0.5901 | val acc 0.7942 | time 2.0s
    Epoch 010/50 | train loss 0.4732 | train acc 0.8378 | val loss 0.6313 | val acc 0.7922 | time 2.0s
    Epoch 011/50 | train loss 0.4563 | train acc 0.8427 | val loss 0.6068 | val acc 0.7902 | time 2.0s
    Epoch 012/50 | train loss 0.4434 | train acc 0.8470 | val loss 0.6881 | val acc 0.7740 | time 2.0s
    Epoch 013/50 | train loss 0.4168 | train acc 0.8564 | val loss 0.4848 | val acc 0.8346 | time 2.0s
    Epoch 014/50 | train loss 0.4104 | train acc 0.8588 | val loss 0.6058 | val acc 0.8036 | time 2.0s
    Epoch 015/50 | train loss 0.3945 | train acc 0.8640 | val loss 0.6244 | val acc 0.7984 | time 2.0s
    Epoch 016/50 | train loss 0.3719 | train acc 0.8722 | val loss 0.7807 | val acc 0.7778 | time 2.0s
    Epoch 017/50 | train loss 0.3695 | train acc 0.8717 | val loss 0.6525 | val acc 0.7888 | time 2.1s
    Epoch 018/50 | train loss 0.3521 | train acc 0.8796 | val loss 0.4913 | val acc 0.8384 | time 2.0s
    Epoch 019/50 | train loss 0.3391 | train acc 0.8831 | val loss 0.6172 | val acc 0.8012 | time 2.0s
    Epoch 020/50 | train loss 0.3324 | train acc 0.8848 | val loss 0.4261 | val acc 0.8534 | time 2.0s
    Epoch 021/50 | train loss 0.3227 | train acc 0.8881 | val loss 0.6580 | val acc 0.7964 | time 2.0s
    Epoch 022/50 | train loss 0.3085 | train acc 0.8939 | val loss 0.4751 | val acc 0.8454 | time 2.0s
    Epoch 023/50 | train loss 0.2933 | train acc 0.9010 | val loss 0.3962 | val acc 0.8674 | time 2.0s
    Epoch 024/50 | train loss 0.2884 | train acc 0.9024 | val loss 0.4371 | val acc 0.8560 | time 2.0s
    Epoch 025/50 | train loss 0.2690 | train acc 0.9070 | val loss 0.3994 | val acc 0.8710 | time 2.0s
    Epoch 026/50 | train loss 0.2615 | train acc 0.9110 | val loss 0.3787 | val acc 0.8728 | time 2.0s
    Epoch 027/50 | train loss 0.2471 | train acc 0.9151 | val loss 0.4264 | val acc 0.8582 | time 2.0s
    Epoch 028/50 | train loss 0.2302 | train acc 0.9215 | val loss 0.3702 | val acc 0.8726 | time 2.0s
    Epoch 029/50 | train loss 0.2250 | train acc 0.9220 | val loss 0.4184 | val acc 0.8678 | time 2.0s
    Epoch 030/50 | train loss 0.2091 | train acc 0.9274 | val loss 0.4003 | val acc 0.8762 | time 2.1s
    Epoch 031/50 | train loss 0.1925 | train acc 0.9343 | val loss 0.3832 | val acc 0.8758 | time 2.0s
    Epoch 032/50 | train loss 0.1831 | train acc 0.9372 | val loss 0.3340 | val acc 0.8910 | time 2.0s
    Epoch 033/50 | train loss 0.1632 | train acc 0.9443 | val loss 0.3732 | val acc 0.8828 | time 2.0s
    Epoch 034/50 | train loss 0.1501 | train acc 0.9485 | val loss 0.3123 | val acc 0.8994 | time 2.0s
    Epoch 035/50 | train loss 0.1366 | train acc 0.9535 | val loss 0.3240 | val acc 0.8968 | time 2.0s
    Epoch 036/50 | train loss 0.1206 | train acc 0.9590 | val loss 0.2973 | val acc 0.9054 | time 2.0s
    Epoch 037/50 | train loss 0.1057 | train acc 0.9655 | val loss 0.2802 | val acc 0.9158 | time 2.0s
    Epoch 038/50 | train loss 0.0902 | train acc 0.9698 | val loss 0.2798 | val acc 0.9116 | time 2.0s
    Epoch 039/50 | train loss 0.0746 | train acc 0.9756 | val loss 0.2756 | val acc 0.9168 | time 2.0s
    Epoch 040/50 | train loss 0.0634 | train acc 0.9800 | val loss 0.2551 | val acc 0.9222 | time 2.0s
    Epoch 041/50 | train loss 0.0508 | train acc 0.9848 | val loss 0.2517 | val acc 0.9232 | time 2.0s
    Epoch 042/50 | train loss 0.0391 | train acc 0.9895 | val loss 0.2348 | val acc 0.9326 | time 2.0s
    Epoch 043/50 | train loss 0.0345 | train acc 0.9910 | val loss 0.2380 | val acc 0.9286 | time 2.0s
    Epoch 044/50 | train loss 0.0282 | train acc 0.9931 | val loss 0.2222 | val acc 0.9354 | time 2.0s
    Epoch 045/50 | train loss 0.0255 | train acc 0.9942 | val loss 0.2181 | val acc 0.9356 | time 2.0s
    Epoch 046/50 | train loss 0.0213 | train acc 0.9956 | val loss 0.2153 | val acc 0.9376 | time 2.0s
    Epoch 047/50 | train loss 0.0191 | train acc 0.9962 | val loss 0.2150 | val acc 0.9376 | time 2.0s
    Epoch 048/50 | train loss 0.0182 | train acc 0.9965 | val loss 0.2139 | val acc 0.9396 | time 2.0s
    Epoch 049/50 | train loss 0.0178 | train acc 0.9968 | val loss 0.2128 | val acc 0.9390 | time 2.0s
    Epoch 050/50 | train loss 0.0174 | train acc 0.9969 | val loss 0.2132 | val acc 0.9388 | time 2.0s
    Final test accuracy for resnet11_dropout00: 0.9280
    Results saved to: results/resnet11_dropout00
    ==========================================================================================
    Run: resnet11_dropout03
    Device: cuda
    Trainable parameters: 4,903,242
    ==========================================================================================
    Epoch 001/50 | train loss 1.6474 | train acc 0.3958 | val loss 1.6610 | val acc 0.4318 | time 2.0s
    Epoch 002/50 | train loss 1.1797 | train acc 0.5735 | val loss 1.1259 | val acc 0.5978 | time 2.0s
    Epoch 003/50 | train loss 0.9620 | train acc 0.6596 | val loss 1.0348 | val acc 0.6288 | time 2.0s
    Epoch 004/50 | train loss 0.7974 | train acc 0.7212 | val loss 0.9356 | val acc 0.6810 | time 2.0s
    Epoch 005/50 | train loss 0.7019 | train acc 0.7574 | val loss 0.7531 | val acc 0.7386 | time 2.0s
    Epoch 006/50 | train loss 0.6285 | train acc 0.7831 | val loss 0.6981 | val acc 0.7636 | time 2.0s
    Epoch 007/50 | train loss 0.5866 | train acc 0.7963 | val loss 0.7871 | val acc 0.7350 | time 2.1s
    Epoch 008/50 | train loss 0.5451 | train acc 0.8143 | val loss 0.5774 | val acc 0.7972 | time 2.0s
    Epoch 009/50 | train loss 0.5202 | train acc 0.8210 | val loss 0.6764 | val acc 0.7764 | time 2.0s
    Epoch 010/50 | train loss 0.4971 | train acc 0.8300 | val loss 0.6586 | val acc 0.7850 | time 2.0s
    Epoch 011/50 | train loss 0.4729 | train acc 0.8392 | val loss 0.5933 | val acc 0.7900 | time 2.0s
    Epoch 012/50 | train loss 0.4584 | train acc 0.8438 | val loss 0.5258 | val acc 0.8226 | time 2.1s
    Epoch 013/50 | train loss 0.4412 | train acc 0.8484 | val loss 0.5504 | val acc 0.8122 | time 2.0s
    Epoch 014/50 | train loss 0.4341 | train acc 0.8517 | val loss 0.5210 | val acc 0.8272 | time 2.0s
    Epoch 015/50 | train loss 0.4094 | train acc 0.8594 | val loss 0.6377 | val acc 0.7932 | time 2.0s
    Epoch 016/50 | train loss 0.3923 | train acc 0.8660 | val loss 0.5075 | val acc 0.8254 | time 2.0s
    Epoch 017/50 | train loss 0.3862 | train acc 0.8677 | val loss 0.5772 | val acc 0.8126 | time 2.0s
    Epoch 018/50 | train loss 0.3752 | train acc 0.8718 | val loss 0.6370 | val acc 0.7928 | time 2.0s
    Epoch 019/50 | train loss 0.3592 | train acc 0.8766 | val loss 0.5212 | val acc 0.8316 | time 2.0s
    Epoch 020/50 | train loss 0.3507 | train acc 0.8791 | val loss 0.4980 | val acc 0.8362 | time 2.0s
    Epoch 021/50 | train loss 0.3357 | train acc 0.8846 | val loss 0.4959 | val acc 0.8400 | time 2.0s
    Epoch 022/50 | train loss 0.3257 | train acc 0.8891 | val loss 0.4228 | val acc 0.8592 | time 2.1s
    Epoch 023/50 | train loss 0.3153 | train acc 0.8920 | val loss 0.6294 | val acc 0.8056 | time 2.0s
    Epoch 024/50 | train loss 0.2940 | train acc 0.8997 | val loss 0.4091 | val acc 0.8646 | time 2.0s
    Epoch 025/50 | train loss 0.2914 | train acc 0.9008 | val loss 0.5490 | val acc 0.8250 | time 2.0s
    Epoch 026/50 | train loss 0.2711 | train acc 0.9070 | val loss 0.4595 | val acc 0.8542 | time 2.0s
    Epoch 027/50 | train loss 0.2682 | train acc 0.9066 | val loss 0.4602 | val acc 0.8502 | time 2.0s
    Epoch 028/50 | train loss 0.2481 | train acc 0.9150 | val loss 0.4682 | val acc 0.8502 | time 2.0s
    Epoch 029/50 | train loss 0.2343 | train acc 0.9190 | val loss 0.4461 | val acc 0.8536 | time 2.0s
    Epoch 030/50 | train loss 0.2210 | train acc 0.9236 | val loss 0.3808 | val acc 0.8752 | time 2.0s
    Epoch 031/50 | train loss 0.2125 | train acc 0.9262 | val loss 0.3338 | val acc 0.8832 | time 2.0s
    Epoch 032/50 | train loss 0.1928 | train acc 0.9337 | val loss 0.3471 | val acc 0.8894 | time 2.0s
    Epoch 033/50 | train loss 0.1791 | train acc 0.9389 | val loss 0.3614 | val acc 0.8848 | time 2.0s
    Epoch 034/50 | train loss 0.1577 | train acc 0.9460 | val loss 0.3274 | val acc 0.8912 | time 2.0s
    Epoch 035/50 | train loss 0.1457 | train acc 0.9495 | val loss 0.3202 | val acc 0.8944 | time 2.1s
    Epoch 036/50 | train loss 0.1296 | train acc 0.9561 | val loss 0.2983 | val acc 0.9042 | time 2.0s
    Epoch 037/50 | train loss 0.1160 | train acc 0.9613 | val loss 0.3051 | val acc 0.9046 | time 2.0s
    Epoch 038/50 | train loss 0.1019 | train acc 0.9656 | val loss 0.2953 | val acc 0.9082 | time 2.0s
    Epoch 039/50 | train loss 0.0843 | train acc 0.9724 | val loss 0.2633 | val acc 0.9178 | time 2.0s
    Epoch 040/50 | train loss 0.0737 | train acc 0.9757 | val loss 0.2522 | val acc 0.9206 | time 2.0s
    Epoch 041/50 | train loss 0.0589 | train acc 0.9820 | val loss 0.2505 | val acc 0.9244 | time 2.0s
    Epoch 042/50 | train loss 0.0504 | train acc 0.9854 | val loss 0.2350 | val acc 0.9242 | time 2.0s
    Epoch 043/50 | train loss 0.0410 | train acc 0.9881 | val loss 0.2306 | val acc 0.9292 | time 2.1s
    Epoch 044/50 | train loss 0.0331 | train acc 0.9913 | val loss 0.2290 | val acc 0.9280 | time 2.0s
    Epoch 045/50 | train loss 0.0299 | train acc 0.9926 | val loss 0.2241 | val acc 0.9314 | time 2.0s
    Epoch 046/50 | train loss 0.0255 | train acc 0.9942 | val loss 0.2245 | val acc 0.9330 | time 1.9s
    Epoch 047/50 | train loss 0.0233 | train acc 0.9948 | val loss 0.2214 | val acc 0.9354 | time 2.0s
    Epoch 048/50 | train loss 0.0220 | train acc 0.9954 | val loss 0.2235 | val acc 0.9326 | time 2.0s
    Epoch 049/50 | train loss 0.0208 | train acc 0.9960 | val loss 0.2218 | val acc 0.9336 | time 2.0s
    Epoch 050/50 | train loss 0.0202 | train acc 0.9964 | val loss 0.2216 | val acc 0.9334 | time 2.0s
    Final test accuracy for resnet11_dropout03: 0.9299
    Results saved to: results/resnet11_dropout03
    ==========================================================================================
    Run: resnet11_dropout05
    Device: cuda
    Trainable parameters: 4,903,242
    ==========================================================================================
    Epoch 001/50 | train loss 1.7352 | train acc 0.3641 | val loss 1.4434 | val acc 0.4722 | time 2.0s
    Epoch 002/50 | train loss 1.2466 | train acc 0.5493 | val loss 1.1016 | val acc 0.6010 | time 2.0s
    Epoch 003/50 | train loss 1.0285 | train acc 0.6335 | val loss 0.9936 | val acc 0.6468 | time 2.0s
    Epoch 004/50 | train loss 0.8782 | train acc 0.6909 | val loss 1.0072 | val acc 0.6546 | time 2.0s
    Epoch 005/50 | train loss 0.7669 | train acc 0.7303 | val loss 0.8541 | val acc 0.7156 | time 2.0s
    Epoch 006/50 | train loss 0.6837 | train acc 0.7636 | val loss 0.7629 | val acc 0.7412 | time 2.0s
    Epoch 007/50 | train loss 0.6262 | train acc 0.7853 | val loss 1.0174 | val acc 0.6826 | time 2.0s
    Epoch 008/50 | train loss 0.5840 | train acc 0.7996 | val loss 0.8759 | val acc 0.7076 | time 2.0s
    Epoch 009/50 | train loss 0.5541 | train acc 0.8084 | val loss 0.6936 | val acc 0.7712 | time 2.0s
    Epoch 010/50 | train loss 0.5286 | train acc 0.8215 | val loss 0.5736 | val acc 0.7986 | time 2.0s
    Epoch 011/50 | train loss 0.5013 | train acc 0.8302 | val loss 0.7022 | val acc 0.7502 | time 1.9s
    Epoch 012/50 | train loss 0.4878 | train acc 0.8338 | val loss 0.5019 | val acc 0.8282 | time 2.0s
    Epoch 013/50 | train loss 0.4662 | train acc 0.8423 | val loss 0.5715 | val acc 0.8088 | time 1.9s
    Epoch 014/50 | train loss 0.4535 | train acc 0.8470 | val loss 0.5264 | val acc 0.8258 | time 2.0s
    Epoch 015/50 | train loss 0.4278 | train acc 0.8539 | val loss 0.6665 | val acc 0.7830 | time 2.0s
    Epoch 016/50 | train loss 0.4144 | train acc 0.8595 | val loss 0.6138 | val acc 0.8034 | time 2.0s
    Epoch 017/50 | train loss 0.4072 | train acc 0.8612 | val loss 0.6483 | val acc 0.7954 | time 2.0s
    Epoch 018/50 | train loss 0.3943 | train acc 0.8669 | val loss 0.5197 | val acc 0.8308 | time 2.0s
    Epoch 019/50 | train loss 0.3764 | train acc 0.8732 | val loss 0.6573 | val acc 0.7954 | time 2.0s
    Epoch 020/50 | train loss 0.3675 | train acc 0.8748 | val loss 0.4585 | val acc 0.8440 | time 2.0s
    Epoch 021/50 | train loss 0.3537 | train acc 0.8811 | val loss 0.4298 | val acc 0.8594 | time 2.0s
    Epoch 022/50 | train loss 0.3460 | train acc 0.8825 | val loss 0.4894 | val acc 0.8362 | time 2.0s
    Epoch 023/50 | train loss 0.3270 | train acc 0.8902 | val loss 0.4193 | val acc 0.8624 | time 2.0s
    Epoch 024/50 | train loss 0.3215 | train acc 0.8906 | val loss 0.4835 | val acc 0.8412 | time 2.0s
    Epoch 025/50 | train loss 0.3027 | train acc 0.8969 | val loss 0.4169 | val acc 0.8640 | time 2.0s
    Epoch 026/50 | train loss 0.2957 | train acc 0.8973 | val loss 0.3792 | val acc 0.8716 | time 2.0s
    Epoch 027/50 | train loss 0.2770 | train acc 0.9066 | val loss 0.3665 | val acc 0.8768 | time 2.0s
    Epoch 028/50 | train loss 0.2637 | train acc 0.9096 | val loss 0.5109 | val acc 0.8410 | time 2.1s
    Epoch 029/50 | train loss 0.2501 | train acc 0.9150 | val loss 0.4093 | val acc 0.8692 | time 2.0s
    Epoch 030/50 | train loss 0.2337 | train acc 0.9203 | val loss 0.4167 | val acc 0.8648 | time 2.0s
    Epoch 031/50 | train loss 0.2231 | train acc 0.9227 | val loss 0.3734 | val acc 0.8768 | time 2.0s
    Epoch 032/50 | train loss 0.2087 | train acc 0.9292 | val loss 0.3503 | val acc 0.8850 | time 2.0s
    Epoch 033/50 | train loss 0.1891 | train acc 0.9358 | val loss 0.3037 | val acc 0.8990 | time 2.0s
    Epoch 034/50 | train loss 0.1710 | train acc 0.9418 | val loss 0.3650 | val acc 0.8862 | time 2.0s
    Epoch 035/50 | train loss 0.1600 | train acc 0.9453 | val loss 0.3093 | val acc 0.9004 | time 2.0s
    Epoch 036/50 | train loss 0.1414 | train acc 0.9516 | val loss 0.2875 | val acc 0.9044 | time 2.0s
    Epoch 037/50 | train loss 0.1230 | train acc 0.9578 | val loss 0.2698 | val acc 0.9198 | time 2.0s
    Epoch 038/50 | train loss 0.1081 | train acc 0.9647 | val loss 0.2981 | val acc 0.9108 | time 2.0s
    Epoch 039/50 | train loss 0.0921 | train acc 0.9701 | val loss 0.2848 | val acc 0.9140 | time 2.0s
    Epoch 040/50 | train loss 0.0776 | train acc 0.9749 | val loss 0.2634 | val acc 0.9196 | time 2.0s
    Epoch 041/50 | train loss 0.0647 | train acc 0.9794 | val loss 0.2550 | val acc 0.9214 | time 2.0s
    Epoch 042/50 | train loss 0.0547 | train acc 0.9832 | val loss 0.2320 | val acc 0.9302 | time 2.0s
    Epoch 043/50 | train loss 0.0455 | train acc 0.9868 | val loss 0.2321 | val acc 0.9288 | time 2.0s
    Epoch 044/50 | train loss 0.0376 | train acc 0.9898 | val loss 0.2335 | val acc 0.9292 | time 2.0s
    Epoch 045/50 | train loss 0.0334 | train acc 0.9913 | val loss 0.2247 | val acc 0.9288 | time 1.9s
    Epoch 046/50 | train loss 0.0282 | train acc 0.9935 | val loss 0.2216 | val acc 0.9306 | time 2.0s
    Epoch 047/50 | train loss 0.0256 | train acc 0.9942 | val loss 0.2205 | val acc 0.9324 | time 2.0s
    Epoch 048/50 | train loss 0.0241 | train acc 0.9948 | val loss 0.2179 | val acc 0.9338 | time 2.0s
    Epoch 049/50 | train loss 0.0237 | train acc 0.9951 | val loss 0.2170 | val acc 0.9328 | time 2.0s
    Epoch 050/50 | train loss 0.0227 | train acc 0.9953 | val loss 0.2177 | val acc 0.9344 | time 2.0s
    Final test accuracy for resnet11_dropout05: 0.9298
    Results saved to: results/resnet11_dropout05
    ==========================================================================================
    Run: resnet18_dropout00
    Device: cuda
    Trainable parameters: 11,173,962
    ==========================================================================================
    Epoch 001/50 | train loss 2.0205 | train acc 0.2893 | val loss 1.7684 | val acc 0.3338 | time 3.3s
    Epoch 002/50 | train loss 1.5235 | train acc 0.4392 | val loss 1.4431 | val acc 0.4768 | time 3.3s
    Epoch 003/50 | train loss 1.3147 | train acc 0.5200 | val loss 1.2902 | val acc 0.5322 | time 3.3s
    Epoch 004/50 | train loss 1.1325 | train acc 0.5946 | val loss 1.0925 | val acc 0.6076 | time 3.3s
    Epoch 005/50 | train loss 0.9656 | train acc 0.6579 | val loss 1.0058 | val acc 0.6430 | time 3.3s
    Epoch 006/50 | train loss 0.8564 | train acc 0.6980 | val loss 0.7871 | val acc 0.7286 | time 3.3s
    Epoch 007/50 | train loss 0.7544 | train acc 0.7348 | val loss 0.7852 | val acc 0.7330 | time 3.3s
    Epoch 008/50 | train loss 0.6645 | train acc 0.7697 | val loss 0.6514 | val acc 0.7734 | time 3.3s
    Epoch 009/50 | train loss 0.6044 | train acc 0.7886 | val loss 0.6662 | val acc 0.7748 | time 3.3s
    Epoch 010/50 | train loss 0.5650 | train acc 0.8063 | val loss 0.8532 | val acc 0.7244 | time 3.3s
    Epoch 011/50 | train loss 0.5307 | train acc 0.8149 | val loss 0.6530 | val acc 0.7746 | time 3.3s
    Epoch 012/50 | train loss 0.4973 | train acc 0.8284 | val loss 0.6344 | val acc 0.7854 | time 3.3s
    Epoch 013/50 | train loss 0.4741 | train acc 0.8362 | val loss 0.7411 | val acc 0.7568 | time 3.3s
    Epoch 014/50 | train loss 0.4557 | train acc 0.8420 | val loss 0.7305 | val acc 0.7708 | time 3.3s
    Epoch 015/50 | train loss 0.4271 | train acc 0.8545 | val loss 0.6236 | val acc 0.7964 | time 3.3s
    Epoch 016/50 | train loss 0.4162 | train acc 0.8581 | val loss 0.6746 | val acc 0.7818 | time 3.4s
    Epoch 017/50 | train loss 0.3975 | train acc 0.8627 | val loss 0.5408 | val acc 0.8276 | time 3.3s
    Epoch 018/50 | train loss 0.3789 | train acc 0.8710 | val loss 0.4720 | val acc 0.8360 | time 3.3s
    Epoch 019/50 | train loss 0.3608 | train acc 0.8763 | val loss 0.4892 | val acc 0.8386 | time 3.3s
    Epoch 020/50 | train loss 0.3474 | train acc 0.8802 | val loss 0.5231 | val acc 0.8314 | time 3.3s
    Epoch 021/50 | train loss 0.3335 | train acc 0.8848 | val loss 0.4189 | val acc 0.8554 | time 3.3s
    Epoch 022/50 | train loss 0.3148 | train acc 0.8924 | val loss 0.4073 | val acc 0.8602 | time 3.3s
    Epoch 023/50 | train loss 0.3075 | train acc 0.8954 | val loss 0.4968 | val acc 0.8322 | time 3.3s
    Epoch 024/50 | train loss 0.2873 | train acc 0.9020 | val loss 0.3807 | val acc 0.8702 | time 3.3s
    Epoch 025/50 | train loss 0.2791 | train acc 0.9037 | val loss 0.4833 | val acc 0.8508 | time 3.3s
    Epoch 026/50 | train loss 0.2629 | train acc 0.9093 | val loss 0.3740 | val acc 0.8764 | time 3.3s
    Epoch 027/50 | train loss 0.2475 | train acc 0.9142 | val loss 0.3450 | val acc 0.8838 | time 3.3s
    Epoch 028/50 | train loss 0.2300 | train acc 0.9207 | val loss 0.3413 | val acc 0.8842 | time 3.3s
    Epoch 029/50 | train loss 0.2173 | train acc 0.9238 | val loss 0.3260 | val acc 0.8862 | time 3.3s
    Epoch 030/50 | train loss 0.2034 | train acc 0.9299 | val loss 0.3598 | val acc 0.8766 | time 3.3s
    Epoch 031/50 | train loss 0.1853 | train acc 0.9360 | val loss 0.3204 | val acc 0.8918 | time 3.3s
    Epoch 032/50 | train loss 0.1751 | train acc 0.9395 | val loss 0.3386 | val acc 0.8938 | time 3.3s
    Epoch 033/50 | train loss 0.1564 | train acc 0.9460 | val loss 0.3495 | val acc 0.8796 | time 3.3s
    Epoch 034/50 | train loss 0.1368 | train acc 0.9532 | val loss 0.2981 | val acc 0.9060 | time 3.3s
    Epoch 035/50 | train loss 0.1276 | train acc 0.9564 | val loss 0.2580 | val acc 0.9164 | time 3.3s
    Epoch 036/50 | train loss 0.1026 | train acc 0.9650 | val loss 0.2726 | val acc 0.9178 | time 3.3s
    Epoch 037/50 | train loss 0.0887 | train acc 0.9692 | val loss 0.2523 | val acc 0.9204 | time 3.3s
    Epoch 038/50 | train loss 0.0749 | train acc 0.9744 | val loss 0.2619 | val acc 0.9194 | time 3.3s
    Epoch 039/50 | train loss 0.0589 | train acc 0.9808 | val loss 0.2363 | val acc 0.9294 | time 3.3s
    Epoch 040/50 | train loss 0.0510 | train acc 0.9835 | val loss 0.2369 | val acc 0.9270 | time 3.3s
    Epoch 041/50 | train loss 0.0354 | train acc 0.9893 | val loss 0.2384 | val acc 0.9272 | time 3.3s
    Epoch 042/50 | train loss 0.0251 | train acc 0.9925 | val loss 0.2164 | val acc 0.9378 | time 3.3s
    Epoch 043/50 | train loss 0.0200 | train acc 0.9947 | val loss 0.2119 | val acc 0.9382 | time 3.3s
    Epoch 044/50 | train loss 0.0163 | train acc 0.9958 | val loss 0.2129 | val acc 0.9382 | time 3.3s
    Epoch 045/50 | train loss 0.0130 | train acc 0.9969 | val loss 0.2094 | val acc 0.9420 | time 3.3s
    Epoch 046/50 | train loss 0.0110 | train acc 0.9976 | val loss 0.2033 | val acc 0.9440 | time 3.3s
    Epoch 047/50 | train loss 0.0100 | train acc 0.9979 | val loss 0.2038 | val acc 0.9426 | time 3.3s
    Epoch 048/50 | train loss 0.0087 | train acc 0.9985 | val loss 0.2031 | val acc 0.9428 | time 3.3s
    Epoch 049/50 | train loss 0.0081 | train acc 0.9986 | val loss 0.2006 | val acc 0.9428 | time 3.3s
    Epoch 050/50 | train loss 0.0086 | train acc 0.9984 | val loss 0.2024 | val acc 0.9428 | time 3.3s
    Final test accuracy for resnet18_dropout00: 0.9335
    Results saved to: results/resnet18_dropout00
    ==========================================================================================
    Run: resnet18_dropout03
    Device: cuda
    Trainable parameters: 11,173,962
    ==========================================================================================
    Epoch 001/50 | train loss 2.0218 | train acc 0.2909 | val loss 1.6361 | val acc 0.3710 | time 3.3s
    Epoch 002/50 | train loss 1.5020 | train acc 0.4471 | val loss 1.3966 | val acc 0.4876 | time 3.3s
    Epoch 003/50 | train loss 1.2324 | train acc 0.5573 | val loss 1.1740 | val acc 0.5824 | time 3.3s
    Epoch 004/50 | train loss 1.0329 | train acc 0.6334 | val loss 1.0979 | val acc 0.6246 | time 3.3s
    Epoch 005/50 | train loss 0.8880 | train acc 0.6880 | val loss 0.8503 | val acc 0.7070 | time 3.3s
    Epoch 006/50 | train loss 0.7620 | train acc 0.7346 | val loss 0.9911 | val acc 0.6690 | time 3.3s
    Epoch 007/50 | train loss 0.6774 | train acc 0.7642 | val loss 0.6470 | val acc 0.7810 | time 3.3s
    Epoch 008/50 | train loss 0.6110 | train acc 0.7900 | val loss 0.5970 | val acc 0.7956 | time 3.3s
    Epoch 009/50 | train loss 0.5659 | train acc 0.8069 | val loss 0.6482 | val acc 0.7792 | time 3.3s
    Epoch 010/50 | train loss 0.5407 | train acc 0.8158 | val loss 0.5467 | val acc 0.8142 | time 3.3s
    Epoch 011/50 | train loss 0.5046 | train acc 0.8272 | val loss 0.7481 | val acc 0.7536 | time 3.4s
    Epoch 012/50 | train loss 0.4865 | train acc 0.8338 | val loss 0.6137 | val acc 0.7954 | time 3.3s
    Epoch 013/50 | train loss 0.4670 | train acc 0.8407 | val loss 0.6838 | val acc 0.7808 | time 3.3s
    Epoch 014/50 | train loss 0.4515 | train acc 0.8455 | val loss 0.5716 | val acc 0.8022 | time 3.3s
    Epoch 015/50 | train loss 0.4326 | train acc 0.8537 | val loss 0.6050 | val acc 0.7980 | time 3.4s
    Epoch 016/50 | train loss 0.4169 | train acc 0.8590 | val loss 0.5739 | val acc 0.8044 | time 3.3s
    Epoch 017/50 | train loss 0.3948 | train acc 0.8653 | val loss 0.9795 | val acc 0.7102 | time 3.3s
    Epoch 018/50 | train loss 0.3837 | train acc 0.8685 | val loss 0.4357 | val acc 0.8488 | time 3.3s
    Epoch 019/50 | train loss 0.3656 | train acc 0.8758 | val loss 0.5296 | val acc 0.8240 | time 3.3s
    Epoch 020/50 | train loss 0.3529 | train acc 0.8797 | val loss 0.5663 | val acc 0.8184 | time 3.3s
    Epoch 021/50 | train loss 0.3416 | train acc 0.8826 | val loss 0.4500 | val acc 0.8444 | time 3.3s
    Epoch 022/50 | train loss 0.3224 | train acc 0.8913 | val loss 0.4233 | val acc 0.8520 | time 3.3s
    Epoch 023/50 | train loss 0.3138 | train acc 0.8913 | val loss 0.4221 | val acc 0.8604 | time 3.3s
    Epoch 024/50 | train loss 0.2963 | train acc 0.9002 | val loss 0.4087 | val acc 0.8708 | time 3.3s
    Epoch 025/50 | train loss 0.2806 | train acc 0.9042 | val loss 0.4229 | val acc 0.8638 | time 3.3s
    Epoch 026/50 | train loss 0.2715 | train acc 0.9067 | val loss 0.4732 | val acc 0.8424 | time 3.3s
    Epoch 027/50 | train loss 0.2522 | train acc 0.9131 | val loss 0.3710 | val acc 0.8764 | time 3.3s
    Epoch 028/50 | train loss 0.2420 | train acc 0.9166 | val loss 0.3248 | val acc 0.8902 | time 3.3s
    Epoch 029/50 | train loss 0.2275 | train acc 0.9212 | val loss 0.3422 | val acc 0.8886 | time 3.3s
    Epoch 030/50 | train loss 0.2068 | train acc 0.9295 | val loss 0.3274 | val acc 0.8918 | time 3.3s
    Epoch 031/50 | train loss 0.1887 | train acc 0.9366 | val loss 0.3565 | val acc 0.8880 | time 3.3s
    Epoch 032/50 | train loss 0.1783 | train acc 0.9383 | val loss 0.3397 | val acc 0.8934 | time 3.3s
    Epoch 033/50 | train loss 0.1653 | train acc 0.9434 | val loss 0.3134 | val acc 0.8950 | time 3.3s
    Epoch 034/50 | train loss 0.1480 | train acc 0.9496 | val loss 0.3541 | val acc 0.8912 | time 3.3s
    Epoch 035/50 | train loss 0.1333 | train acc 0.9543 | val loss 0.2792 | val acc 0.9124 | time 3.3s
    Epoch 036/50 | train loss 0.1109 | train acc 0.9627 | val loss 0.3160 | val acc 0.9058 | time 3.4s
    Epoch 037/50 | train loss 0.0983 | train acc 0.9665 | val loss 0.2577 | val acc 0.9228 | time 3.3s
    Epoch 038/50 | train loss 0.0818 | train acc 0.9718 | val loss 0.2695 | val acc 0.9194 | time 3.3s
    Epoch 039/50 | train loss 0.0663 | train acc 0.9784 | val loss 0.2393 | val acc 0.9284 | time 3.3s
    Epoch 040/50 | train loss 0.0547 | train acc 0.9824 | val loss 0.2355 | val acc 0.9256 | time 3.3s
    Epoch 041/50 | train loss 0.0390 | train acc 0.9876 | val loss 0.2447 | val acc 0.9302 | time 3.3s
    Epoch 042/50 | train loss 0.0325 | train acc 0.9901 | val loss 0.2309 | val acc 0.9382 | time 3.3s
    Epoch 043/50 | train loss 0.0243 | train acc 0.9928 | val loss 0.2293 | val acc 0.9376 | time 3.3s
    Epoch 044/50 | train loss 0.0194 | train acc 0.9944 | val loss 0.2216 | val acc 0.9390 | time 3.3s
    Epoch 045/50 | train loss 0.0134 | train acc 0.9968 | val loss 0.2227 | val acc 0.9386 | time 3.3s
    Epoch 046/50 | train loss 0.0122 | train acc 0.9970 | val loss 0.2225 | val acc 0.9412 | time 3.3s
    Epoch 047/50 | train loss 0.0110 | train acc 0.9976 | val loss 0.2198 | val acc 0.9416 | time 3.3s
    Epoch 048/50 | train loss 0.0105 | train acc 0.9976 | val loss 0.2195 | val acc 0.9430 | time 3.3s
    Epoch 049/50 | train loss 0.0091 | train acc 0.9982 | val loss 0.2170 | val acc 0.9428 | time 3.3s
    Epoch 050/50 | train loss 0.0089 | train acc 0.9981 | val loss 0.2198 | val acc 0.9414 | time 3.3s
    Final test accuracy for resnet18_dropout03: 0.9394
    Results saved to: results/resnet18_dropout03
    ==========================================================================================
    Run: resnet18_dropout05
    Device: cuda
    Trainable parameters: 11,173,962
    ==========================================================================================
    Epoch 001/50 | train loss 2.1537 | train acc 0.2589 | val loss 1.6973 | val acc 0.3538 | time 3.3s
    Epoch 002/50 | train loss 1.5703 | train acc 0.4178 | val loss 1.4922 | val acc 0.4560 | time 3.3s
    Epoch 003/50 | train loss 1.3271 | train acc 0.5170 | val loss 1.1060 | val acc 0.6056 | time 3.3s
    Epoch 004/50 | train loss 1.1125 | train acc 0.6074 | val loss 0.9818 | val acc 0.6534 | time 3.3s
    Epoch 005/50 | train loss 0.9195 | train acc 0.6791 | val loss 0.8861 | val acc 0.6908 | time 3.3s
    Epoch 006/50 | train loss 0.7949 | train acc 0.7251 | val loss 0.8104 | val acc 0.7246 | time 3.3s
    Epoch 007/50 | train loss 0.6974 | train acc 0.7583 | val loss 0.6979 | val acc 0.7650 | time 3.3s
    Epoch 008/50 | train loss 0.6340 | train acc 0.7847 | val loss 0.6195 | val acc 0.7844 | time 3.3s
    Epoch 009/50 | train loss 0.5883 | train acc 0.7997 | val loss 0.6519 | val acc 0.7738 | time 3.3s
    Epoch 010/50 | train loss 0.5614 | train acc 0.8092 | val loss 0.6729 | val acc 0.7724 | time 3.3s
    Epoch 011/50 | train loss 0.5326 | train acc 0.8192 | val loss 0.7061 | val acc 0.7690 | time 3.3s
    Epoch 012/50 | train loss 0.5080 | train acc 0.8272 | val loss 0.5205 | val acc 0.8232 | time 3.3s
    Epoch 013/50 | train loss 0.4838 | train acc 0.8388 | val loss 0.5540 | val acc 0.8164 | time 3.3s
    Epoch 014/50 | train loss 0.4697 | train acc 0.8415 | val loss 0.6442 | val acc 0.7770 | time 3.3s
    Epoch 015/50 | train loss 0.4387 | train acc 0.8524 | val loss 0.6060 | val acc 0.7970 | time 3.3s
    Epoch 016/50 | train loss 0.4299 | train acc 0.8558 | val loss 0.5751 | val acc 0.8096 | time 3.3s
    Epoch 017/50 | train loss 0.4101 | train acc 0.8600 | val loss 0.5871 | val acc 0.8080 | time 3.3s
    Epoch 018/50 | train loss 0.3975 | train acc 0.8668 | val loss 0.5363 | val acc 0.8212 | time 3.3s
    Epoch 019/50 | train loss 0.3786 | train acc 0.8724 | val loss 0.5094 | val acc 0.8274 | time 3.3s
    Epoch 020/50 | train loss 0.3678 | train acc 0.8754 | val loss 0.6337 | val acc 0.8026 | time 3.3s
    Epoch 021/50 | train loss 0.3544 | train acc 0.8796 | val loss 0.4798 | val acc 0.8416 | time 3.3s
    Epoch 022/50 | train loss 0.3310 | train acc 0.8897 | val loss 0.3950 | val acc 0.8692 | time 3.3s
    Epoch 023/50 | train loss 0.3255 | train acc 0.8912 | val loss 0.4531 | val acc 0.8462 | time 3.3s
    Epoch 024/50 | train loss 0.3019 | train acc 0.8992 | val loss 0.4929 | val acc 0.8444 | time 3.3s
    Epoch 025/50 | train loss 0.2952 | train acc 0.9010 | val loss 0.4583 | val acc 0.8562 | time 3.3s
    Epoch 026/50 | train loss 0.2808 | train acc 0.9048 | val loss 0.4537 | val acc 0.8524 | time 3.3s
    Epoch 027/50 | train loss 0.2656 | train acc 0.9094 | val loss 0.3923 | val acc 0.8708 | time 3.3s
    Epoch 028/50 | train loss 0.2491 | train acc 0.9171 | val loss 0.3517 | val acc 0.8814 | time 3.3s
    Epoch 029/50 | train loss 0.2335 | train acc 0.9224 | val loss 0.3649 | val acc 0.8760 | time 3.3s
    Epoch 030/50 | train loss 0.2211 | train acc 0.9250 | val loss 0.3257 | val acc 0.8894 | time 3.3s
    Epoch 031/50 | train loss 0.2039 | train acc 0.9315 | val loss 0.3403 | val acc 0.8888 | time 3.3s
    Epoch 032/50 | train loss 0.1902 | train acc 0.9363 | val loss 0.3968 | val acc 0.8792 | time 3.3s
    Epoch 033/50 | train loss 0.1706 | train acc 0.9435 | val loss 0.3151 | val acc 0.8924 | time 3.3s
    Epoch 034/50 | train loss 0.1572 | train acc 0.9468 | val loss 0.3032 | val acc 0.9040 | time 3.3s
    Epoch 035/50 | train loss 0.1382 | train acc 0.9535 | val loss 0.2890 | val acc 0.9084 | time 3.3s
    Epoch 036/50 | train loss 0.1219 | train acc 0.9596 | val loss 0.2875 | val acc 0.9086 | time 3.3s
    Epoch 037/50 | train loss 0.1027 | train acc 0.9655 | val loss 0.2832 | val acc 0.9150 | time 3.4s
    Epoch 038/50 | train loss 0.0897 | train acc 0.9698 | val loss 0.2833 | val acc 0.9174 | time 3.3s
    Epoch 039/50 | train loss 0.0746 | train acc 0.9753 | val loss 0.2674 | val acc 0.9220 | time 3.3s
    Epoch 040/50 | train loss 0.0621 | train acc 0.9793 | val loss 0.2616 | val acc 0.9224 | time 3.3s
    Epoch 041/50 | train loss 0.0450 | train acc 0.9863 | val loss 0.2465 | val acc 0.9304 | time 3.3s
    Epoch 042/50 | train loss 0.0351 | train acc 0.9895 | val loss 0.2397 | val acc 0.9346 | time 3.3s
    Epoch 043/50 | train loss 0.0291 | train acc 0.9912 | val loss 0.2392 | val acc 0.9332 | time 3.3s
    Epoch 044/50 | train loss 0.0225 | train acc 0.9937 | val loss 0.2435 | val acc 0.9362 | time 3.3s
    Epoch 045/50 | train loss 0.0179 | train acc 0.9950 | val loss 0.2344 | val acc 0.9384 | time 3.3s
    Epoch 046/50 | train loss 0.0143 | train acc 0.9966 | val loss 0.2280 | val acc 0.9408 | time 3.3s
    Epoch 047/50 | train loss 0.0133 | train acc 0.9968 | val loss 0.2277 | val acc 0.9422 | time 3.4s
    Epoch 048/50 | train loss 0.0121 | train acc 0.9972 | val loss 0.2282 | val acc 0.9430 | time 3.3s
    Epoch 049/50 | train loss 0.0110 | train acc 0.9973 | val loss 0.2253 | val acc 0.9426 | time 3.3s
    Epoch 050/50 | train loss 0.0103 | train acc 0.9976 | val loss 0.2296 | val acc 0.9410 | time 3.3s
    Final test accuracy for resnet18_dropout05: 0.9393
    Results saved to: results/resnet18_dropout05




  <div id="df-7ba2e420-0917-4140-b5ba-96acbbb2a362" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>run</th>
      <th>model</th>
      <th>dropout</th>
      <th>epochs</th>
      <th>parameters</th>
      <th>best_validation_accuracy</th>
      <th>test_accuracy</th>
      <th>test_loss</th>
      <th>avg_epoch_time_sec</th>
      <th>device</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>alexnet_dropout00</td>
      <td>alexnet</td>
      <td>0.0</td>
      <td>30</td>
      <td>3192458</td>
      <td>0.8584</td>
      <td>0.8517</td>
      <td>0.447245</td>
      <td>2.064835</td>
      <td>cuda</td>
    </tr>
    <tr>
      <th>1</th>
      <td>alexnet_dropout03</td>
      <td>alexnet</td>
      <td>0.3</td>
      <td>30</td>
      <td>3192458</td>
      <td>0.8554</td>
      <td>0.8500</td>
      <td>0.443515</td>
      <td>2.050161</td>
      <td>cuda</td>
    </tr>
    <tr>
      <th>2</th>
      <td>alexnet_dropout05</td>
      <td>alexnet</td>
      <td>0.5</td>
      <td>30</td>
      <td>3192458</td>
      <td>0.8486</td>
      <td>0.8473</td>
      <td>0.452922</td>
      <td>2.056139</td>
      <td>cuda</td>
    </tr>
    <tr>
      <th>3</th>
      <td>vgg_dropout00</td>
      <td>vgg</td>
      <td>0.0</td>
      <td>30</td>
      <td>3586698</td>
      <td>0.1028</td>
      <td>0.1000</td>
      <td>2.302670</td>
      <td>2.038783</td>
      <td>cuda</td>
    </tr>
    <tr>
      <th>4</th>
      <td>vgg_dropout03</td>
      <td>vgg</td>
      <td>0.3</td>
      <td>30</td>
      <td>3586698</td>
      <td>0.1028</td>
      <td>0.1000</td>
      <td>2.302676</td>
      <td>2.063646</td>
      <td>cuda</td>
    </tr>
    <tr>
      <th>5</th>
      <td>vgg_dropout05</td>
      <td>vgg</td>
      <td>0.5</td>
      <td>30</td>
      <td>3586698</td>
      <td>0.1028</td>
      <td>0.1000</td>
      <td>2.302674</td>
      <td>2.047839</td>
      <td>cuda</td>
    </tr>
    <tr>
      <th>6</th>
      <td>resnet11_dropout00</td>
      <td>resnet11</td>
      <td>0.0</td>
      <td>50</td>
      <td>4903242</td>
      <td>0.9396</td>
      <td>0.9280</td>
      <td>0.247211</td>
      <td>1.988500</td>
      <td>cuda</td>
    </tr>
    <tr>
      <th>7</th>
      <td>resnet11_dropout03</td>
      <td>resnet11</td>
      <td>0.3</td>
      <td>50</td>
      <td>4903242</td>
      <td>0.9354</td>
      <td>0.9299</td>
      <td>0.245837</td>
      <td>1.994428</td>
      <td>cuda</td>
    </tr>
    <tr>
      <th>8</th>
      <td>resnet11_dropout05</td>
      <td>resnet11</td>
      <td>0.5</td>
      <td>50</td>
      <td>4903242</td>
      <td>0.9344</td>
      <td>0.9298</td>
      <td>0.233818</td>
      <td>1.981786</td>
      <td>cuda</td>
    </tr>
    <tr>
      <th>9</th>
      <td>resnet18_dropout00</td>
      <td>resnet18</td>
      <td>0.0</td>
      <td>50</td>
      <td>11173962</td>
      <td>0.9440</td>
      <td>0.9335</td>
      <td>0.237261</td>
      <td>3.312322</td>
      <td>cuda</td>
    </tr>
    <tr>
      <th>10</th>
      <td>resnet18_dropout03</td>
      <td>resnet18</td>
      <td>0.3</td>
      <td>50</td>
      <td>11173962</td>
      <td>0.9430</td>
      <td>0.9394</td>
      <td>0.226434</td>
      <td>3.324515</td>
      <td>cuda</td>
    </tr>
    <tr>
      <th>11</th>
      <td>resnet18_dropout05</td>
      <td>resnet18</td>
      <td>0.5</td>
      <td>50</td>
      <td>11173962</td>
      <td>0.9430</td>
      <td>0.9393</td>
      <td>0.236496</td>
      <td>3.322602</td>
      <td>cuda</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-7ba2e420-0917-4140-b5ba-96acbbb2a362')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-7ba2e420-0917-4140-b5ba-96acbbb2a362 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-7ba2e420-0917-4140-b5ba-96acbbb2a362');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


  <div id="id_a2d65130-3d31-43e9-9148-455603eb2502">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('summary_df')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_a2d65130-3d31-43e9-9148-455603eb2502 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('summary_df');
      }
      })();
    </script>
  </div>

    </div>
  </div>



## 13. Final summary table and accuracy chart


```python
summary_df = pd.DataFrame(all_results)

summary_path = RESULTS_DIR / "full_experiment_summary.csv"
summary_df.to_csv(summary_path, index=False)

display(summary_df.sort_values("test_accuracy", ascending=False))

plt.figure(figsize=(11, 5))
plt.bar(summary_df["run"], summary_df["test_accuracy"] * 100.0)
plt.xticks(rotation=75, ha="right")
plt.xlabel("Run")
plt.ylabel("Test Accuracy (%)")
plt.title("Full Experiment Test Accuracy")
plt.tight_layout()

chart_path = RESULTS_DIR / "full_experiment_accuracy_bar_chart.png"
plt.savefig(chart_path, dpi=200)
plt.show()

print("Saved summary table:", summary_path)
print("Saved bar chart:", chart_path)
```



  <div id="df-d82b5a64-48df-4f19-b712-8ba1ab114335" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>run</th>
      <th>model</th>
      <th>dropout</th>
      <th>epochs</th>
      <th>parameters</th>
      <th>best_validation_accuracy</th>
      <th>test_accuracy</th>
      <th>test_loss</th>
      <th>avg_epoch_time_sec</th>
      <th>device</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>resnet18_dropout03</td>
      <td>resnet18</td>
      <td>0.3</td>
      <td>50</td>
      <td>11173962</td>
      <td>0.9430</td>
      <td>0.9394</td>
      <td>0.226434</td>
      <td>3.324515</td>
      <td>cuda</td>
    </tr>
    <tr>
      <th>11</th>
      <td>resnet18_dropout05</td>
      <td>resnet18</td>
      <td>0.5</td>
      <td>50</td>
      <td>11173962</td>
      <td>0.9430</td>
      <td>0.9393</td>
      <td>0.236496</td>
      <td>3.322602</td>
      <td>cuda</td>
    </tr>
    <tr>
      <th>9</th>
      <td>resnet18_dropout00</td>
      <td>resnet18</td>
      <td>0.0</td>
      <td>50</td>
      <td>11173962</td>
      <td>0.9440</td>
      <td>0.9335</td>
      <td>0.237261</td>
      <td>3.312322</td>
      <td>cuda</td>
    </tr>
    <tr>
      <th>7</th>
      <td>resnet11_dropout03</td>
      <td>resnet11</td>
      <td>0.3</td>
      <td>50</td>
      <td>4903242</td>
      <td>0.9354</td>
      <td>0.9299</td>
      <td>0.245837</td>
      <td>1.994428</td>
      <td>cuda</td>
    </tr>
    <tr>
      <th>8</th>
      <td>resnet11_dropout05</td>
      <td>resnet11</td>
      <td>0.5</td>
      <td>50</td>
      <td>4903242</td>
      <td>0.9344</td>
      <td>0.9298</td>
      <td>0.233818</td>
      <td>1.981786</td>
      <td>cuda</td>
    </tr>
    <tr>
      <th>6</th>
      <td>resnet11_dropout00</td>
      <td>resnet11</td>
      <td>0.0</td>
      <td>50</td>
      <td>4903242</td>
      <td>0.9396</td>
      <td>0.9280</td>
      <td>0.247211</td>
      <td>1.988500</td>
      <td>cuda</td>
    </tr>
    <tr>
      <th>0</th>
      <td>alexnet_dropout00</td>
      <td>alexnet</td>
      <td>0.0</td>
      <td>30</td>
      <td>3192458</td>
      <td>0.8584</td>
      <td>0.8517</td>
      <td>0.447245</td>
      <td>2.064835</td>
      <td>cuda</td>
    </tr>
    <tr>
      <th>1</th>
      <td>alexnet_dropout03</td>
      <td>alexnet</td>
      <td>0.3</td>
      <td>30</td>
      <td>3192458</td>
      <td>0.8554</td>
      <td>0.8500</td>
      <td>0.443515</td>
      <td>2.050161</td>
      <td>cuda</td>
    </tr>
    <tr>
      <th>2</th>
      <td>alexnet_dropout05</td>
      <td>alexnet</td>
      <td>0.5</td>
      <td>30</td>
      <td>3192458</td>
      <td>0.8486</td>
      <td>0.8473</td>
      <td>0.452922</td>
      <td>2.056139</td>
      <td>cuda</td>
    </tr>
    <tr>
      <th>3</th>
      <td>vgg_dropout00</td>
      <td>vgg</td>
      <td>0.0</td>
      <td>30</td>
      <td>3586698</td>
      <td>0.1028</td>
      <td>0.1000</td>
      <td>2.302670</td>
      <td>2.038783</td>
      <td>cuda</td>
    </tr>
    <tr>
      <th>4</th>
      <td>vgg_dropout03</td>
      <td>vgg</td>
      <td>0.3</td>
      <td>30</td>
      <td>3586698</td>
      <td>0.1028</td>
      <td>0.1000</td>
      <td>2.302676</td>
      <td>2.063646</td>
      <td>cuda</td>
    </tr>
    <tr>
      <th>5</th>
      <td>vgg_dropout05</td>
      <td>vgg</td>
      <td>0.5</td>
      <td>30</td>
      <td>3586698</td>
      <td>0.1028</td>
      <td>0.1000</td>
      <td>2.302674</td>
      <td>2.047839</td>
      <td>cuda</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-d82b5a64-48df-4f19-b712-8ba1ab114335')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-d82b5a64-48df-4f19-b712-8ba1ab114335 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-d82b5a64-48df-4f19-b712-8ba1ab114335');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>




    
![png](README_files/README_23_1.png)
    


    Saved summary table: results/full_experiment_summary.csv
    Saved bar chart: results/full_experiment_accuracy_bar_chart.png


## 14. Best model


```python
best_row = summary_df.loc[summary_df["test_accuracy"].idxmax()]

print("Best model based on test accuracy:")
display(best_row.to_frame(name="value"))
```

    Best model based on test accuracy:




  <div id="df-ae607c37-b48a-4d88-93bd-f2e707537946" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>run</th>
      <td>resnet18_dropout03</td>
    </tr>
    <tr>
      <th>model</th>
      <td>resnet18</td>
    </tr>
    <tr>
      <th>dropout</th>
      <td>0.3</td>
    </tr>
    <tr>
      <th>epochs</th>
      <td>50</td>
    </tr>
    <tr>
      <th>parameters</th>
      <td>11173962</td>
    </tr>
    <tr>
      <th>best_validation_accuracy</th>
      <td>0.943</td>
    </tr>
    <tr>
      <th>test_accuracy</th>
      <td>0.9394</td>
    </tr>
    <tr>
      <th>test_loss</th>
      <td>0.226434</td>
    </tr>
    <tr>
      <th>avg_epoch_time_sec</th>
      <td>3.324515</td>
    </tr>
    <tr>
      <th>device</th>
      <td>cuda</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-ae607c37-b48a-4d88-93bd-f2e707537946')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-ae607c37-b48a-4d88-93bd-f2e707537946 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-ae607c37-b48a-4d88-93bd-f2e707537946');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>



## 15. Display saved result images


```python
if chart_path.exists():
    print(chart_path)
    display(Image(filename=str(chart_path)))

for image_path in sorted(RESULTS_DIR.glob("**/*.png")):
    print(image_path)
    display(Image(filename=str(image_path)))
```
