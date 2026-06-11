import argparse
import csv
import json
import platform
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from models.alexnet_cifar import AlexNetCIFAR
from models.vgg_cifar import VGGCIFAR
from models.resnet_cifar import ResNet11CIFAR, ResNet18CIFAR
from utils.data import get_cifar10_loaders, CIFAR10_CLASSES
from utils.metrics import evaluate, count_trainable_parameters
from utils.plotting import plot_curves, plot_confusion_matrix
from utils.seed import set_seed


def build_model(model_name: str, dropout: float):
    if model_name == "alexnet":
        return AlexNetCIFAR(dropout=dropout)
    if model_name == "vgg":
        return VGGCIFAR(dropout=dropout)
    if model_name == "resnet11":
        return ResNet11CIFAR(dropout=dropout)
    if model_name == "resnet18":
        return ResNet18CIFAR(dropout=dropout)

    raise ValueError(f"Unknown model: {model_name}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        batch_size = targets.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True,
                        choices=["alexnet", "vgg", "resnet11", "resnet18"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dropout_tag = str(args.dropout).replace(".", "")
    run_name = f"{args.model}_dropout{dropout_tag}"
    output_dir = Path(args.results_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers
    )

    model = build_model(args.model, args.dropout).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    param_count = count_trainable_parameters(model)

    config = {
        "model": args.model,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "scheduler": "CosineAnnealingLR",
        "seed": args.seed,
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

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    log_path = output_dir / "training_log.csv"

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

    print(f"Run: {run_name}")
    print(f"Device: {device}")
    print(f"Trainable parameters: {param_count:,}")

    for epoch in range(1, args.epochs + 1):
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
            torch.save(model.state_dict(), output_dir / "best_model.pt")

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} | "
            f"val acc {val_acc:.4f} | "
            f"time {epoch_time:.1f}s"
        )

    # Load best checkpoint before final test.
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))

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

    with open(output_dir / "final_results.json", "w") as f:
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

    print(f"Final test accuracy: {test_acc:.4f}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
