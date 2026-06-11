"""
Verification runner for ECGR 4106 Homework 1.

This script verifies that the repository can run successfully in Google Colab
or a local Python environment.

It performs the following checks:

1. imports and tests all model definitions,
2. runs a forward pass for every model,
3. trains every required model/dropout configuration for a short 1-epoch
   verification run,
4. saves result files,
5. creates a summary table and bar chart.

This verification run is not intended to replace the final homework experiments.
The final report should use the full 30-epoch AlexNet/VGG runs and the
50-epoch ResNet runs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


VERIFICATION_EXPERIMENTS = [
    # model, dropout, epochs, learning rate
    ("alexnet", "0.0", "1", "0.01"),
    ("alexnet", "0.3", "1", "0.01"),
    ("alexnet", "0.5", "1", "0.01"),

    ("vgg", "0.0", "1", "0.01"),
    ("vgg", "0.3", "1", "0.01"),
    ("vgg", "0.5", "1", "0.01"),

    ("resnet11", "0.0", "1", "0.1"),
    ("resnet11", "0.3", "1", "0.1"),
    ("resnet11", "0.5", "1", "0.1"),

    ("resnet18", "0.0", "1", "0.1"),
    ("resnet18", "0.3", "1", "0.1"),
    ("resnet18", "0.5", "1", "0.1"),
]


FULL_EXPERIMENTS = [
    ("alexnet", "0.0", "30", "0.01"),
    ("alexnet", "0.3", "30", "0.01"),
    ("alexnet", "0.5", "30", "0.01"),

    ("vgg", "0.0", "30", "0.01"),
    ("vgg", "0.3", "30", "0.01"),
    ("vgg", "0.5", "30", "0.01"),

    ("resnet11", "0.0", "50", "0.1"),
    ("resnet11", "0.3", "50", "0.1"),
    ("resnet11", "0.5", "50", "0.1"),

    ("resnet18", "0.0", "50", "0.1"),
    ("resnet18", "0.3", "50", "0.1"),
    ("resnet18", "0.5", "50", "0.1"),
]


def run_command(cmd: list[str]) -> None:
    print("\n" + "=" * 90)
    print("Running:", " ".join(cmd))
    print("=" * 90)
    subprocess.run(cmd, check=True)


def summarize_results(results_dir: Path, output_prefix: str) -> pd.DataFrame:
    rows = []

    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        config_path = run_dir / "config.json"
        final_path = run_dir / "final_results.json"
        log_path = run_dir / "training_log.csv"

        if not (config_path.exists() and final_path.exists() and log_path.exists()):
            continue

        config = json.loads(config_path.read_text())
        final = json.loads(final_path.read_text())
        log = pd.read_csv(log_path)

        rows.append({
            "run": run_dir.name,
            "model": config["model"],
            "dropout": config["dropout"],
            "epochs": config["epochs"],
            "parameters": config["parameters"],
            "best_validation_accuracy": final["best_validation_accuracy"],
            "test_accuracy": final["test_accuracy"],
            "test_loss": final["test_loss"],
            "avg_epoch_time_sec": log["epoch_time_sec"].mean(),
            "device": config["device"],
            "gpu_name": config.get("gpu_name", "CPU"),
        })

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(f"No completed result folders found in {results_dir}")

    summary_path = results_dir / f"{output_prefix}_summary.csv"
    df.to_csv(summary_path, index=False)

    plt.figure(figsize=(11, 5))
    labels = df["run"]
    values = df["test_accuracy"] * 100.0
    plt.bar(labels, values)
    plt.xticks(rotation=75, ha="right")
    plt.xlabel("Run")
    plt.ylabel("Test Accuracy (%)")
    plt.title(f"{output_prefix.replace('_', ' ').title()} Test Accuracy")
    plt.tight_layout()

    chart_path = results_dir / f"{output_prefix}_accuracy_bar_chart.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()

    print("\n" + "=" * 90)
    print("RESULT SUMMARY")
    print("=" * 90)
    print(df.to_string(index=False))
    print("\nSaved summary table:", summary_path)
    print("Saved bar chart:", chart_path)

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["verification", "full"],
        default="verification",
        help=(
            "verification = 1 epoch per required run to verify reproducibility; "
            "full = final homework epoch counts"
        ),
    )
    args = parser.parse_args()

    if args.mode == "verification":
        experiments = VERIFICATION_EXPERIMENTS
        results_dir = Path("./verification_results")
        output_prefix = "verification"
    else:
        experiments = FULL_EXPERIMENTS
        results_dir = Path("./results")
        output_prefix = "full_experiment"

    results_dir.mkdir(parents=True, exist_ok=True)

    print("\nRUN MODE:", args.mode)
    if args.mode == "verification":
        print("This runs every required model/dropout configuration for 1 epoch.")
        print("Use this to verify that the repository runs correctly.")
    else:
        print("This runs the full homework experiments. This can take a long time.")

    run_command([sys.executable, "quick_model_test.py"])

    for model, dropout, epochs, lr in experiments:
        run_command([
            sys.executable,
            "train.py",
            "--model", model,
            "--dropout", dropout,
            "--epochs", epochs,
            "--lr", lr,
            "--results_dir", str(results_dir),
        ])

    summarize_results(results_dir, output_prefix)


if __name__ == "__main__":
    main()
