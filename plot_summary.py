import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    results_dir = Path("./results")

    rows = []

    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        config_path = run_dir / "config.json"
        final_path = run_dir / "final_results.json"
        log_path = run_dir / "training_log.csv"

        if not config_path.exists() or not final_path.exists() or not log_path.exists():
            continue

        config = json.loads(config_path.read_text())
        final = json.loads(final_path.read_text())
        log = pd.read_csv(log_path)

        rows.append({
            "run": run_dir.name,
            "model": config["model"],
            "dropout": config["dropout"],
            "parameters": config["parameters"],
            "test_accuracy": final["test_accuracy"],
            "best_validation_accuracy": final["best_validation_accuracy"],
            "avg_epoch_time_sec": log["epoch_time_sec"].mean(),
            "epochs": config["epochs"],
            "learning_rate": config["learning_rate"],
            "weight_decay": config["weight_decay"],
        })

    df = pd.DataFrame(rows)

    if df.empty:
        print("No completed runs found.")
        return

    df.to_csv(results_dir / "summary_table.csv", index=False)

    # Best run per model family.
    best = df.sort_values("test_accuracy", ascending=False).groupby("model").head(1)

    plt.figure()
    plt.bar(best["model"], best["test_accuracy"] * 100.0)
    plt.xlabel("Model")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Best Test Accuracy by Model Family")
    plt.tight_layout()
    plt.savefig(results_dir / "best_model_test_accuracy_bar_chart.png", dpi=200)
    plt.close()

    print(df.sort_values(["model", "dropout"]))
    print()
    print("Saved:")
    print(results_dir / "summary_table.csv")
    print(results_dir / "best_model_test_accuracy_bar_chart.png")


if __name__ == "__main__":
    main()
