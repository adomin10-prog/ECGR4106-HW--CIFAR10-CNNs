import subprocess
import sys


EXPERIMENTS = [
    ["alexnet", "0.0", "30", "0.01"],
    ["alexnet", "0.3", "30", "0.01"],
    ["alexnet", "0.5", "30", "0.01"],

    ["vgg", "0.0", "30", "0.01"],
    ["vgg", "0.3", "30", "0.01"],
    ["vgg", "0.5", "30", "0.01"],

    ["resnet11", "0.0", "50", "0.1"],
    ["resnet11", "0.3", "50", "0.1"],
    ["resnet11", "0.5", "50", "0.1"],

    ["resnet18", "0.0", "50", "0.1"],
    ["resnet18", "0.3", "50", "0.1"],
    ["resnet18", "0.5", "50", "0.1"],
]


def main():
    for model, dropout, epochs, lr in EXPERIMENTS:
        cmd = [
            sys.executable,
            "train.py",
            "--model", model,
            "--dropout", dropout,
            "--epochs", epochs,
            "--lr", lr,
        ]

        print("=" * 80)
        print("Running:", " ".join(cmd))
        print("=" * 80)

        subprocess.run(cmd, check=True)

    subprocess.run([sys.executable, "plot_summary.py"], check=True)


if __name__ == "__main__":
    main()
