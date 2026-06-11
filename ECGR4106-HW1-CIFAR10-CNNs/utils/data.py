import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    seed: int = 42,
    num_workers: int = 2,
):
    """
    Creates a consistent CIFAR-10 train/validation/test split.

    Training uses augmentation:
    - RandomCrop(32, padding=4)
    - RandomHorizontalFlip()

    Validation and test do NOT use augmentation.
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    full_train_aug = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )

    full_train_eval = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=eval_transform
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=eval_transform
    )

    generator = torch.Generator().manual_seed(seed)

    train_size = 45000
    val_size = 5000

    # Same indices, but different transforms for train and validation.
    train_subset_aug, _ = random_split(
        full_train_aug,
        [train_size, val_size],
        generator=generator
    )

    generator = torch.Generator().manual_seed(seed)
    _, val_subset_eval = random_split(
        full_train_eval,
        [train_size, val_size],
        generator=generator
    )

    train_loader = DataLoader(
        train_subset_aug,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_subset_eval,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
