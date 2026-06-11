import torch
from sklearn.metrics import confusion_matrix


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_targets = []
    all_preds = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)

        preds = logits.argmax(dim=1)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

        all_targets.extend(targets.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    cm = confusion_matrix(all_targets, all_preds)

    return avg_loss, accuracy, cm


def count_trainable_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
