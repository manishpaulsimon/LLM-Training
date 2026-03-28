"""
Project 1: CIFAR-10 Image Classifier from Scratch
===================================================
Goal: Learn PyTorch fundamentals — tensors, datasets, training loops, GPU usage.
Target: ~85% accuracy on CIFAR-10.

Run:
    python 01-fundamentals/01_cifar_classifier.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import os


# ---------------------------------------------------------------------------
# 1. Hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 25
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

# ---------------------------------------------------------------------------
# 2. Data loading & augmentation
# ---------------------------------------------------------------------------
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

data_dir = os.path.join(os.path.dirname(__file__), "data")

train_dataset = torchvision.datasets.CIFAR10(
    root=data_dir, train=True, download=True, transform=train_transform,
)
test_dataset = torchvision.datasets.CIFAR10(
    root=data_dir, train=False, download=True, transform=test_transform,
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
)


# ---------------------------------------------------------------------------
# 3. CNN Architecture
# ---------------------------------------------------------------------------
class CIFAR10Net(nn.Module):
    """
    A moderately deep CNN for CIFAR-10.
    Conv blocks with batch norm → global average pool → classifier.
    """

    def __init__(self):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            conv_block(3, 64),     # 32x32 → 16x16
            conv_block(64, 128),   # 16x16 → 8x8
            conv_block(128, 256),  # 8x8  → 4x4
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 4x4 → 1x1
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        # Move data to GPU
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


# ---------------------------------------------------------------------------
# 5. Evaluation loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


# ---------------------------------------------------------------------------
# 6. Main training script
# ---------------------------------------------------------------------------
def main():
    print(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    model = CIFAR10Net().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training history for plotting
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Loss':>9} | {'Test Acc':>8} | {'Time':>6}")
    print("-" * 70)

    for epoch in range(1, EPOCHS + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        scheduler.step()

        elapsed = time.time() - start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:8.2f}% | {test_loss:9.4f} | {test_acc:7.2f}% | {elapsed:5.1f}s")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "cifar10_best.pth"))

    print(f"\nBest test accuracy: {best_acc:.2f}%")

    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["test_loss"], label="Test")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()

    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["test_acc"], label="Test")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy Curves")
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "cifar10_training.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Training plot saved to {plot_path}")


if __name__ == "__main__":
    main()
