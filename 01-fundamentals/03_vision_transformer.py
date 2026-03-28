"""
Project 3: Vision Transformer (ViT) from Scratch
==================================================
Goal: Apply the transformer architecture to images — same attention mechanism
as Baby GPT, but for image classification instead of text generation.

Run:
    python 01-fundamentals/03_vision_transformer.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
PATCH_SIZE = 8            # split 32x32 image into 8x8 patches → 16 patches
EMBED_DIM = 128           # dimension of each patch/token embedding
NUM_HEADS = 4
NUM_LAYERS = 6
FF_DIM = 512
DROPOUT = 0.1
LEARNING_RATE = 3e-4
EPOCHS = 25
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_PATCHES = (32 // PATCH_SIZE) ** 2  # 16 patches for 32x32 with 8x8 patches
NUM_CLASSES = 10

# ---------------------------------------------------------------------------
# 2. Data loading — same CIFAR-10 as Phase 1B
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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


# ---------------------------------------------------------------------------
# 3. Transformer building blocks (same concepts as Baby GPT, no causal mask)
# ---------------------------------------------------------------------------
class SelfAttention(nn.Module):
    """Single head of bidirectional self-attention (no causal mask — unlike GPT)."""

    def __init__(self, head_dim):
        super().__init__()
        self.query = nn.Linear(EMBED_DIM, head_dim, bias=False)
        self.key = nn.Linear(EMBED_DIM, head_dim, bias=False)
        self.value = nn.Linear(EMBED_DIM, head_dim, bias=False)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # No causal mask — every patch can attend to every other patch
        scores = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        return weights @ v


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        head_dim = EMBED_DIM // NUM_HEADS
        self.heads = nn.ModuleList([SelfAttention(head_dim) for _ in range(NUM_HEADS)])
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBED_DIM, FF_DIM),
            nn.GELU(),       # ViT uses GELU instead of ReLU
            nn.Linear(FF_DIM, EMBED_DIM),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.attn = MultiHeadAttention()
        self.ln2 = nn.LayerNorm(EMBED_DIM)
        self.ff = FeedForward()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# 4. Vision Transformer model
# ---------------------------------------------------------------------------
class VisionTransformer(nn.Module):
    """
    ViT: split image into patches → embed → transformer → classify.
    The key insight: treat image patches like words in a sentence.
    """

    def __init__(self):
        super().__init__()
        patch_dim = 3 * PATCH_SIZE * PATCH_SIZE  # 3 channels * 8 * 8 = 192

        # Patch embedding: flatten each 8x8x3 patch → project to EMBED_DIM
        self.patch_embed = nn.Linear(patch_dim, EMBED_DIM)

        # CLS token: a learnable vector prepended to the patch sequence
        # After the transformer, this token's output is used for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, EMBED_DIM))

        # Position embeddings: 16 patches + 1 CLS token = 17 positions
        self.pos_emb = nn.Embedding(NUM_PATCHES + 1, EMBED_DIM)

        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(NUM_LAYERS)])
        self.ln_final = nn.LayerNorm(EMBED_DIM)

        # Classification head: CLS token output → class prediction
        self.head = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(EMBED_DIM, NUM_CLASSES),
        )

    def forward(self, x):
        B = x.shape[0]

        # Split image into patches and flatten each patch
        # (B, 3, 32, 32) → unfold into (B, 16, 192) patches
        patches = x.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
        patches = patches.contiguous().view(B, 3, NUM_PATCHES, PATCH_SIZE, PATCH_SIZE)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()  # (B, 16, 3, 8, 8)
        patches = patches.view(B, NUM_PATCHES, -1)               # (B, 16, 192)

        # Embed patches
        x = self.patch_embed(patches)                             # (B, 16, EMBED_DIM)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)                   # (B, 1, EMBED_DIM)
        x = torch.cat([cls, x], dim=1)                           # (B, 17, EMBED_DIM)

        # Add position embeddings
        positions = torch.arange(NUM_PATCHES + 1, device=DEVICE)
        x = x + self.pos_emb(positions)                           # (B, 17, EMBED_DIM)

        # Transformer
        x = self.blocks(x)
        x = self.ln_final(x)

        # Classify using the CLS token (first token)
        cls_output = x[:, 0]                                      # (B, EMBED_DIM)
        return self.head(cls_output)                               # (B, 10)


# ---------------------------------------------------------------------------
# 5. Training and evaluation (same pattern as CIFAR-10 CNN)
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


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

    return running_loss / total, 100.0 * correct / total


def main():
    print(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = VisionTransformer().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE} → {NUM_PATCHES} patches per image")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0
    save_path = os.path.join(os.path.dirname(__file__), "vit_best.pth")

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

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)

    print(f"\nBest test accuracy: {best_acc:.2f}%")

    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["test_loss"], label="Test")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("ViT Loss Curves")
    ax1.legend()

    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["test_acc"], label="Test")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("ViT Accuracy Curves")
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "vit_training.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Training plot saved to {plot_path}")


if __name__ == "__main__":
    main()
