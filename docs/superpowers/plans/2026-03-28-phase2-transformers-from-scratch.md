# Phase 2: Transformers from Scratch — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Baby GPT character-level text generator and a Vision Transformer image classifier from scratch to learn the transformer architecture.

**Architecture:** Two standalone scripts in `01-fundamentals/`. Baby GPT uses causal (masked) self-attention for autoregressive text generation. ViT uses bidirectional self-attention on image patches for classification. Both share the same core transformer block structure.

**Tech Stack:** PyTorch, torchvision (CIFAR-10), matplotlib

---

## File Structure

| File | Purpose |
|------|---------|
| `01-fundamentals/02_baby_gpt.py` | Character-level transformer text generator |
| `01-fundamentals/03_vision_transformer.py` | ViT image classifier on CIFAR-10 |
| `docs/02-baby-gpt-theory.md` | Theory doc: attention, positional encoding, generation |
| `docs/03-vision-transformer-theory.md` | Theory doc: patches, CLS token, ViT vs CNN |

---

### Task 1: Baby GPT — Data Loading

**Files:**
- Create: `01-fundamentals/02_baby_gpt.py`

- [ ] **Step 1: Create the script with imports, config, and data loading**

```python
"""
Project 2: Baby GPT — Character-Level Text Generator
======================================================
Goal: Learn the transformer architecture by building a small GPT from scratch.
Trains on Shakespeare text, then generates new text character by character.

Run:
    python 01-fundamentals/02_baby_gpt.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import urllib.request

# ---------------------------------------------------------------------------
# 1. Hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
BLOCK_SIZE = 128          # context window (how many chars the model sees at once)
EMBED_DIM = 128           # size of each token's vector representation
NUM_HEADS = 4             # attention heads (EMBED_DIM must be divisible by this)
NUM_LAYERS = 4            # transformer blocks stacked
FF_DIM = 512              # inner dimension of feed-forward network
DROPOUT = 0.1
LEARNING_RATE = 3e-4
MAX_ITERS = 5000
EVAL_INTERVAL = 500       # evaluate and print sample every N iterations
EVAL_ITERS = 200          # batches to average for eval loss
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 2. Data loading — Shakespeare text, character-level tokenization
# ---------------------------------------------------------------------------
data_dir = os.path.join(os.path.dirname(__file__), "data")
data_path = os.path.join(data_dir, "input.txt")

# Download Shakespeare if not present
if not os.path.exists(data_path):
    os.makedirs(data_dir, exist_ok=True)
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print(f"Downloading Shakespeare text...")
    urllib.request.urlretrieve(url, data_path)

with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

# Character-level tokenizer — each unique character becomes a token
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}  # char → index
itos = {i: ch for i, ch in enumerate(chars)}  # index → char
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# Train/val split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Dataset: {len(text):,} characters, {vocab_size} unique tokens")
print(f"Train: {len(train_data):,} | Val: {len(val_data):,}")
print(f"Sample tokens: {chars[:20]}")


def get_batch(split):
    """Get a random batch of input sequences and their targets."""
    d = train_data if split == "train" else val_data
    # Random starting positions for each sequence in the batch
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([d[i:i + BLOCK_SIZE] for i in ix])
    # Target is the same sequence shifted by one character
    y = torch.stack([d[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)
```

- [ ] **Step 2: Test the data loading works**

Run: `python -c "exec(open('01-fundamentals/02_baby_gpt.py').read().split('def get_batch')[0])"`

Expected: Prints dataset stats — ~1.1M characters, 65 unique tokens, no errors.

---

### Task 2: Baby GPT — Transformer Architecture

**Files:**
- Modify: `01-fundamentals/02_baby_gpt.py`

- [ ] **Step 1: Add the self-attention head**

Append after the `get_batch` function:

```python
# ---------------------------------------------------------------------------
# 3. Transformer building blocks
# ---------------------------------------------------------------------------
class SelfAttention(nn.Module):
    """Single head of causal (masked) self-attention."""

    def __init__(self, head_dim):
        super().__init__()
        self.query = nn.Linear(EMBED_DIM, head_dim, bias=False)
        self.key = nn.Linear(EMBED_DIM, head_dim, bias=False)
        self.value = nn.Linear(EMBED_DIM, head_dim, bias=False)
        self.dropout = nn.Dropout(DROPOUT)
        # Causal mask — prevents attending to future characters
        self.register_buffer("mask", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)  # (B, T, head_dim)
        k = self.key(x)    # (B, T, head_dim)
        v = self.value(x)  # (B, T, head_dim)

        # Attention scores: how much each position attends to every other
        scores = q @ k.transpose(-2, -1) * (C ** -0.5)  # scaled dot-product
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))  # mask future
        weights = F.softmax(scores, dim=-1)  # normalize to probabilities
        weights = self.dropout(weights)

        return weights @ v  # weighted sum of values
```

- [ ] **Step 2: Add multi-head attention**

```python
class MultiHeadAttention(nn.Module):
    """Multiple attention heads in parallel, then concatenated."""

    def __init__(self):
        super().__init__()
        head_dim = EMBED_DIM // NUM_HEADS
        self.heads = nn.ModuleList([SelfAttention(head_dim) for _ in range(NUM_HEADS)])
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM)  # linear projection after concat
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        # Run all heads in parallel, concatenate results
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))
```

- [ ] **Step 3: Add feed-forward network and transformer block**

```python
class FeedForward(nn.Module):
    """Two-layer MLP with ReLU — processes each position independently."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBED_DIM, FF_DIM),
            nn.ReLU(),
            nn.Linear(FF_DIM, EMBED_DIM),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """One transformer block: attention → feed-forward, with residuals and layer norm."""

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.attn = MultiHeadAttention()
        self.ln2 = nn.LayerNorm(EMBED_DIM)
        self.ff = FeedForward()

    def forward(self, x):
        # Pre-norm architecture (GPT-2 style): norm before each sub-layer
        x = x + self.attn(self.ln1(x))   # residual connection around attention
        x = x + self.ff(self.ln2(x))     # residual connection around feed-forward
        return x
```

- [ ] **Step 4: Add the full BabyGPT model**

```python
# ---------------------------------------------------------------------------
# 4. The full model
# ---------------------------------------------------------------------------
class BabyGPT(nn.Module):
    """
    A small GPT: token embeddings + positional embeddings → transformer blocks → output logits.
    """

    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, EMBED_DIM)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(NUM_LAYERS)])
        self.ln_final = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)                              # (B, T, EMBED_DIM)
        pos = self.pos_emb(torch.arange(T, device=DEVICE))    # (T, EMBED_DIM)
        x = tok + pos                                          # combine token + position info
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.head(x)                                  # (B, T, vocab_size)

        if targets is None:
            return logits, None

        # Reshape for cross-entropy: flatten batch and sequence dimensions
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Autoregressive generation: predict one character at a time."""
        for _ in range(max_new_tokens):
            # Crop to last BLOCK_SIZE tokens (model's context window)
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            # Take logits at the last position, convert to probabilities
            probs = F.softmax(logits[:, -1, :], dim=-1)
            # Sample from the distribution (not argmax — adds variety)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
```

---

### Task 3: Baby GPT — Training Loop and Generation

**Files:**
- Modify: `01-fundamentals/02_baby_gpt.py`

- [ ] **Step 1: Add evaluation and training loop**

```python
# ---------------------------------------------------------------------------
# 5. Training
# ---------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(model):
    """Average loss over multiple batches for stable measurement."""
    model.eval()
    losses = {}
    for split in ["train", "val"]:
        batch_losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            batch_losses[k] = loss.item()
        losses[split] = batch_losses.mean().item()
    model.train()
    return losses


def main():
    print(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = BabyGPT().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_loss = float("inf")
    save_path = os.path.join(os.path.dirname(__file__), "babygpt_best.pth")

    print(f"\n{'Iter':>6} | {'Train Loss':>10} | {'Val Loss':>8} | {'Time':>6}")
    print("-" * 50)

    start = time.time()

    for step in range(1, MAX_ITERS + 1):
        # Get a random batch and train
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Periodic evaluation
        if step % EVAL_INTERVAL == 0 or step == 1:
            losses = estimate_loss(model)
            elapsed = time.time() - start
            print(f"{step:6d} | {losses['train']:10.4f} | {losses['val']:8.4f} | {elapsed:5.1f}s")

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save(model.state_dict(), save_path)

            # Generate a sample
            prompt = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)  # start with newline char
            sample = decode(model.generate(prompt, max_new_tokens=200)[0].tolist())
            print(f"\n--- Sample (iter {step}) ---")
            print(sample[:300])
            print("---\n")

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run Baby GPT training**

Run: `python 01-fundamentals/02_baby_gpt.py`

Expected:
- Downloads Shakespeare text on first run
- Prints ~1.1M characters, 65 unique tokens
- Trains for 5000 iterations (~5-10 min on RTX 3070)
- Every 500 steps: prints train/val loss + generates a text sample
- Early samples = gibberish. Later samples = Shakespeare-like text with real words
- Val loss should reach ~1.4-1.6

- [ ] **Step 3: Commit**

```bash
git add 01-fundamentals/02_baby_gpt.py
git commit -m "feat: add Baby GPT character-level transformer"
```

---

### Task 4: Baby GPT — Theory Doc

**Files:**
- Create: `docs/02-baby-gpt-theory.md`

- [ ] **Step 1: Write the theory doc**

Write a comprehensive theory doc covering:
- What is a transformer and why it replaced RNNs/LSTMs
- Self-attention explained with the query/key/value analogy
- Causal masking — why GPT can't look at future tokens
- Multi-head attention — why multiple heads help
- Positional encoding — why transformers need position info
- Feed-forward network — what it does in the transformer block
- Residual connections and layer norm — why they're needed
- The full forward pass with tensor shapes at each step
- Autoregressive generation — how the model generates text one token at a time
- Cross-entropy loss on characters vs CIFAR's 10 classes
- Code-to-concept mapping: which lines implement which concepts
- Connection to real GPT/LLaMA: what's the same, what's different at scale

Style: same as `docs/01-cifar10-classifier-theory.md` — explain the "why", reference line numbers, include the actual code alongside theory.

- [ ] **Step 2: Commit**

```bash
git add docs/02-baby-gpt-theory.md
git commit -m "docs: add Baby GPT theory walkthrough"
```

---

### Task 5: Vision Transformer — Full Script

**Files:**
- Create: `01-fundamentals/03_vision_transformer.py`

- [ ] **Step 1: Create the ViT script**

```python
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
```

- [ ] **Step 2: Run the ViT training**

Run: `python 01-fundamentals/03_vision_transformer.py`

Expected:
- Uses already-downloaded CIFAR-10 data
- ~3-4M parameters
- 25 epochs, ~60-90s per epoch on RTX 3070
- Target: ~80-85% test accuracy (lower than CNN — ViTs need more data to match CNNs on small datasets)
- Saves `vit_best.pth` and `vit_training.png`

- [ ] **Step 3: Commit**

```bash
git add 01-fundamentals/03_vision_transformer.py
git commit -m "feat: add Vision Transformer (ViT) from scratch"
```

---

### Task 6: Vision Transformer — Theory Doc

**Files:**
- Create: `docs/03-vision-transformer-theory.md`

- [ ] **Step 1: Write the theory doc**

Write a comprehensive theory doc covering:
- What is ViT and why apply transformers to images
- Patch embedding — how images become sequences (like words)
- CLS token — what it is and why it works for classification
- Bidirectional vs causal attention — ViT sees all patches, GPT can't see future tokens
- GELU vs ReLU — why ViT uses GELU
- Why ViT underperforms CNN on small datasets (CIFAR-10) but dominates on large datasets (ImageNet)
- Code-to-concept mapping with tensor shapes at each step
- Connection to vision-language models (Phase 3): how ViT patch embeddings become the "visual tokens" fed to an LLM

Style: same as previous theory docs.

- [ ] **Step 2: Commit**

```bash
git add docs/03-vision-transformer-theory.md
git commit -m "docs: add Vision Transformer theory walkthrough"
```

---

### Task 7: Update README Progress

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the progress checklist**

Change the Phase 1B and Phase 2 lines from `- [ ]` to `- [x]`:

```markdown
- [x] Phase 1A: Environment setup
- [x] Phase 1B: CIFAR-10 classifier (PyTorch fundamentals)
- [ ] Phase 1C: Data collection & labeling schema
- [x] Phase 2: Vision Transformer + Baby GPT from scratch
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: mark Phase 1B and Phase 2 as complete"
```
