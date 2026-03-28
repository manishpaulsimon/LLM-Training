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
