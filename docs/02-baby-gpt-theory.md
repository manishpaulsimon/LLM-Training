# Phase 2: Baby GPT — Theory & Code Walkthrough

> Script: `01-fundamentals/02_baby_gpt.py`
> Run: `python 01-fundamentals/02_baby_gpt.py`

---

## Section 1: What Is a Transformer?

A **transformer** is a neural network architecture designed for sequences (text, audio, time series). It was introduced in the 2017 paper "Attention Is All You Need" and rapidly replaced RNNs and LSTMs as the dominant approach for language tasks.

**Why did transformers replace RNNs/LSTMs?**

| Problem | RNN/LSTM | Transformer |
|---------|----------|-------------|
| Long-range dependencies | Struggles — information fades over many time steps | Handles easily — every token can attend to every other token directly |
| Training speed | Sequential — must process token 1 before token 2 | Parallel — processes all tokens simultaneously |
| Vanishing gradients | Chronic issue even with LSTM gates | Residual connections keep gradients flowing |
| Scalability | Hard to scale to billions of parameters | Scales efficiently — GPT-4, LLaMA, etc. |

**The core idea:** Instead of reading text one word at a time (like RNNs), a transformer looks at the entire sequence at once and uses **attention** to figure out which parts are relevant to each other.

**Analogy:** Reading an essay.
- RNN approach: read word by word, left to right, trying to remember everything.
- Transformer approach: spread the whole essay on a table and highlight connections between any two sentences instantly.

---

## Section 2: Hyperparameters (lines 21-32)

```python
BATCH_SIZE = 64
BLOCK_SIZE = 128          # context window (how many chars the model sees at once)
EMBED_DIM = 128           # size of each token's vector representation
NUM_HEADS = 4             # attention heads (EMBED_DIM must be divisible by this)
NUM_LAYERS = 4            # transformer blocks stacked
FF_DIM = 512              # inner dimension of feed-forward network
DROPOUT = 0.1
LEARNING_RATE = 3e-4
MAX_ITERS = 5000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

| Parameter | What it controls | Analogy |
|-----------|-----------------|---------|
| `BLOCK_SIZE = 128` | How many characters the model can see at once (context window) | How much of the page you can read at once |
| `EMBED_DIM = 128` | Dimensionality of each token's vector representation | How detailed each token's "description" is |
| `NUM_HEADS = 4` | How many independent attention patterns to learn | Having 4 readers each looking for different things |
| `NUM_LAYERS = 4` | How many transformer blocks stacked | Processing depth — each layer refines understanding |
| `FF_DIM = 512` | Width of the feed-forward network inside each block | Processing power per position |
| `DROPOUT = 0.1` | Randomly zero 10% of values during training | Prevents over-reliance on any single feature |
| `LEARNING_RATE = 3e-4` | Step size for Adam optimizer | The standard "safe" rate for transformers |

**Why 128 embedding dim?** This is a baby model. GPT-2 uses 768, GPT-3 uses 12,288. Larger dimensions capture more nuance but need more compute.

**Why 3e-4?** Karpathy's go-to for small transformers. Large models often use lower rates (1e-4 or less) with warmup schedules.

---

## Section 3: Character-Level Tokenization (lines 37-66)

```python
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}  # char -> index
itos = {i: ch for i, ch in enumerate(chars)}  # index -> char
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])
```

**Theory — why tokenize?**

Neural networks work with numbers, not text. Tokenization converts text into a sequence of integers that the model can process.

```
"Hello" -> encode -> [20, 43, 50, 50, 53]
[20, 43, 50, 50, 53] -> decode -> "Hello"
```

Each unique character (letter, digit, punctuation, space, newline) gets its own integer ID. Shakespeare's text has about 65 unique characters, so `vocab_size = 65`.

**Character-level vs. subword tokenization:**

| Approach | Vocabulary | Used by |
|----------|-----------|---------|
| Character-level (this code) | ~65 tokens (one per character) | Baby GPT, early experiments |
| BPE / subword (real models) | 50,000-100,000 tokens (common words and word pieces) | GPT-2, GPT-4, LLaMA |

Character-level is simpler to understand but slower to train — the model must learn to spell before it can learn grammar. Real models use BPE (Byte Pair Encoding) which merges frequent character pairs into tokens like "the", "ing", "tion".

### Batching sequences (lines 69-77)

```python
def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([d[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)
```

**Theory — input/target pairs for language modeling:**

The target is the input shifted by one character. The model's job: given all characters so far, predict the next one.

```
Input  (x): "To be or not t"
Target (y): "o be or not to"
                           ^--- model must predict this from what came before

Position:    0  1  2  3  4  5  6  7  8  9  10 11 12 13
Input:       T  o     b  e     o  r     n  o  t     t
Target:      o     b  e     o  r     n  o  t     t  o
```

Each position in the input has a corresponding target: the next character. So one sequence of length 128 provides 128 training examples simultaneously.

**Why random starting positions?** `torch.randint` picks 64 random starting points in the text. Each batch samples different windows, so the model sees different parts of Shakespeare each time.

---

## Section 4: Self-Attention — The Core Innovation (lines 83-107)

Self-attention is the mechanism that lets every token in a sequence "look at" every other token to gather context. It is the single most important idea in the transformer.

### The Query / Key / Value Analogy

Think of a library search:

- **Query (Q):** Your search question — "I need information about X"
- **Key (K):** The label on each book — "This book is about Y"
- **Value (V):** The actual content of each book

The attention mechanism:
1. Each token generates a **query** ("what am I looking for?")
2. Each token generates a **key** ("what do I contain?")
3. Each token generates a **value** ("here is my information")
4. Match queries against keys to find relevant tokens
5. Use the match scores to create a weighted sum of values

```python
class SelfAttention(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.query = nn.Linear(EMBED_DIM, head_dim, bias=False)
        self.key = nn.Linear(EMBED_DIM, head_dim, bias=False)
        self.value = nn.Linear(EMBED_DIM, head_dim, bias=False)
```

Three separate linear projections transform each token's embedding into Q, K, and V vectors. These are learned during training — the model discovers what makes a useful query, a useful key, and a useful value.

### The Forward Pass

```python
def forward(self, x):
    B, T, C = x.shape
    q = self.query(x)  # (B, T, head_dim)
    k = self.key(x)    # (B, T, head_dim)
    v = self.value(x)  # (B, T, head_dim)
```

- `B` = batch size (64)
- `T` = sequence length (up to 128)
- `C` = embedding dimension (128) flowing in, `head_dim` (32) flowing out of each projection

### Scaled Dot-Product Attention

```python
    scores = q @ k.transpose(-2, -1) * (C ** -0.5)  # (B, T, T)
```

This computes how much each token should attend to every other token:

```
scores[i][j] = dot_product(query_i, key_j) / sqrt(head_dim)
```

The result is a T x T matrix where entry (i, j) says "how relevant is token j to token i?"

**Why scale by sqrt(d_k)?**

Without scaling, the dot products grow large as the dimension increases. Large values push softmax into regions where gradients are extremely small (near 0 or 1), making learning nearly impossible.

```
Example with head_dim = 32:
  Unscaled dot product might be:  150.0
  After softmax: [0.0, 0.0, 0.0, 1.0, 0.0]  <- gradient is ~0 everywhere

  Scaled by 1/sqrt(32) = 1/5.66:  26.5
  After softmax: [0.01, 0.05, 0.2, 0.6, 0.14]  <- gradient flows properly
```

Scaling keeps the variance of the dot products at approximately 1, regardless of dimension. This is critical for stable training.

### Causal Masking — Why GPT Can't See the Future (lines 93, 103)

```python
# In __init__:
self.register_buffer("mask", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

# In forward:
scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
```

GPT is a **causal** (autoregressive) language model. When predicting the next character, it can only use characters that came before — never characters that come after. This mirrors how text generation works: you write left to right.

The mask is a lower-triangular matrix:

```
          Position attending TO:
          0    1    2    3    4
    0  [ 1    0    0    0    0 ]   Token 0 can only see itself
P   1  [ 1    1    0    0    0 ]   Token 1 can see 0, 1
o   2  [ 1    1    1    0    0 ]   Token 2 can see 0, 1, 2
s   3  [ 1    1    1    1    0 ]   Token 3 can see 0, 1, 2, 3
    4  [ 1    1    1    1    1 ]   Token 4 can see everything before it

Zeros become -inf before softmax -> softmax(-inf) = 0 -> future tokens ignored
```

**Why `-inf`?** After `masked_fill`, future positions have score `-inf`. When softmax converts scores to probabilities, `e^(-inf) = 0`, so those positions get zero attention weight. The model literally cannot see them.

**Why `register_buffer`?** The mask is not a learnable parameter — it is a constant. `register_buffer` tells PyTorch to move it to the GPU with the model but not include it in gradient updates.

### Softmax and Weighted Sum

```python
    weights = F.softmax(scores, dim=-1)  # normalize to probabilities
    weights = self.dropout(weights)
    return weights @ v                   # weighted sum of values
```

After masking, softmax converts raw scores into probabilities that sum to 1. Then we compute a weighted sum of value vectors — each token's output is a blend of information from all tokens it can attend to, weighted by relevance.

```
If token 5 attends to tokens 0-5 with weights [0.1, 0.0, 0.3, 0.1, 0.4, 0.1]:
  output_5 = 0.1*v0 + 0.0*v1 + 0.3*v2 + 0.1*v3 + 0.4*v4 + 0.1*v5
  -> token 5's representation is enriched with context from tokens 2 and 4
```

---

## Section 5: Multi-Head Attention — Why Multiple Heads? (lines 110-123)

```python
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        head_dim = EMBED_DIM // NUM_HEADS   # 128 / 4 = 32 per head
        self.heads = nn.ModuleList([SelfAttention(head_dim) for _ in range(NUM_HEADS)])
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))
```

**Theory — why not one big attention head?**

One attention head can only learn one pattern of "what is relevant." Language has many simultaneous relationships:

| Head | What it might learn |
|------|-------------------|
| Head 1 | Syntactic structure — subject attends to its verb |
| Head 2 | Nearby context — each token attends to its neighbors |
| Head 3 | Pattern matching — closing quotes attend to opening quotes |
| Head 4 | Semantic similarity — words with related meanings attend to each other |

With 4 heads of dimension 32 each, we get 4 independent attention patterns that are concatenated back to dimension 128. The final linear projection (`self.proj`) lets the model mix information across heads.

**Why split the dimension?** 4 heads x 32 dims = 128 total dims = same compute cost as one head of 128. Multi-head attention is essentially "free" — same cost, much richer representations.

```
Input x:  (B, T, 128)
            |
     ┌──────┼──────┬──────┐
     v      v      v      v
   Head1  Head2  Head3  Head4
  (B,T,32)(B,T,32)(B,T,32)(B,T,32)
     |      |      |      |
     └──────┼──────┴──────┘
            v  concatenate
       (B, T, 128)
            |
            v  linear projection
       (B, T, 128)
```

---

## Section 6: Positional Encoding — Why Transformers Need Position Info (lines 169-170)

```python
self.token_emb = nn.Embedding(vocab_size, EMBED_DIM)
self.pos_emb = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
```

**The problem:** Self-attention treats the input as a **set**, not a sequence. The operation `Q @ K^T` is the same regardless of token order. But "the cat sat on the mat" and "the mat sat on the cat" have very different meanings.

**The solution:** Add position information to each token's embedding.

```python
# In forward():
tok = self.token_emb(idx)                           # (B, T, 128) — what the token is
pos = self.pos_emb(torch.arange(T, device=DEVICE))  # (T, 128) — where it is
x = tok + pos                                        # combine both signals
```

`self.pos_emb` is a learned embedding table with 128 entries (one per position). Position 0 gets one 128-dim vector, position 1 gets another, etc. These are **learned** during training — the model discovers what position representations work best.

**Why addition, not concatenation?** Concatenation would double the dimension. Addition is cheaper and works well in practice — the model learns to use different dimensions for position vs. content information.

**Comparison of positional encoding approaches:**

| Approach | Used by | How it works |
|----------|---------|-------------|
| Learned embeddings (this code) | GPT-2, GPT-3 | Lookup table, one vector per position |
| Sinusoidal (original Transformer) | "Attention Is All You Need" | Fixed sine/cosine waves at different frequencies |
| RoPE (Rotary Position Embedding) | LLaMA, Mistral | Rotates Q and K vectors by position-dependent angles |

Our Baby GPT uses the simplest approach: a learned embedding per position. The limitation is that it cannot handle sequences longer than BLOCK_SIZE (128). RoPE, used by modern models, generalizes better to longer sequences.

---

## Section 7: Feed-Forward Network (lines 126-139)

```python
class FeedForward(nn.Module):
    """Two-layer MLP with ReLU — processes each position independently."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBED_DIM, FF_DIM),     # 128 -> 512 (expand)
            nn.ReLU(),                         # non-linearity
            nn.Linear(FF_DIM, EMBED_DIM),     # 512 -> 128 (compress back)
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)
```

**Theory — what does the feed-forward network do?**

Attention lets tokens gather information from each other. The feed-forward network (FFN) then **processes** that gathered information at each position independently.

Think of it as a two-step process in each transformer block:
1. **Attention:** "Let me look at the context and gather relevant information"
2. **Feed-forward:** "Now let me think about what I gathered and update my understanding"

**Why expand then compress (128 -> 512 -> 128)?**

The expansion to 4x the dimension (512 = 4 * 128) gives the network a larger space to process information. The ReLU activation then selectively zeros out dimensions, acting as a learned "routing" mechanism. The compression back to 128 forces the network to distill the result.

Research has shown that FFN layers act like key-value memories — different neurons activate for different patterns and "store" knowledge about language.

**Why process each position independently?** The FFN applies the same computation to every position. Cross-position communication already happened in the attention layer. This separation is what makes transformers so parallelizable.

---

## Section 8: Residual Connections & Layer Norm (lines 142-156)

```python
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.attn = MultiHeadAttention()
        self.ln2 = nn.LayerNorm(EMBED_DIM)
        self.ff = FeedForward()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # residual connection around attention
        x = x + self.ff(self.ln2(x))     # residual connection around feed-forward
        return x
```

### Residual Connections (the `x + ...` pattern)

**The problem:** Deep networks suffer from vanishing gradients. By the time the gradient signal travels back through 4+ layers, it can become tiny — early layers barely learn.

**The solution:** Skip connections. Instead of `x = f(x)`, we do `x = x + f(x)`. The gradient flows through both the function AND the direct path (the "+"). Even if `f` has small gradients, the identity path (`x`) always has gradient = 1.

```
Without residual:     x -> [attention] -> [feed-forward] -> output
                      Gradient must flow through every layer (can vanish)

With residual:        x ----+----> [attention] ----+----> [feed-forward] ----+--->
                            |                      |                         |
                            +--- direct path ------+--- direct path ---------+
                      Gradient has a "highway" that bypasses layers
```

**Analogy:** Imagine passing a message through 4 people in a chain. Without residuals, each person rephrases the message (information lost). With residuals, each person adds a post-it note to the original message (original preserved, annotations added).

### Layer Normalization

```python
self.ln1 = nn.LayerNorm(EMBED_DIM)  # normalizes across the 128-dim embedding
```

**What it does:** For each token at each position, normalize the 128 embedding values to have mean=0 and std=1. Then apply a learned scale and shift.

**Why?** Without normalization, activation values can drift to extreme ranges as they pass through layers. This makes optimization unstable — the loss landscape becomes jagged. LayerNorm keeps values in a well-behaved range, smoothing the loss landscape.

**Pre-norm vs post-norm:**

| Style | Order | Used by |
|-------|-------|---------|
| Post-norm (original Transformer) | attention -> add -> norm | Original 2017 paper |
| Pre-norm (this code, GPT-2) | norm -> attention -> add | GPT-2, most modern models |

Our code uses **pre-norm** (normalize before the sub-layer). This is more stable during training and is what GPT-2 introduced. The original transformer paper used post-norm, which can be harder to train without careful learning rate warmup.

---

## Section 9: The Full Model & Forward Pass (lines 162-189)

```python
class BabyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, EMBED_DIM)    # 65 x 128
        self.pos_emb = nn.Embedding(BLOCK_SIZE, EMBED_DIM)      # 128 x 128
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(NUM_LAYERS)])
        self.ln_final = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, vocab_size)            # 128 -> 65
```

### Forward Pass with Tensor Shapes

```python
def forward(self, idx, targets=None):
    B, T = idx.shape
    tok = self.token_emb(idx)                           # (B, T, 128)
    pos = self.pos_emb(torch.arange(T, device=DEVICE))  # (T, 128)
    x = tok + pos                                        # (B, T, 128)
    x = self.blocks(x)                                   # (B, T, 128)
    x = self.ln_final(x)                                 # (B, T, 128)
    logits = self.head(x)                                # (B, T, 65)
```

### Complete Data Flow Diagram

```
Input: idx (B=64, T=128) — batch of integer sequences
  │
  ▼ Token Embedding
  tok (64, 128, 128) — each integer becomes a 128-dim vector
  │
  + Position Embedding
  pos (128, 128) — broadcasts to (64, 128, 128)
  │
  ▼ x = tok + pos (64, 128, 128) — tokens now know their position
  │
  ▼ ═══════════════════════════════════════════════
  ║  Transformer Block 1                          ║
  ║    │                                          ║
  ║    ├─ LayerNorm ──► Multi-Head Attention ─┐   ║
  ║    │                                      │   ║
  ║    └──────── + (residual) ◄───────────────┘   ║
  ║    │                                          ║
  ║    ├─ LayerNorm ──► Feed-Forward ─────────┐   ║
  ║    │                                      │   ║
  ║    └──────── + (residual) ◄───────────────┘   ║
  ║                                               ║
  ▼ ═══════════════════════════════════════════════
  │  ... Blocks 2, 3, 4 (same structure) ...
  │
  ▼ (64, 128, 128) — refined representations
  │
  ▼ Final LayerNorm
  │
  ▼ Linear Head (128 -> 65)
  │
  ▼ logits (64, 128, 65) — 65 scores per position per sequence
```

**Key insight:** The shape stays `(B, T, 128)` through all transformer blocks. Only the final linear layer changes it to `(B, T, 65)` to produce one score per vocabulary character.

---

## Section 10: Cross-Entropy Loss — Characters vs CIFAR Classes (lines 187-188)

```python
loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
```

**Theory — same loss function, different vocabulary:**

| | CIFAR-10 Classifier | Baby GPT |
|---|---|---|
| Task | "Which of 10 classes is this image?" | "Which of 65 characters comes next?" |
| Output shape | (batch, 10) | (batch * seq_len, 65) |
| Target | One class label per image | One character per position |
| Loss function | Cross-entropy | Cross-entropy |
| What it measures | How far predicted class probabilities are from the true class | How far predicted character probabilities are from the true next character |

The `.view(-1, vocab_size)` reshapes from `(64, 128, 65)` to `(8192, 65)` — flattening batch and sequence into one dimension. Each of the 8,192 positions is an independent prediction: "given everything before this position, what character comes next?"

Similarly, targets reshape from `(64, 128)` to `(8192,)` — one correct answer per position.

**Cross-entropy in a nutshell:**
```
Model outputs logits for position i: [2.1, -0.5, 0.3, ..., 1.8]  (65 values)
Softmax converts to probabilities:    [0.15, 0.01, 0.03, ..., 0.12]
True character is 'e' (index 43), model gives it probability 0.15
Loss = -log(0.15) = 1.90

If model had given 'e' probability 0.95:
Loss = -log(0.95) = 0.05  (much lower — model is confident and correct)
```

---

## Section 11: Autoregressive Generation (lines 191-203)

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -BLOCK_SIZE:]           # crop to context window
        logits, _ = self(idx_cond)                 # forward pass
        probs = F.softmax(logits[:, -1, :], dim=-1)  # last position only
        idx_next = torch.multinomial(probs, num_samples=1)  # sample
        idx = torch.cat([idx, idx_next], dim=1)    # append
    return idx
```

**Theory — autoregressive generation:**

The model generates text one character at a time in a loop:

```
Step 1: Input "T"         -> Model predicts next char -> samples "o"
Step 2: Input "To"        -> Model predicts next char -> samples " "
Step 3: Input "To "       -> Model predicts next char -> samples "b"
Step 4: Input "To b"      -> Model predicts next char -> samples "e"
...
Step N: Input "To be or"  -> Model predicts next char -> samples " "
```

**Key details:**

1. **Context window crop** (`idx[:, -BLOCK_SIZE:]`): The model can only see 128 characters at a time. If the generated text exceeds this, we take only the last 128 characters as input. Earlier context is lost.

2. **Last position only** (`logits[:, -1, :]`): The model outputs predictions for every position, but we only care about the last one — that is the prediction for what comes next.

3. **Sampling vs argmax** (`torch.multinomial`): Instead of always picking the most likely character (argmax), we sample from the probability distribution. This adds variety:
   - Argmax: always outputs the same text for a given prompt (deterministic, repetitive)
   - Sampling: different text each time (creative, sometimes surprising)

4. **Append and repeat** (`torch.cat`): The newly sampled character is appended to the sequence, and the loop continues. Each new character can influence all future predictions.

**Temperature (not in this code, but important to know):**

Real models divide logits by a temperature value before softmax:
- Temperature 0.1: very confident, picks likely characters (conservative)
- Temperature 1.0: normal sampling (balanced)
- Temperature 2.0: more random, picks unlikely characters (creative/chaotic)

---

## Section 12: Training Loop (lines 225-269)

```python
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for step in range(1, MAX_ITERS + 1):
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**The training loop is identical to CIFAR-10:** Forward -> Loss -> Backward -> Update.

| | CIFAR-10 | Baby GPT |
|---|---|---|
| Data loading | DataLoader iterates over full dataset per epoch | `get_batch()` samples random windows each step |
| Training unit | Epochs (25 passes over full data) | Iterations (5,000 random batches) |
| LR schedule | CosineAnnealingLR | None (constant LR) |
| Loss function | `nn.CrossEntropyLoss()` | `F.cross_entropy()` (same thing, functional form) |
| Optimizer | Adam | Adam |
| Model saving | Save on best test accuracy | Save on best validation loss |

**Why iterations instead of epochs?** The Shakespeare dataset is small enough that random sampling works well. Each "iteration" grabs a random batch regardless of whether we have seen the full dataset. This is simpler and common in language model training.

### Evaluation (lines 209-222)

```python
@torch.no_grad()
def estimate_loss(model):
    model.eval()
    for split in ["train", "val"]:
        batch_losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            batch_losses[k] = loss.item()
        losses[split] = batch_losses.mean().item()
    model.train()
    return losses
```

**Why average over 200 batches?** Because each batch is randomly sampled, a single batch loss is noisy. Averaging over 200 batches gives a stable estimate of the true loss.

---

## Section 13: Code-to-Concept Mapping

| Concept | Code Location | Class/Function |
|---------|--------------|----------------|
| Character tokenizer | Lines 50-56 | `stoi`, `itos`, `encode`, `decode` |
| Input/target batching | Lines 69-77 | `get_batch()` |
| Query/Key/Value projections | Lines 88-90 | `SelfAttention.__init__` |
| Scaled dot-product attention | Line 102 | `SelfAttention.forward` |
| Causal mask | Lines 93, 103 | `SelfAttention` (register_buffer, masked_fill) |
| Multi-head attention | Lines 110-123 | `MultiHeadAttention` |
| Feed-forward network | Lines 126-139 | `FeedForward` |
| Residual connections | Lines 154-155 | `TransformerBlock.forward` |
| Pre-norm layer normalization | Lines 147-148, 154-155 | `TransformerBlock` |
| Token + position embeddings | Lines 169-170, 177-179 | `BabyGPT.__init__`, `BabyGPT.forward` |
| Final projection to vocab | Line 173, 182 | `self.head` |
| Cross-entropy loss | Line 188 | `BabyGPT.forward` |
| Autoregressive generation | Lines 192-203 | `BabyGPT.generate` |
| Training loop | Lines 243-249 | `main()` |

---

## Section 14: Connection to Real GPT / LLaMA

Our Baby GPT is a real transformer — the same architecture used by GPT-2/3/4 and LLaMA, just much smaller. Here is what stays the same and what changes at scale:

### What is the same

| Component | Baby GPT | Real GPT / LLaMA |
|-----------|----------|-------------------|
| Core architecture | Decoder-only transformer | Decoder-only transformer |
| Attention mechanism | Scaled dot-product with causal mask | Same |
| Residual connections | `x = x + sublayer(norm(x))` | Same |
| Autoregressive generation | Predict next token, sample, append | Same |
| Training objective | Cross-entropy on next token | Same |
| Optimizer | Adam | AdamW (Adam with weight decay) |

### What changes at scale

| Aspect | Baby GPT | GPT-3 / LLaMA 2 |
|--------|----------|------------------|
| Parameters | ~200K | 7B - 175B |
| Embedding dim | 128 | 4,096 - 12,288 |
| Layers | 4 | 32 - 96 |
| Heads | 4 | 32 - 96 |
| Context window | 128 chars | 2,048 - 128,000 tokens |
| Vocabulary | 65 characters | 32,000 - 100,000 subword tokens |
| Tokenizer | Character-level | BPE (Byte Pair Encoding) |
| Activation | ReLU | SwiGLU (LLaMA), GELU (GPT) |
| Position encoding | Learned embeddings | RoPE (LLaMA), learned (GPT) |
| Normalization | LayerNorm | RMSNorm (LLaMA), LayerNorm (GPT) |
| Training data | 1MB Shakespeare | Terabytes of internet text |
| Training time | Minutes on one GPU | Months on thousands of GPUs |
| Post-training | None | RLHF / DPO alignment |

### Key concepts that appear at scale but not in Baby GPT

- **BPE Tokenizer:** Instead of one token per character, common words and subwords get their own tokens. "understanding" might be ["under", "standing"] or even a single token. This dramatically reduces sequence length.

- **KV Cache:** During generation, real models cache the key and value tensors from previous positions to avoid recomputing them. Without this, generation would be O(n^2) per token.

- **RLHF (Reinforcement Learning from Human Feedback):** After pre-training, models like ChatGPT are fine-tuned with human preferences to be helpful, harmless, and honest. This is what makes them conversational.

- **Grouped Query Attention (GQA):** LLaMA 2 shares key/value heads across multiple query heads, reducing memory use during inference.

- **Flash Attention:** An optimized attention implementation that reduces GPU memory reads/writes, making training faster without changing the math.

- **Mixed Precision Training (bf16/fp16):** Real models use 16-bit floats for most computation, halving memory use and doubling throughput.

---

## The Complete Architecture at a Glance

```
"To be or not to"  (input text)
       │
       ▼  encode()
[24, 53, 1, 40, 43, ...]  (integer tokens)
       │
       ▼  token_emb + pos_emb
  (B, T, 128)  token vectors with position info
       │
  ┌────▼────────────────────────────────────────┐
  │    Transformer Block x 4                    │
  │    ┌──────────────────────────────────┐     │
  │    │  LayerNorm                       │     │
  │    │       │                          │     │
  │    │  Multi-Head Attention (4 heads)  │     │
  │    │       │                          │     │
  │    │  + residual ◄────────────────────│     │
  │    │       │                          │     │
  │    │  LayerNorm                       │     │
  │    │       │                          │     │
  │    │  Feed-Forward (128→512→128)      │     │
  │    │       │                          │     │
  │    │  + residual ◄────────────────────│     │
  │    └──────────────────────────────────┘     │
  └─────────────────────────────────────────────┘
       │
       ▼  ln_final
  (B, T, 128)
       │
       ▼  linear head (128 → 65)
  (B, T, 65)  logits — one score per character per position
       │
       ├──► Training: cross-entropy loss against target characters
       │
       └──► Generation: softmax → sample → append → repeat
```

---

## Key Takeaways

1. **Self-attention is the core:** It lets every token look at relevant context, replacing the sequential processing of RNNs.

2. **The architecture is simple:** Embedding + N x (Attention + FFN + Residuals + Norm) + Linear head. That is the whole thing.

3. **Training objective is next-token prediction:** Given context, predict the next character. This simple objective, scaled up with more data and parameters, produces the capabilities we see in modern LLMs.

4. **The training loop is the same as CIFAR:** Forward -> Loss -> Backward -> Update. The architecture inside the model changed (CNN to Transformer), but the learning process did not.

5. **Scale is the main difference:** Baby GPT and GPT-4 share the same fundamental architecture. The difference is ~200K parameters vs. ~1.8 trillion, 65 characters vs. 100K tokens, 1MB of data vs. terabytes.
