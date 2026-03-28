# Phase 1B: Vision Transformer (ViT) — Theory & Code Walkthrough

> Script: `01-fundamentals/03_vision_transformer.py`
> Run: `python 01-fundamentals/03_vision_transformer.py`

---

## Why Apply Transformers to Images?

You already built two things:
1. A **CNN** that classifies CIFAR-10 images using convolutional filters (script 01)
2. A **Baby GPT** that generates text using transformer attention (script 02)

The Vision Transformer asks: **what if we skip convolutions entirely and use the same attention mechanism from Baby GPT to understand images?**

CNNs have a built-in assumption (inductive bias): nearby pixels matter more than distant ones. That is why they use small 3x3 filters that slide locally. Transformers have no such assumption — every element can attend to every other element from the start. This is both ViT's weakness on small data and its strength on large data, which we will unpack later.

The key insight of the 2020 ViT paper (Dosovitskiy et al.) is simple: **chop an image into patches, treat each patch like a word token, and feed the sequence to a standard transformer.** That is exactly what our code does.

---

## Section 1: Hyperparameters (lines 24-37)

```python
PATCH_SIZE = 8            # split 32x32 image into 8x8 patches -> 16 patches
EMBED_DIM = 128           # dimension of each patch/token embedding
NUM_HEADS = 4
NUM_LAYERS = 6
FF_DIM = 512
DROPOUT = 0.1
LEARNING_RATE = 3e-4
EPOCHS = 25

NUM_PATCHES = (32 // PATCH_SIZE) ** 2  # 16 patches for 32x32 with 8x8 patches
```

**Theory — Patch size controls the sequence length:**

| Patch size | Patches per image | Sequence length (+ CLS) | Computation |
|------------|------------------|------------------------|-------------|
| 16x16 | 4 | 5 | Very fast, low resolution |
| **8x8** | **16** | **17** | **Good balance for 32x32 images** |
| 4x4 | 64 | 65 | Slow, high resolution |

Attention is O(n^2) in sequence length, so smaller patches = quadratically more compute. For CIFAR-10's tiny 32x32 images, 8x8 patches give us 16 tokens — a manageable sequence.

Compare to Baby GPT: there your sequence length was the context window of characters. Here your sequence length is the number of image patches. **Same transformer, different "words."**

---

## Section 2: Patch Embedding — How Images Become Sequences (lines 142-178)

This is the core idea that makes ViT work. A transformer expects a sequence of vectors. An image is a 3D grid of pixels. Patch embedding bridges the gap.

### Step 1: Chop the image into patches

```python
# (B, 3, 32, 32) -> unfold into patches
patches = x.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
patches = patches.contiguous().view(B, 3, NUM_PATCHES, PATCH_SIZE, PATCH_SIZE)
patches = patches.permute(0, 2, 1, 3, 4).contiguous()  # (B, 16, 3, 8, 8)
patches = patches.view(B, NUM_PATCHES, -1)               # (B, 16, 192)
```

**What `unfold` does visually:**

```
Original 32x32 image (one channel shown):

 ┌────────┬────────┬────────┬────────┐
 │ patch  │ patch  │ patch  │ patch  │
 │  0     │  1     │  2     │  3     │   8px
 ├────────┼────────┼────────┼────────┤
 │ patch  │ patch  │ patch  │ patch  │
 │  4     │  5     │  6     │  7     │   8px
 ├────────┼────────┼────────┼────────┤
 │ patch  │ patch  │ patch  │ patch  │
 │  8     │  9     │  10    │  11    │   8px
 ├────────┼────────┼────────┼────────┤
 │ patch  │ patch  │ patch  │ patch  │
 │  12    │  13    │  14    │  15    │   8px
 └────────┴────────┴────────┴────────┘
   8px      8px      8px      8px

4 x 4 = 16 patches, each 8x8 pixels, 3 color channels
Each patch flattened: 3 * 8 * 8 = 192 values
```

**Shape progression through the unfold:**

```
x                                    (B, 3, 32, 32)    original image
x.unfold(2, 8, 8)                   (B, 3, 4, 32, 8)  split height into 4 strips
x.unfold(2, 8, 8).unfold(3, 8, 8)  (B, 3, 4, 4, 8, 8) split width too -> 4x4 grid
view(B, 3, 16, 8, 8)                (B, 3, 16, 8, 8)  flatten grid to 16 patches
permute(0, 2, 1, 3, 4)              (B, 16, 3, 8, 8)  move patches to dim 1
view(B, 16, -1)                      (B, 16, 192)      flatten each patch to vector
```

**Analogy to text:** In Baby GPT, a sentence like "the cat sat" becomes a sequence of token indices [4, 12, 8], each embedded to a vector. Here, an image becomes a sequence of 16 patch vectors, each 192-dimensional. Same idea — different modality.

### Step 2: Project patches to embedding dimension

```python
patch_dim = 3 * PATCH_SIZE * PATCH_SIZE  # 192
self.patch_embed = nn.Linear(patch_dim, EMBED_DIM)  # 192 -> 128
```

```
x = self.patch_embed(patches)     # (B, 16, 192) -> (B, 16, 128)
```

Each raw 192-dimensional patch is projected down to 128 dimensions. This is the same as a word embedding layer in GPT — it maps raw inputs into the dimension the transformer works in.

**Why project?** The raw patch (192 values of pixel intensities) is not a useful representation. The linear layer learns to compress it into a meaningful 128-dimensional vector where similar patches end up nearby.

---

## Section 3: The CLS Token (lines 149-152, 180-182)

```python
# In __init__:
self.cls_token = nn.Parameter(torch.randn(1, 1, EMBED_DIM))

# In forward:
cls = self.cls_token.expand(B, -1, -1)       # (B, 1, 128)
x = torch.cat([cls, x], dim=1)               # (B, 17, 128)
```

**What is the CLS token?**

CLS (short for "classification") is a learnable vector prepended to the patch sequence. It does not correspond to any part of the image. After passing through the transformer, the CLS token's output is used as the image representation for classification.

```
Before CLS:   [patch_0, patch_1, patch_2, ... patch_15]   16 tokens
After CLS:    [CLS, patch_0, patch_1, patch_2, ... patch_15]   17 tokens
                 ^
                 This token's final output goes to the classifier
```

**Why not just average all patch outputs?**

You could (and some ViT variants do). But CLS has a specific advantage: it forces the transformer to aggregate information from all patches into a single token. It acts as a "summary slot" — the transformer learns to write the global image understanding into this position.

**BERT analogy:** This is exactly how BERT works for text classification. BERT prepends a [CLS] token to the input sentence, runs the transformer, and uses the [CLS] output for the task. ViT borrowed this idea directly. If you later work with BERT-style models, you already understand CLS.

---

## Section 4: Position Embeddings (lines 154, 185-186)

```python
self.pos_emb = nn.Embedding(NUM_PATCHES + 1, EMBED_DIM)  # 17 positions, each 128-dim

positions = torch.arange(NUM_PATCHES + 1, device=DEVICE)  # [0, 1, 2, ..., 16]
x = x + self.pos_emb(positions)                             # (B, 17, 128)
```

**Why position embeddings?**

Attention is permutation-invariant — without positions, the transformer cannot tell if a patch came from the top-left or bottom-right. Patches would be a bag of visual features with no spatial structure.

Position embeddings add a unique learned vector to each position. The transformer learns that position 0 is CLS, position 1 is top-left, position 4 is second-row-left, etc.

**Same idea as Baby GPT:** In your character transformer, you added position embeddings so the model knew which position each character was at. Same principle, same code pattern.

---

## Section 5: Bidirectional vs Causal Attention (lines 70-89)

This is the most important architectural difference between ViT and GPT.

### ViT: Bidirectional attention (every patch sees every patch)

```python
class SelfAttention(nn.Module):
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # No causal mask -- every patch can attend to every other patch
        scores = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        return weights @ v
```

**No mask at all.** Every token attends to every other token, including CLS attending to all patches and all patches attending to CLS.

### Compare: GPT uses causal (masked) attention

In Baby GPT, you had something like:

```python
# GPT masks future tokens so position i can only see positions 0..i
mask = torch.tril(torch.ones(T, T))
scores = scores.masked_fill(mask == 0, float('-inf'))
```

### The two attention patterns visualized

```
ViT ATTENTION (bidirectional)          GPT ATTENTION (causal)

       CLS p0  p1  p2  p3                 t0  t1  t2  t3  t4
  CLS [ ok  ok  ok  ok  ok ]         t0 [ ok   .   .   .   . ]
  p0  [ ok  ok  ok  ok  ok ]         t1 [ ok  ok   .   .   . ]
  p1  [ ok  ok  ok  ok  ok ]         t2 [ ok  ok  ok   .   . ]
  p2  [ ok  ok  ok  ok  ok ]         t3 [ ok  ok  ok  ok   . ]
  p3  [ ok  ok  ok  ok  ok ]         t4 [ ok  ok  ok  ok  ok ]

  ok = can attend                     . = masked (cannot attend)
  Every token sees everything         Each token only sees past + self
```

**Why the difference?**

- **GPT generates text left-to-right.** If token 3 could see token 4, it would be "cheating" — seeing the answer before predicting it. The causal mask enforces this.
- **ViT classifies a complete image.** All patches exist simultaneously. There is no "future" — the entire image is available. Masking would only throw away information.

This is the same distinction as BERT vs GPT in NLP: BERT reads the whole sentence at once (bidirectional, like ViT), GPT reads left-to-right (causal, autoregressive).

---

## Section 6: GELU vs ReLU — Why ViT Uses GELU (lines 105-116)

```python
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBED_DIM, FF_DIM),
            nn.GELU(),       # ViT uses GELU instead of ReLU
            nn.Linear(FF_DIM, EMBED_DIM),
            nn.Dropout(DROPOUT),
        )
```

In the CNN (script 01), you used ReLU. Here we use GELU. Why?

### ReLU: hard cutoff at zero

```
ReLU(x) = max(0, x)

      output
        |      /
        |     /
        |    /
        |   /
   -----+--/---------> input
        | 0
        |
```

ReLU has a sharp corner at x=0. For negative inputs, the gradient is exactly 0 — these neurons are "dead" and contribute nothing to learning.

### GELU: smooth approximation

```
GELU(x) = x * P(X <= x)    where X ~ Normal(0,1)

      output
        |      /
        |     /
        |    /
        |   /
   -----+-~/----------> input
        |~
        |
```

GELU smoothly transitions around zero. Small negative values still get a small non-zero output instead of being hard-killed. The gradient is smooth everywhere.

### Why does smooth matter for transformers?

| Property | ReLU | GELU |
|----------|------|------|
| Gradient at x=0 | Undefined (sharp corner) | Smooth, well-defined |
| Negative inputs | Completely killed (gradient = 0) | Slightly allowed through |
| Dead neurons | Common problem | Rare |
| Used in | CNNs (tradition + works fine) | Transformers (GPT, BERT, ViT) |

Transformers are deeper and more sensitive to gradient flow than CNNs. GELU's smooth gradient helps information flow through many layers without neurons dying. The original transformer papers (BERT, GPT-2, ViT) all adopted GELU, and it became the standard for attention-based architectures.

**Practical impact:** For our small model, GELU vs ReLU might only matter by 1-2% accuracy. At scale (hundreds of layers, billions of parameters), the smoother gradients compound into meaningfully better training dynamics.

---

## Section 7: Transformer Blocks (lines 119-130)

```python
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.attn = MultiHeadAttention()
        self.ln2 = nn.LayerNorm(EMBED_DIM)
        self.ff = FeedForward()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))    # Residual + attention
        x = x + self.ff(self.ln2(x))      # Residual + feedforward
        return x
```

This is the **same** transformer block from Baby GPT with one difference: **Pre-Norm** instead of Post-Norm.

```
Pre-Norm (ViT, modern):         Post-Norm (original transformer):
  x -> LayerNorm -> Attention      x -> Attention -> LayerNorm
     -> + residual                    -> + residual
  x -> LayerNorm -> FFN            x -> FFN -> LayerNorm
     -> + residual                    -> + residual
```

Pre-Norm puts LayerNorm before the attention/FFN. This is more stable for training because the residual path carries un-normalized values, preventing gradient explosion in deep networks.

**The residual connection (`x = x + ...`)** is critical. Without it, gradients would need to flow through 6 attention layers and 6 FFN layers — they would vanish. The `+` provides a "highway" for gradients to flow directly from output to input.

---

## Section 8: The Full Forward Pass with Tensor Shapes (lines 167-194)

Here is the complete data flow through the model, with exact shapes at every step:

```python
def forward(self, x):                                    # x: (B, 3, 32, 32)
    B = x.shape[0]

    # 1. Split into patches and flatten
    patches = x.unfold(2, 8, 8).unfold(3, 8, 8)         # (B, 3, 4, 4, 8, 8)
    patches = patches.contiguous().view(B, 3, 16, 8, 8)  # (B, 3, 16, 8, 8)
    patches = patches.permute(0, 2, 1, 3, 4).contiguous() # (B, 16, 3, 8, 8)
    patches = patches.view(B, 16, -1)                     # (B, 16, 192)

    # 2. Project to embedding dimension
    x = self.patch_embed(patches)                         # (B, 16, 128)

    # 3. Prepend CLS token
    cls = self.cls_token.expand(B, -1, -1)               # (B, 1, 128)
    x = torch.cat([cls, x], dim=1)                       # (B, 17, 128)

    # 4. Add position embeddings
    x = x + self.pos_emb(positions)                       # (B, 17, 128)

    # 5. Pass through 6 transformer blocks
    x = self.blocks(x)                                    # (B, 17, 128)

    # 6. Final layer norm
    x = self.ln_final(x)                                  # (B, 17, 128)

    # 7. Extract CLS token and classify
    cls_output = x[:, 0]                                  # (B, 128)
    return self.head(cls_output)                           # (B, 10)
```

### Visual diagram of the full forward pass

```
Input image
(B, 3, 32, 32)
       |
       v
  +-----------+
  | unfold +  |     Chop 32x32 image into 16 patches of 8x8,
  | flatten   |     flatten each patch to 192 values
  +-----------+
       |
  (B, 16, 192)
       |
       v
  +-----------+
  | Linear    |     Project 192 -> 128
  | patch_emb |
  +-----------+
       |
  (B, 16, 128)
       |
       v
  +-----------+
  | prepend   |     Add learnable CLS token at position 0
  | CLS token |
  +-----------+
       |
  (B, 17, 128)
       |
       v
  +-----------+
  | + pos_emb |     Add position embeddings (17 learned vectors)
  +-----------+
       |
  (B, 17, 128)
       |
       v
  +-----------+
  | Transformer|    6 blocks of: LayerNorm -> MultiHeadAttn -> + residual
  | Block x6   |                 LayerNorm -> FFN -> + residual
  +-----------+
       |
  (B, 17, 128)
       |
       v
  +-----------+
  | LayerNorm |     Final normalization
  +-----------+
       |
  (B, 17, 128)
       |
       v
  +-----------+
  | x[:, 0]   |     Extract CLS token (first position)
  +-----------+
       |
  (B, 128)
       |
       v
  +-----------+
  | MLP head  |     Linear(128,128) -> GELU -> Dropout -> Linear(128,10)
  +-----------+
       |
  (B, 10)            10 class logits
```

**Key observation:** The spatial structure of the image (32x32 grid) is destroyed in step 1 and only preserved through position embeddings. The transformer operates on a flat sequence of 17 vectors — it does not "know" these came from a 2D image. All spatial reasoning must be learned from the position embeddings.

---

## Section 9: Code-to-Concept Mapping

| Concept | Code Location | What It Does |
|---------|---------------|--------------|
| Patch embedding | `VisionTransformer.__init__` lines 144-147, `forward` lines 171-178 | Chops image into patches, projects to embedding dim |
| CLS token | Lines 150-151, 181-182 | Learnable summary token prepended to sequence |
| Position embedding | Lines 154, 185-186 | Adds spatial position info to each token |
| Bidirectional attention | `SelfAttention.forward` lines 80-89 | Q*K^T/sqrt(d) with no mask — all tokens see all tokens |
| Multi-head attention | `MultiHeadAttention` lines 92-102 | 4 parallel attention heads, concatenated and projected |
| GELU activation | `FeedForward` line 110 | Smooth activation function for transformer FFN |
| Pre-norm residual | `TransformerBlock.forward` lines 127-130 | LayerNorm before attention/FFN, residual connections |
| CLS extraction | Line 193 | `x[:, 0]` takes only the CLS token for classification |
| Classification head | Lines 160-165, 194 | MLP that maps CLS embedding to 10 class logits |

---

## Section 10: Why ViT Underperforms CNN on Small Datasets

On CIFAR-10 (50k training images), our CNN from script 01 reaches ~88% accuracy. This ViT typically reaches ~75-80%. Why does the transformer lose?

### The Inductive Bias Tradeoff

**CNNs have strong inductive biases built in:**
1. **Locality** — 3x3 conv filters only look at neighboring pixels. The assumption that nearby pixels are related is baked in from the start.
2. **Translation equivariance** — a conv filter that detects a cat ear works the same in the top-left as in the bottom-right. The network does not need to learn this.
3. **Hierarchical structure** — pooling layers force a coarse-to-fine hierarchy (edges -> parts -> objects).

**ViT has almost no inductive biases:**
1. **No locality** — every patch attends to every other patch from layer 1. It must learn that nearby patches are more relevant.
2. **No translation equivariance** — a pattern at patch 0 and patch 15 look completely different to the model. It must learn this equivalence from data.
3. **No hierarchy** — all layers process the same flat sequence. It must learn to build hierarchical features.

### The dataset size effect

```
                    Accuracy
                       |
  CNN on CIFAR-10 ---> |----*---------
                       |   /     *--------- ViT on ImageNet-21k
  ViT on CIFAR-10 --> |--*    /
                       | /   /
                       |/  /
                       | /
                       |/
                       +----------------------------> Dataset size
                     50k                          14M images

  Small data: CNN wins (inductive biases = free knowledge)
  Large data: ViT wins (fewer biases = higher ceiling)
```

**With 50k images:** The CNN's built-in assumptions give it a head start. It "knows" that neighboring pixels matter before seeing a single image. ViT must learn this from scratch — 50k images is not enough.

**With 14M+ images (ImageNet-21k):** ViT has seen enough data to learn spatial relationships, translation invariance, and hierarchical features on its own. And because it is not constrained by CNN's rigid local-to-global structure, it can learn richer, more flexible representations. ViT-Large achieves 88.5% on ImageNet vs 85.8% for the best CNNs.

**Analogy:** A CNN is like a student given a textbook with diagrams and summaries — they learn fast with limited practice. A ViT is like a student given raw data and no structure — they struggle at first but eventually develop deeper understanding if given enough examples.

---

## Section 11: CNN (Script 01) vs ViT (Script 03) Comparison

Both models are trained on the same CIFAR-10 dataset with the same training loop. Here is how they compare:

| Aspect | CNN (01_cifar_classifier.py) | ViT (03_vision_transformer.py) |
|--------|-----|-----|
| **Architecture** | Conv layers + pooling | Transformer blocks + attention |
| **Core operation** | 3x3 convolution (local) | Self-attention (global) |
| **How it sees the image** | Sliding filters over pixels | Sequence of 16 patch tokens |
| **Spatial structure** | Preserved through all layers | Flattened to 1D, encoded as positions |
| **Inductive bias** | Strong (locality, translation equiv.) | Weak (only position embeddings) |
| **Activation** | ReLU | GELU |
| **Normalization** | BatchNorm | LayerNorm |
| **Classification** | Global avg pool over spatial dims | CLS token output |
| **Parameters** | ~1.1M | ~1.0M (similar) |
| **CIFAR-10 accuracy** | ~88% | ~75-80% |
| **Training speed** | Faster (conv is optimized) | Slower (attention is O(n^2)) |
| **Scales to large data** | Diminishing returns | Keeps improving |

### Key differences in normalization

**BatchNorm (CNN):** normalizes across the batch dimension. For each feature channel, computes mean and std across all images in the batch. Works well for CNNs where each channel has a consistent meaning.

**LayerNorm (ViT):** normalizes across the feature dimension. For each token in each image, computes mean and std across the 128 embedding dimensions. Works better for transformers because sequence elements are more variable than spatial feature maps.

---

## Section 12: Connection to Vision-Language Models (Phase 3 Preview)

Understanding ViT is not just about image classification — it is the foundation for modern vision-language models like GPT-4V, LLaVA, and CLIP.

### How ViT patch embeddings become visual tokens for an LLM

```
                     ViT (what you built)              LLM (what you will build)
                    +-----------------+                +-----------------+
                    |                 |                |                 |
 Image             | Patch Embed     |   Visual       | Transformer     |
 (3, 224, 224) --> | + Transformer   | --> Tokens --> | Decoder         | --> "A cat
                    | blocks          |   (N, D)       | (like GPT)      |     sitting
                    |                 |                |                 |     on..."
                    +-----------------+                +-----------------+
                                                             ^
                                                             |
                                             Text tokens: "Describe this image"
```

Here is the key idea: **In a vision-language model, the ViT does not use the CLS token for classification.** Instead, it takes ALL patch token outputs (not just CLS) and feeds them as "visual tokens" directly into a language model's input sequence.

```
Language model sees this sequence:

[visual_patch_0, visual_patch_1, ..., visual_patch_15, "Describe", "this", "image"]
 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 These come from the ViT encoder (the model you just built!)

The LLM then generates text conditioned on both the visual and text tokens.
```

**What you are building toward:**

| Phase | What You Train | Key Component |
|-------|---------------|---------------|
| 1B - Script 01 | CNN classifier | Convolutions, training loop |
| 1B - Script 02 | Baby GPT | Transformer, attention, text generation |
| **1B - Script 03** | **ViT classifier** | **Patch embedding, CLS, bidirectional attention** |
| Phase 3 | Vision-Language Model | ViT encoder + LLM decoder, connected |

The ViT you built today is the exact "visual encoder" used in production vision-language models. The only differences at scale are:
1. Larger images (224x224 or 384x384 instead of 32x32)
2. More layers and wider embeddings
3. Pre-trained on millions of images before being connected to the LLM

**Every concept you learned here — patch embedding, position encoding, bidirectional attention, CLS tokens — appears directly in state-of-the-art multimodal models.**

---

## The Complete Data Flow (visual summary)

```
CIFAR-10 Dataset (50k images)
       |
       v
  DataLoader (batches of 128)
       |
       v for each batch:
  +--------------------------------------------------+
  |  images [128, 3, 32, 32]                         |
  |       |                                           |
  |       v PATCH + EMBED                             |
  |  unfold + Linear -> [128, 16, 128]               |
  |  prepend CLS      -> [128, 17, 128]              |
  |  + pos embeddings -> [128, 17, 128]              |
  |       |                                           |
  |       v TRANSFORMER (x6 blocks)                   |
  |  LayerNorm -> MultiHeadAttn -> + residual         |
  |  LayerNorm -> FFN (GELU)   -> + residual          |
  |       |                                           |
  |       v CLASSIFY                                  |
  |  CLS token -> MLP head -> [128, 10] (logits)     |
  |       |                                           |
  |       v LOSS                                      |
  |  CrossEntropy(logits, labels) -> single number    |
  |       |                                           |
  |       v BACKWARD                                  |
  |  loss.backward() -> gradients for all params      |
  |       |                                           |
  |       v UPDATE                                    |
  |  optimizer.step() -> nudge all weights            |
  +--------------------------------------------------+
       |
       repeat 391 batches x 25 epochs
       |
       v
  Trained model -> vit_best.pth
```

---

## Key Takeaways

1. **ViT treats images like text.** Chop into patches, embed, add positions, run a transformer. The architecture is almost identical to Baby GPT.

2. **Bidirectional attention is the key difference from GPT.** ViT has no causal mask because all patches exist at once. GPT masks future tokens because it generates left-to-right.

3. **CLS token is a summary slot.** The transformer writes a global image representation into this learnable token, which then goes to the classifier.

4. **ViT trades inductive bias for flexibility.** It needs more data than a CNN to learn the same patterns, but has a higher ceiling when data is abundant.

5. **This is the visual encoder for modern multimodal AI.** The patch embeddings from a ViT become the "visual tokens" that let language models understand images.
