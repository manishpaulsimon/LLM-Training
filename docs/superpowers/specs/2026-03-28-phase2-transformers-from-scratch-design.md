# Phase 2: Transformers from Scratch — Design Spec

## Goal

Build two transformer-based models from scratch to learn the architecture that powers all modern LLMs: a character-level text generator (Baby GPT) and a Vision Transformer (ViT) image classifier.

## Script 1: Baby GPT (`01-fundamentals/02_baby_gpt.py`)

### What it does
Trains a small transformer to predict the next character in Shakespeare text. After training, generates new Shakespeare-like text.

### Dataset
- Shakespeare's complete works (~1.1MB text file)
- Downloaded automatically from a public URL
- Character-level tokenization (each unique char = one token, ~65 tokens)
- Train/val split: 90% / 10%

### Architecture
- Character embedding: vocab_size → embed_dim (64)
- Positional encoding: max_seq_len (128) → embed_dim (64)
- 4 transformer blocks, each containing:
  - Multi-head self-attention (4 heads) with causal mask
  - Feed-forward network (2 layers with ReLU, inner dim 256)
  - Layer norm + residual connections
- Output head: embed_dim → vocab_size (next-char prediction)
- ~2-3M parameters

### Training
- Cross-entropy loss on next-character prediction
- Adam optimizer, lr=3e-4
- ~5000 iterations on random sequence chunks
- Generates sample text every 500 steps
- Saves best model checkpoint

### Code structure
- Swappable dataset: loads from `01-fundamentals/data/input.txt`, can be replaced with any text file

## Script 2: Vision Transformer (`01-fundamentals/03_vision_transformer.py`)

### What it does
Classifies CIFAR-10 images using a transformer instead of CNN. Same dataset as Phase 1B.

### Architecture
- Patch embedding: split 32x32 image into 16 patches of 8x8, project each to embed_dim (128)
- CLS token: learnable token prepended to patch sequence
- Positional encoding: 17 positions (16 patches + CLS) → embed_dim
- 6 transformer blocks (same structure as Baby GPT but bidirectional — no causal mask)
- Classification head: CLS token output → 10 classes
- ~3-4M parameters

### Training
- Same CIFAR-10 dataset/transforms as Phase 1B
- Cross-entropy loss, Adam optimizer
- 25 epochs, prints train/test loss+accuracy per epoch
- Saves best model and training curves plot

## Theory Docs
- `docs/02-baby-gpt-theory.md` — attention mechanism, positional encoding, autoregressive generation, transformer block anatomy
- `docs/03-vision-transformer-theory.md` — image patches, CLS token, ViT vs CNN, bidirectional vs causal attention

## Comment style
"Explain the why, not the what" — comments on concepts and non-obvious parts, clean readable code.
