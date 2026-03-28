# Phase 1B: CIFAR-10 Classifier — Theory & Code Walkthrough

> Script: `01-fundamentals/01_cifar_classifier.py`
> Run: `python 01-fundamentals/01_cifar_classifier.py`

---

## Section 1: Imports (lines 11-19)

```python
import torch                            # PyTorch — the ML framework
import torch.nn as nn                   # Neural network building blocks (layers, loss functions)
import torch.optim as optim             # Optimizers (Adam, SGD, etc.)
import torchvision                      # Image datasets and pretrained models
import torchvision.transforms as transforms  # Image preprocessing/augmentation
from torch.utils.data import DataLoader # Feeds data to the model in batches
import matplotlib.pyplot as plt         # Plotting training curves
```

**Theory:** PyTorch is the most popular framework for ML research. Everything is built on **tensors** — multi-dimensional arrays like NumPy but they run on GPU and track gradients for backpropagation.

---

## Section 2: Hyperparameters (lines 25-34)

```python
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 25
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**Theory — What are hyperparameters?**
These are settings YOU choose before training. The model can't learn these — they control HOW the model learns.

| Parameter | What it controls | Analogy |
|-----------|-----------------|---------|
| `BATCH_SIZE = 128` | How many images to process at once | Studying 128 flashcards, then checking answers |
| `LEARNING_RATE = 0.001` | How much to adjust weights after each batch | Step size when walking downhill — too big = overshoot, too small = takes forever |
| `EPOCHS = 25` | How many times to go through ALL training data | Re-reading a textbook 25 times |
| `NUM_WORKERS = 2` | CPU threads preparing the next batch while GPU trains | A helper loading the next set of flashcards while you study |
| `DEVICE = "cuda"` | Run on GPU instead of CPU | GPU = 10-50x faster for matrix math |

**Why 128?** Larger batches = faster training but use more VRAM. 128 is a sweet spot for 8GB VRAM.

**Why 0.001?** This is the default for Adam optimizer. It works well for most problems. You'd lower it (0.0001) for fine-tuning, raise it (0.01) rarely.

---

## Section 3: Data Loading & Augmentation (lines 39-65)

### Training transforms (applied to every training image)

```python
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),       # Pad image by 4px, then randomly crop back to 32x32
    transforms.RandomHorizontalFlip(),           # 50% chance to mirror horizontally
    transforms.ToTensor(),                       # Convert PIL image → PyTorch tensor (0-255 → 0.0-1.0)
    transforms.Normalize(                        # Center values around 0 using dataset statistics
        (0.4914, 0.4822, 0.4465),               # Mean of R, G, B channels across all CIFAR-10
        (0.2470, 0.2435, 0.2616)                # Std dev of R, G, B channels
    ),
])
```

**Theory — Why augment?**
Without augmentation, the model sees the exact same 50,000 images every epoch. It will memorize them (overfitting). Augmentation creates slight variations:

```
Original image of a cat:
  → RandomCrop might shift it 2px left
  → RandomHorizontalFlip might mirror it
  → Now it looks slightly different each epoch
  → Model learns "what makes a cat" not "this exact arrangement of pixels"
```

### Test transforms (no augmentation — we want consistent evaluation)

```python
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
```

**Why no augmentation on test data?** We want to measure real performance. Augmenting test data would give inconsistent results.

### Loading the dataset

```python
train_dataset = torchvision.datasets.CIFAR10(
    root=data_dir, train=True, download=True, transform=train_transform,
)
```

**Theory:** `torchvision.datasets` provides common datasets. `download=True` fetches it the first time (~170MB). The `transform` is applied to every image when it's loaded.

### DataLoader — feeds batches to the model

```python
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
)
```

**Theory — Why batches?**
- Can't fit all 50,000 images in GPU memory at once
- Processing one image at a time is too slow (GPU is designed for parallel math)
- **Batch of 128** = sweet spot. GPU processes all 128 in parallel
- `shuffle=True` — randomize order each epoch so the model doesn't learn the sequence
- `num_workers=2` — CPU threads that prepare the next batch while GPU is training on current batch

**Math:** 50,000 images ÷ 128 per batch = **391 batches per epoch**. Each batch does one forward+backward pass.

---

## Section 4: CNN Architecture (lines 71-106)

### What is a CNN?

A **Convolutional Neural Network** learns to detect visual patterns by sliding small filters across an image. Early layers detect simple patterns (edges, colors), deeper layers combine them into complex patterns (eyes, wheels, wings).

### The model definition

```python
class CIFAR10Net(nn.Module):        # All PyTorch models inherit from nn.Module
    def __init__(self):
        super().__init__()          # Initialize the parent class
```

**Theory:** `nn.Module` is the base class for all neural networks in PyTorch. It handles:
- Tracking all learnable parameters
- Moving the model to GPU (`.to(DEVICE)`)
- Switching between train/eval mode

### Conv block — the repeating building block

```python
def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),   # Convolution: detect patterns
        nn.BatchNorm2d(out_c),                    # Normalize: stabilize training
        nn.ReLU(inplace=True),                    # Activate: introduce non-linearity
        nn.Conv2d(out_c, out_c, 3, padding=1),   # Another convolution: detect more complex patterns
        nn.BatchNorm2d(out_c),                    # Normalize again
        nn.ReLU(inplace=True),                    # Activate again
        nn.MaxPool2d(2),                          # Shrink spatial dimensions by half
    )
```

**Theory — each layer explained:**

**`nn.Conv2d(in_c, out_c, 3, padding=1)`** — Convolution
- Slides a 3x3 filter across the image
- `in_c` = input channels (3 for RGB, 64/128 for deeper layers)
- `out_c` = output channels (number of different patterns to detect)
- `padding=1` = add 1px border so output size matches input size
- Each filter learns to detect one pattern (edge, curve, texture...)

```
Example: Conv2d(3, 64, 3) on a 32x32 RGB image
  → 64 different 3x3 filters, each scanning the whole image
  → Output: 64 "feature maps" of size 32x32
  → Each feature map highlights where one pattern was found
```

**`nn.BatchNorm2d(out_c)`** — Batch Normalization
- Normalizes the output of each channel to mean=0, std=1
- **Why?** Without this, values can explode or vanish as they pass through layers
- Makes training faster and more stable
- Like re-centering a scale to zero between measurements

**`nn.ReLU(inplace=True)`** — Rectified Linear Unit
- `f(x) = max(0, x)` — keeps positive values, zeros out negatives
- **Why?** Without activation functions, stacking layers would just be one big linear function (useless). ReLU introduces **non-linearity**, letting the network learn complex patterns
- `inplace=True` = saves memory by modifying values directly

**`nn.MaxPool2d(2)`** — Max Pooling
- Takes every 2x2 block, keeps only the maximum value
- Shrinks 32x32 → 16x16, 16x16 → 8x8, etc.
- **Why?** Reduces computation and makes the model focus on strongest activations
- Also gives slight position invariance (cat shifted by 1px still activates same neuron)

### Stacking three conv blocks

```python
self.features = nn.Sequential(
    conv_block(3, 64),     # Block 1: 3 channels (RGB) → 64 feature maps, 32x32 → 16x16
    conv_block(64, 128),   # Block 2: 64 → 128 feature maps, 16x16 → 8x8
    conv_block(128, 256),  # Block 3: 128 → 256 feature maps, 8x8 → 4x4
)
```

**Theory — the hierarchy of learning:**
```
Block 1 (3→64, 32x32→16x16):   learns edges, colors, simple textures
Block 2 (64→128, 16x16→8x8):   learns shapes, patterns, object parts
Block 3 (128→256, 8x8→4x4):    learns high-level features (wheels, wings, fur)
```

Each block doubles the channels (more patterns to detect) and halves the spatial size (more abstract, less position-specific).

### The classifier head

```python
self.classifier = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),  # Average all 4x4 values into single value per channel → 256x1x1
    nn.Flatten(),              # Reshape from 256x1x1 → flat vector of 256
    nn.Dropout(0.3),           # Randomly zero 30% of values during training
    nn.Linear(256, 10),        # Final layer: 256 features → 10 class scores
)
```

**`AdaptiveAvgPool2d(1)`** — Global Average Pooling
- Each of the 256 feature maps is a 4x4 grid. This averages each into a single number.
- Output: a vector of 256 values, each representing "how much of pattern X was found in the image"

**`Dropout(0.3)`** — Regularization
- During training: randomly zeros 30% of the 256 values
- During evaluation: does nothing (all values used)
- **Why?** Prevents the model from relying on any single feature too much. Forces it to learn redundant representations. Key defense against overfitting.

**`Linear(256, 10)`** — The final decision
- Takes 256 features → outputs 10 scores (one per class)
- The class with the highest score is the prediction
- These 10 raw scores are called **logits**

### The forward pass

```python
def forward(self, x):         # x = batch of images, shape: [128, 3, 32, 32]
    x = self.features(x)     # After conv blocks: [128, 256, 4, 4]
    x = self.classifier(x)   # After classifier:  [128, 10]
    return x                  # 10 scores per image
```

**Theory:** `forward()` defines the path data takes through the network. PyTorch calls this automatically when you do `model(images)`. The shape transforms:

```
[128, 3, 32, 32]    128 images, 3 color channels, 32x32 pixels
  → features →
[128, 256, 4, 4]    128 images, 256 feature maps, 4x4 spatial
  → classifier →
[128, 10]           128 images, 10 class scores each
```

---

## Section 5: Training Loop (lines 112-139)

```python
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()               # Enable training mode (dropout active, batchnorm updates)
    running_loss = 0.0
    correct = 0
    total = 0
```

**`model.train()`** — switches on training behaviors:
- Dropout randomly zeros values
- BatchNorm updates its running statistics

### The batch loop — where learning happens

```python
    for images, labels in loader:     # Loop through all 391 batches
        images, labels = images.to(DEVICE), labels.to(DEVICE)  # Move to GPU
```

**Step 1: Forward pass**
```python
        outputs = model(images)           # Feed 128 images → get 128x10 scores
        loss = criterion(outputs, labels) # Compare predictions to true labels
```

**Theory — Cross-Entropy Loss:**
- `outputs` = 10 raw scores (logits) per image, e.g., `[2.1, -0.5, 0.3, ...]`
- `labels` = correct class index, e.g., `3` (cat)
- Cross-entropy converts logits to probabilities (softmax), then measures how far the predicted probability distribution is from the true answer
- If model says 90% cat and it IS a cat → low loss
- If model says 10% cat and it IS a cat → high loss
- The loss is a single number that summarizes "how wrong was the model on this batch"

**Step 2: Backward pass (backpropagation)**
```python
        optimizer.zero_grad()    # Clear gradients from previous batch
        loss.backward()          # Compute gradient of loss w.r.t. every parameter
        optimizer.step()         # Update every parameter using its gradient
```

**Theory — this is the core of all neural network training:**

1. **`zero_grad()`** — PyTorch accumulates gradients by default. We clear them so each batch starts fresh.

2. **`loss.backward()`** — **Backpropagation.** Starting from the loss, it works backward through every layer computing: "if I nudge this weight up slightly, does the loss go up or down, and by how much?" This gradient tells each weight which direction to move.

3. **`optimizer.step()`** — **Adam optimizer** uses the gradients to update all 1.1M parameters. Adam is smarter than basic gradient descent:
   - Tracks momentum (keeps moving in a consistent direction)
   - Adapts learning rate per-parameter (frequent updates get smaller steps)
   - `weight_new = weight_old - learning_rate * gradient` (simplified)

**Analogy:** You're blindfolded on a hilly landscape, trying to find the lowest valley.
- `loss.backward()` = feeling the slope under your feet (which direction is downhill?)
- `optimizer.step()` = taking one step downhill
- Repeat 391 times per epoch, 25 epochs = ~9,775 steps total

**Step 3: Track metrics**
```python
        running_loss += loss.item() * images.size(0)   # Accumulate total loss
        _, predicted = outputs.max(1)                    # Pick class with highest score
        total += labels.size(0)                          # Count images seen
        correct += predicted.eq(labels).sum().item()     # Count correct predictions
```

- `outputs.max(1)` returns `(max_values, indices)` — we only want the indices (predicted classes)
- `.item()` converts a single-element tensor to a Python number

---

## Section 6: Evaluation Loop (lines 145-164)

```python
@torch.no_grad()                    # Disable gradient computation (saves memory + speed)
def evaluate(model, loader, criterion):
    model.eval()                    # Disable dropout, freeze batchnorm statistics
```

**Theory — Train vs Eval mode:**

| | `model.train()` | `model.eval()` |
|---|---|---|
| Dropout | Randomly zeros 30% of values | Does nothing (all values used) |
| BatchNorm | Updates running mean/std | Uses fixed mean/std from training |
| Gradients | Computed (for backprop) | Skipped via `@torch.no_grad()` |

**Why `@torch.no_grad()`?** During evaluation we're just measuring, not learning. Skipping gradient computation uses ~50% less memory and is faster.

The rest is identical to training but **without the backward pass** — we just measure loss and accuracy on the test set.

---

## Section 7: Main Script (lines 170-234)

### Model setup

```python
model = CIFAR10Net().to(DEVICE)    # Create model and move all parameters to GPU
total_params = sum(p.numel() for p in model.parameters())  # Count total parameters
# → 1,149,770 parameters
```

### Loss function and optimizer

```python
criterion = nn.CrossEntropyLoss()   # The loss function (measures how wrong predictions are)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # The optimizer (updates weights)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)  # LR schedule
```

**Theory — CosineAnnealingLR:**
```
Epoch 1:  lr = 0.001    (big steps — explore the landscape)
Epoch 13: lr = 0.0005   (medium steps — narrowing in)
Epoch 25: lr = 0.0      (tiny steps — fine-tuning the minimum)
```
Follows a cosine curve from max to min. Like running fast when far from the finish, slowing to a walk as you approach it.

### The epoch loop

```python
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    scheduler.step()   # Reduce learning rate according to cosine schedule
```

### Saving the best model

```python
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "cifar10_best.pth")
```

**Theory:** `model.state_dict()` is a dictionary of all parameter names → values. Saving it lets you reload the best model later without retraining. We save whenever test accuracy improves — so the final file has the weights from the best epoch, not the last epoch.

### Training curves plot (lines 215-234)

Plots loss and accuracy over epochs. **What healthy curves look like:**
- Loss: both lines go down, train slightly lower than test
- Accuracy: both lines go up, train slightly higher than test
- If train keeps improving but test plateaus/gets worse → **overfitting**

---

## The Complete Data Flow (visual summary)

```
CIFAR-10 Dataset (50k images)
       │
       ▼
  DataLoader (batches of 128)
       │
       ▼ for each batch:
  ┌────────────────────────────────────────────────┐
  │  images [128, 3, 32, 32]                       │
  │       │                                         │
  │       ▼ FORWARD                                 │
  │  Conv Block 1 → [128, 64, 16, 16]             │
  │  Conv Block 2 → [128, 128, 8, 8]              │
  │  Conv Block 3 → [128, 256, 4, 4]              │
  │  AvgPool+Linear → [128, 10] (logits)          │
  │       │                                         │
  │       ▼ LOSS                                    │
  │  CrossEntropy(logits, labels) → single number  │
  │       │                                         │
  │       ▼ BACKWARD                                │
  │  loss.backward() → gradients for all 1.1M params│
  │       │                                         │
  │       ▼ UPDATE                                  │
  │  optimizer.step() → nudge all weights           │
  └────────────────────────────────────────────────┘
       │
       repeat 391 batches × 25 epochs = 9,775 updates
       │
       ▼
  Trained model → cifar10_best.pth
```

---

## Training Output: What Each Column Means

```
Epoch | Train Loss | Train Acc | Test Loss | Test Acc | Time
```

| Column | What It Means | Good Direction |
|--------|---------------|----------------|
| **Epoch** | One full pass through all 50,000 training images | Goes up 1-25 |
| **Train Loss** | How wrong the model is on training data (cross-entropy) | Lower is better |
| **Train Acc** | % of training images classified correctly | Higher is better |
| **Test Loss** | How wrong the model is on 10,000 **unseen** images | Lower is better |
| **Test Acc** | % correct on unseen images — **the number that matters most** | Higher is better |
| **Time** | Seconds per epoch (first epoch slower due to setup overhead) | ~40s on RTX 3070 |

---

## Key Concepts

### Overfitting vs Underfitting
- **Overfitting:** Train Acc >> Test Acc (e.g., 95% vs 75%). Model memorized training data.
  - Defenses in this code: Dropout (line 99), data augmentation (lines 39-44)
- **Underfitting:** Both accuracies are low. Model isn't powerful enough or needs more training.

### Gradient Descent (the whole point)
The model starts with random weights → makes terrible predictions → loss is high → backward pass computes which direction to nudge each weight → optimizer nudges them → predictions get slightly better → repeat.

Over 9,775 updates, the model goes from random guessing (10% accuracy) to ~88% accuracy.

### Parameters vs Hyperparameters
- **Parameters** (1.1M): learned by the model during training (conv filter values, linear weights)
- **Hyperparameters** (batch size, lr, epochs): chosen by you before training

---

## How This Connects to LLM Training

| This Script | LLM Training |
|-------------|--------------|
| CNN architecture | Transformer architecture |
| 1.1M parameters | Billions of parameters |
| Image pixels as input | Text tokens as input |
| Cross-entropy over 10 classes | Cross-entropy over 50k+ vocabulary tokens |
| "Is this a cat or dog?" | "What word comes next?" |
| Same training loop | **Same training loop** |
| Adam optimizer | AdamW optimizer (similar) |
| 25 epochs, minutes | 1 epoch, weeks on GPU clusters |
| `model(images)` → class scores | `model(tokens)` → next-token probabilities |

**The training loop is identical.** Forward → Loss → Backward → Update. Everything you learn here applies directly to training language models. The only difference is the architecture inside the model and the scale.
