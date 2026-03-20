# Vastu Compliance AI + BIM Drawing Analyzer

AI-powered tools for architectural analysis:

1. **Vastu Scorer** — Takes a floor plan image → outputs Vastu compliance score + specific violations
2. **BIM/Drawing Analyzer** — Takes architectural screenshots → checks drawing correctness

## Project Structure

```
├── 01-fundamentals/        # Learning projects (PyTorch, ViT, GPT from scratch)
├── 02-vastu-model/          # Vastu compliance scoring model
│   ├── data/
│   │   ├── raw/             # Floor plan images
│   │   ├── labeled/         # Annotated training data
│   │   └── synthetic/       # Generated training data
│   ├── scripts/
│   ├── notebooks/
│   └── models/
├── 03-bim-analyzer/         # BIM drawing checker
├── verify_setup.py          # Environment verification
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python verify_setup.py
```

## Hardware

- NVIDIA RTX 3060 Laptop (6GB VRAM)
- Uses 4-bit quantization + LoRA for fine-tuning within VRAM limits

## Progress

- [x] Phase 1A: Environment setup
- [ ] Phase 1B: CIFAR-10 classifier (PyTorch fundamentals)
- [ ] Phase 1C: Data collection & labeling schema
- [ ] Phase 2: Vision Transformer + Baby GPT from scratch
- [ ] Phase 3: Vision-Language Model
- [ ] Phase 4: Vastu scoring model
- [ ] Phase 5: BIM analyzer
- [ ] Phase 6: Local web app
