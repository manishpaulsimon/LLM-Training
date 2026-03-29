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

## Offline Codex with Ollama

Use Codex's built-in local OSS provider mode with Ollama if you want to run with a local model and no internet dependency during inference.

### One-time shell session

```powershell
& "E:\_LLM-Training\.venv\Scripts\Activate.ps1"
codex --oss --local-provider ollama --model qwen2.5-coder:7b
```

Check your installed local model names with:

```powershell
ollama list
```

Then replace `qwen2.5-coder:7b` with the exact tag you have locally, such as `qwen2.5:14b` or `llama3.1:8b`.

### PowerShell functions + aliases

Add this to your PowerShell profile if you want shortcut commands with the model name embedded in the alias:

```powershell
function Start-CodexQwenCoder7B {
    codex --oss --local-provider ollama --model qwen2.5-coder:7b @Args
}

function Start-CodexQwen14B {
    codex --oss --local-provider ollama --model qwen2.5:14b @Args
}

Set-Alias clq7b Start-CodexQwenCoder7B
Set-Alias clq14b Start-CodexQwen14B
```

### One copy-paste profile install

If you want a single command that installs both aliases into your PowerShell profile, run:

```powershell
if (!(Test-Path $PROFILE)) { New-Item -ItemType File -Path $PROFILE -Force | Out-Null }; Add-Content -Path $PROFILE -Value @'

function Start-CodexQwenCoder7B {
    codex --oss --local-provider ollama --model qwen2.5-coder:7b @Args
}

function Start-CodexQwen14B {
    codex --oss --local-provider ollama --model qwen2.5:14b @Args
}

Set-Alias clq7b Start-CodexQwenCoder7B
Set-Alias clq14b Start-CodexQwen14B
'@
```

Then reload your PowerShell profile:

```powershell
. $PROFILE
```

After reloading PowerShell, run:

```powershell
clq7b
clq14b
```

To use different local models, change the `--model` value inside each function.

### Temporary current-session functions

If you do not want to edit your profile yet, run this once in the current PowerShell session:

```powershell
function Start-CodexQwenCoder7B {
    codex --oss --local-provider ollama --model qwen2.5-coder:7b @Args
}

function Start-CodexQwen14B {
    codex --oss --local-provider ollama --model qwen2.5:14b @Args
}

Set-Alias clq7b Start-CodexQwenCoder7B
Set-Alias clq14b Start-CodexQwen14B
```

### Offline capabilities

With this setup, Codex uses Ollama on your machine for model inference, so normal coding actions can stay local:

- read and edit files
- create `README.md` files
- create folders
- run local shell commands
- inspect and modify code in the workspace

Limits:

- web browsing, package downloads, and API calls still need network access
- tool use depends on the Codex CLI session and the local model following instructions reliably
- weaker local models may be less dependable than hosted models for multi-step tool use

If you want fully local usage, keep your models, repo, dependencies, and docs available on disk before starting the session.

If you see `405 Method Not Allowed` against `/v1/responses`, the session is likely using the deprecated `OPENAI_BASE_URL` route instead of Codex's local Ollama mode. Use `--oss --local-provider ollama` in the command or alias.

## Offline OpenCode with Ollama

`opencode` is also installed locally and can be run against Ollama models in this repo.

### Launch commands

Start OpenCode in this repo with an explicit Ollama model:

```powershell
opencode -m ollama/qwen2.5-coder:7b .
```

```powershell
opencode -m ollama/qwen2.5:14b .
```

Check your local models first if needed:

```powershell
ollama list
```

### OpenCode shortcodes

Add these PowerShell functions and aliases if you want short model-switch commands:

```powershell
function Start-OpenCodeQwenCoder7B {
    opencode -m ollama/qwen2.5-coder:7b .
}

function Start-OpenCodeQwen14B {
    opencode -m ollama/qwen2.5:14b .
}

function Start-OpenCodeMistral7B {
    opencode -m ollama/dolphin-mistral:7b .
}

Set-Alias ocq7b Start-OpenCodeQwenCoder7B
Set-Alias ocq14b Start-OpenCodeQwen14B
Set-Alias ocm7b Start-OpenCodeMistral7B
```

Then run:

```powershell
ocq7b
ocq14b
ocm7b
```

Run those commands from the normal PowerShell terminal, not from inside the OpenCode chat box.

### Important usage note

OpenCode has two different contexts:

- PowerShell terminal: use this to launch OpenCode with commands like `opencode -m ollama/qwen2.5-coder:7b .` or `ocq7b`
- OpenCode prompt box: use this only for natural-language instructions after the UI has opened

Do not paste shell commands like `& E:\_LLM-Training\.venv\Scripts\Activate.ps1` into the OpenCode prompt box. That input is treated as a prompt, not executed as a terminal command.

In practice, the most reliable startup path is to launch with an explicit model:

```powershell
opencode -m ollama/qwen2.5-coder:7b .
```

or:

```powershell
ocq7b
```

### Tool calling note

If OpenCode starts with `dolphin-mistral:7b`, you may see:

```text
registry.ollama.ai/library/dolphin-mistral:7b does not support tools
```

That means the active model cannot do tool calling reliably, so file creation, folder creation, and shell-driven edits will not work through the agent.

Use `qwen2.5-coder:7b` or another Ollama model that supports tools instead:

```powershell
opencode -m ollama/qwen2.5-coder:7b .
```

### Prompt guidance for local runs

When using local models, be explicit about staying offline and local-only:

```text
Only use local filesystem and local shell tools. Do not use web, GitHub, search, or remote connectors. Scan this repo and create missing README.md files for important folders.
```

## Ollama Cloud Models

As of 2026-03-29, these are useful Ollama cloud-capable models to consider for OpenCode and coding-agent workflows.

Before using cloud models, sign in to Ollama:

```powershell
ollama signin
```

Then launch OpenCode with one of these model tags:

```powershell
ollama launch opencode --model glm-4.7:cloud
ollama launch opencode --model minimax-m2.1:cloud
ollama launch opencode --model minimax-m2.5:cloud
ollama launch opencode --model gpt-oss:120b-cloud
ollama launch opencode --model qwen3-coder-next:cloud
ollama launch opencode --model qwen3-coder:480b-cloud
ollama launch opencode --model deepseek-v3.2:cloud
ollama launch opencode --model kimi-k2:cloud
ollama launch opencode --model kimi-k2-thinking:cloud
ollama launch opencode --model kimi-k2.5:cloud
ollama launch opencode --model mistral-large-3:cloud
ollama launch opencode --model devstral-2:cloud
ollama launch opencode --model nemotron-3-super:cloud
ollama launch opencode --model glm-5:cloud
ollama launch opencode --model cogito-2.1:cloud
ollama launch opencode --model rnj-1:cloud
```

Recommended starting points:

- `glm-4.7:cloud` — recommended by Ollama's OpenCode integration docs
- `minimax-m2.1:cloud` or `minimax-m2.5:cloud` — strong coding and agentic workflow options
- `qwen3-coder-next:cloud` — coding-focused model optimized for agentic coding
- `gpt-oss:120b-cloud` — strong large-model fallback
- `devstral-2:cloud` — coding-agent oriented model

Notes:

- Cloud models are not offline. They use Ollama's cloud service.
- Model availability can change, so verify current options in Ollama's library if a tag stops working.
- `glm-4.7:cloud` is the current recommended OpenCode cloud model in Ollama's integration docs.

## Hardware

- NVIDIA RTX 3060 Laptop (6GB VRAM)
- Uses 4-bit quantization + LoRA for fine-tuning within VRAM limits

## Progress

- [x] Phase 1A: Environment setup
- [x] Phase 1B: CIFAR-10 classifier (PyTorch fundamentals)
- [ ] Phase 1C: Data collection & labeling schema
- [x] Phase 2: Vision Transformer + Baby GPT from scratch
- [ ] Phase 3: Vision-Language Model
- [ ] Phase 4: Vastu scoring model
- [ ] Phase 5: BIM analyzer
- [ ] Phase 6: Local web app
