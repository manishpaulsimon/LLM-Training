"""Quick verification that the environment is set up correctly."""

import sys

def check(name, test_fn):
    try:
        result = test_fn()
        print(f"  [OK] {name}: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return False

print("Environment Check")
print("=" * 50)
print(f"  Python: {sys.version}")

all_ok = True

all_ok &= check("PyTorch", lambda: __import__("torch").__version__)
all_ok &= check("CUDA available", lambda: __import__("torch").cuda.is_available())
all_ok &= check("GPU name", lambda: __import__("torch").cuda.get_device_name(0))
all_ok &= check("VRAM", lambda: f"{__import__('torch').cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
all_ok &= check("torchvision", lambda: __import__("torchvision").__version__)
all_ok &= check("transformers", lambda: __import__("transformers").__version__)
all_ok &= check("PIL", lambda: __import__("PIL").__version__)
all_ok &= check("matplotlib", lambda: __import__("matplotlib").__version__)
all_ok &= check("pandas", lambda: __import__("pandas").__version__)

print("=" * 50)
print("ALL GOOD!" if all_ok else "Some checks failed — install missing packages.")
