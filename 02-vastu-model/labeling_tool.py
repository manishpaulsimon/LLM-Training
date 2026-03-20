"""
Vastu Floor Plan Labeling Tool
===============================
Shows each floor plan image and asks you to label Vastu compliance rules.
Saves results to a CSV file for training.

Run:
    python 02-vastu-model/labeling_tool.py

Place your floor plan images in: 02-vastu-model/data/raw/
"""

import os
import csv
import sys
from pathlib import Path

try:
    from PIL import Image
    import matplotlib.pyplot as plt
except ImportError:
    print("Install dependencies first: pip install pillow matplotlib")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Vastu Rules & Directions
# ---------------------------------------------------------------------------
DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "CENTER", "UNKNOWN"]

VASTU_RULES = [
    {
        "id": "entrance_dir",
        "question": "Main entrance direction?",
        "type": "direction",
        "ideal": ["N", "E", "NE"],
        "description": "North, East, or NE entrances are preferred",
    },
    {
        "id": "kitchen_placement",
        "question": "Kitchen placement?",
        "type": "direction",
        "ideal": ["SE"],
        "description": "Southeast is the ideal kitchen location",
    },
    {
        "id": "master_bedroom",
        "question": "Master bedroom placement?",
        "type": "direction",
        "ideal": ["SW"],
        "description": "Southwest is the ideal master bedroom location",
    },
    {
        "id": "toilet_placement",
        "question": "Toilet/bathroom placement?",
        "type": "direction",
        "ideal": ["NW", "W"],
        "description": "Northwest or West is ideal for toilets",
    },
    {
        "id": "pooja_room",
        "question": "Pooja/prayer room placement? (NONE if absent)",
        "type": "direction_optional",
        "ideal": ["NE"],
        "description": "Northeast is the ideal pooja room location",
    },
    {
        "id": "living_room",
        "question": "Living room placement?",
        "type": "direction",
        "ideal": ["N", "NE", "E"],
        "description": "North, NE, or East is ideal for living room",
    },
    {
        "id": "staircase",
        "question": "Staircase placement? (NONE if absent)",
        "type": "direction_optional",
        "ideal": ["S", "SW", "W"],
        "description": "South, SW, or West is ideal for staircases",
    },
    {
        "id": "open_space",
        "question": "More open space / courtyard direction? (NONE if absent)",
        "type": "direction_optional",
        "ideal": ["N", "E", "NE"],
        "description": "Open space in N, E, or NE is preferred",
    },
    {
        "id": "overall_shape",
        "question": "Is the plan roughly square/rectangular? (yes/no)",
        "type": "yesno",
        "ideal": "yes",
        "description": "Square or rectangular shapes are preferred",
    },
    {
        "id": "notes",
        "question": "Any additional notes? (press Enter to skip)",
        "type": "freetext",
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
LABELS_CSV = DATA_DIR / "labeled" / "vastu_labels.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def get_image_files():
    """Find all images in raw/ that haven't been labeled yet."""
    if not RAW_DIR.exists():
        print(f"No raw image directory found at {RAW_DIR}")
        print("Place your floor plan images there and re-run.")
        return []

    all_images = sorted(
        f.name for f in RAW_DIR.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not all_images:
        print(f"No images found in {RAW_DIR}")
        return []

    # Skip already-labeled images
    labeled = set()
    if LABELS_CSV.exists():
        with open(LABELS_CSV, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labeled.add(row["image_path"])

    unlabeled = [img for img in all_images if img not in labeled]
    print(f"Found {len(all_images)} images, {len(labeled)} already labeled, {len(unlabeled)} remaining.")
    return unlabeled


def ask_direction(prompt, allow_none=False):
    """Ask user for a compass direction."""
    options = DIRECTIONS.copy()
    if allow_none:
        options.append("NONE")
    hint = "/".join(options)

    while True:
        answer = input(f"  {prompt} [{hint}]: ").strip().upper()
        if answer in options:
            return answer
        print(f"  Invalid. Choose from: {hint}")


def ask_yesno(prompt):
    while True:
        answer = input(f"  {prompt} [yes/no]: ").strip().lower()
        if answer in ("yes", "no", "y", "n"):
            return "yes" if answer in ("yes", "y") else "no"
        print("  Please answer yes or no.")


def compute_vastu_score(labels):
    """Compute a simple Vastu compliance score (0-10)."""
    score = 0
    total_rules = 0

    for rule in VASTU_RULES:
        if rule["type"] == "freetext":
            continue

        value = labels.get(rule["id"])
        if value == "NONE":
            continue  # skip absent features

        total_rules += 1
        if rule["type"] == "yesno":
            if value == rule["ideal"]:
                score += 1
        else:
            if value in rule["ideal"]:
                score += 1

    if total_rules == 0:
        return 5  # default middle score
    return round(10 * score / total_rules, 1)


def label_image(image_name):
    """Show an image and collect labels interactively."""
    image_path = RAW_DIR / image_name
    img = Image.open(image_path)

    # Display the floor plan
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img)
    ax.set_title(f"Floor Plan: {image_name}", fontsize=14)
    ax.axis("off")

    # Add compass overlay
    ax.text(0.5, 0.98, "N", transform=ax.transAxes, ha="center", va="top",
            fontsize=16, fontweight="bold", color="red")
    ax.text(0.5, 0.02, "S", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=16, fontweight="bold", color="red")
    ax.text(0.02, 0.5, "W", transform=ax.transAxes, ha="left", va="center",
            fontsize=16, fontweight="bold", color="red")
    ax.text(0.98, 0.5, "E", transform=ax.transAxes, ha="right", va="center",
            fontsize=16, fontweight="bold", color="red")

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)

    print(f"\n{'='*60}")
    print(f"Labeling: {image_name}")
    print(f"{'='*60}")
    print("Refer to the displayed image. Assume North is UP.\n")

    labels = {"image_path": image_name}

    for rule in VASTU_RULES:
        if "description" in rule:
            print(f"  ({rule['description']})")

        if rule["type"] == "direction":
            labels[rule["id"]] = ask_direction(rule["question"])
        elif rule["type"] == "direction_optional":
            labels[rule["id"]] = ask_direction(rule["question"], allow_none=True)
        elif rule["type"] == "yesno":
            labels[rule["id"]] = ask_yesno(rule["question"])
        elif rule["type"] == "freetext":
            labels[rule["id"]] = input(f"  {rule['question']}: ").strip()
        print()

    # Compute score
    vastu_score = compute_vastu_score(labels)
    labels["vastu_score"] = vastu_score
    labels["compliant"] = "yes" if vastu_score >= 6 else "no"

    print(f"  → Vastu Score: {vastu_score}/10 ({'COMPLIANT' if labels['compliant'] == 'yes' else 'NON-COMPLIANT'})")

    plt.close(fig)
    return labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Vastu Floor Plan Labeling Tool")
    print("=" * 60)

    unlabeled = get_image_files()
    if not unlabeled:
        print("\nNothing to label. Add images to 02-vastu-model/data/raw/ and re-run.")
        return

    # Prepare CSV
    fieldnames = (
        ["image_path"]
        + [r["id"] for r in VASTU_RULES]
        + ["vastu_score", "compliant"]
    )

    write_header = not LABELS_CSV.exists()
    LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nLabeling {len(unlabeled)} images. Press Ctrl+C to stop anytime (progress is saved).\n")

    for i, image_name in enumerate(unlabeled):
        print(f"\n[{i+1}/{len(unlabeled)}]")
        try:
            labels = label_image(image_name)
        except KeyboardInterrupt:
            print("\n\nStopped. Progress saved.")
            break

        # Append to CSV after each image (no data loss on crash)
        with open(LABELS_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
                write_header = False
            writer.writerow(labels)

        print(f"  Saved to {LABELS_CSV}")

    print(f"\nDone! Labels at: {LABELS_CSV}")


if __name__ == "__main__":
    main()
