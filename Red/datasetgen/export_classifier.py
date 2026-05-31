"""
export_classifier.py
--------------------
Stage 2 (Solution B): Generate an augmented dataset of isolated bags and items
for training a high-precision PyTorch Image Classifier (e.g. ResNet50).

Output structure:
  classifier_dataset/
    train/
      bag_<bag_name>/
        img_000.png ...
      item_<item_name>/
        img_000.png ...
    val/
      bag_<bag_name>/
        img_000.png ...
      item_<item_name>/
        img_000.png ...
"""

from __future__ import annotations

import argparse
import os
import random
import shutil

from PIL import Image, ImageEnhance, ImageFilter

from catalog import Catalog

# --------------------------------------------------------------------------- #
#  Color / Texture Augmentations (Pure PIL implementation)
# --------------------------------------------------------------------------- #
def apply_augmentations(img: Image.Image, rng: random.Random) -> Image.Image:
    """Apply random brightness, contrast, color, and blur augmentations."""
    # 1. Brightness
    if rng.random() < 0.8:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(rng.uniform(0.85, 1.15))
        
    # 2. Contrast
    if rng.random() < 0.8:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(rng.uniform(0.85, 1.15))
        
    # 3. Saturation (Color)
    if rng.random() < 0.8:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(rng.uniform(0.8, 1.2))
        
    # 4. Blur
    if rng.random() < 0.15:
        radius = rng.uniform(0.2, 1.0)
        img = img.filter(ImageFilter.GaussianBlur(radius))
        
    return img


def create_random_bg(w: int, h: int, mode: str, rng: random.Random) -> Image.Image:
    """Create a random background canvas."""
    if mode == "black":
        return Image.new("RGBA", (w, h), (0, 0, 0, 255))
    elif mode == "color":
        # Random dark neutral colors (grey, brown, dark green, dark blue)
        r = rng.randint(10, 40)
        g = rng.randint(10, 40)
        b = rng.randint(10, 40)
        return Image.new("RGBA", (w, h), (r, g, b, 255))
    else:
        # A simple noise background or a dark gradient
        bg = Image.new("RGBA", (w, h), (0, 0, 0, 255))
        # Add slight noise
        pixels = bg.load()
        for x in range(w):
            for y in range(h):
                noise = rng.randint(-15, 15)
                pixels[x, y] = (max(0, min(255, 25 + noise)),
                                max(0, min(255, 25 + noise)),
                                max(0, min(255, 25 + noise)), 255)
        return bg


def generate_item_variant(piece_img: Image.Image, size: int, rng: random.Random) -> Image.Image:
    """Rotate the piece, paste it on a random background, and resize to standard size."""
    # 1. Random 90 deg rotation
    rot = rng.choice([0, 90, 180, 270])
    if rot != 0:
        # PIL rotates counter-clockwise
        rotated = piece_img.rotate(-rot, expand=True)
    else:
        rotated = piece_img.copy()

    # Apply some augmentations to the rotated object itself
    rotated = apply_augmentations(rotated, rng)

    # 2. Create square canvas with some padding
    max_dim = max(rotated.width, rotated.height)
    padding = int(max_dim * rng.uniform(0.05, 0.20))
    canvas_size = max_dim + 2 * padding
    
    # Random background type: black (50%), dark color (25%), textured noise (25%)
    bg_type = rng.choice(["black", "black", "color", "noise"])
    canvas = create_random_bg(canvas_size, canvas_size, bg_type, rng)

    # 3. Paste rotated piece in center
    px = (canvas_size - rotated.width) // 2
    py = (canvas_size - rotated.height) // 2
    canvas.alpha_composite(rotated, (px, py))

    # 4. Convert to RGB and resize to target dimension (e.g., 224x224)
    rgb_canvas = canvas.convert("RGB")
    final_img = rgb_canvas.resize((size, size), Image.Resampling.LANCZOS)
    return final_img


# --------------------------------------------------------------------------- #
#  Main exporter
# --------------------------------------------------------------------------- #
def export_classifier(out_dir: str, n_train: int = 40, n_val: int = 10,
                      size: int = 224, seed: int = 0):
    cat = Catalog()
    rng = random.Random(seed)

    print(f"Exporting classification dataset to {out_dir}")
    print(f"Total pieces to export: {len(cat.bags)} bags, {len(cat.items)} items")
    print(f"Output image size: {size}x{size}")
    print(f"Train samples per class: {n_train}, Val samples per class: {n_val}")

    # Clear old directories if they exist
    for split in ("train", "val"):
        d = os.path.join(out_dir, split)
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    def clean_class_name(kind: str, name: str) -> str:
        s = name.replace(" ", "_")
        s = s.replace("'", "")
        return f"{kind}_{s}"

    all_pieces = []
    for b in cat.bags:
        all_pieces.append((b, clean_class_name("bag", b.name)))
    for i in cat.items:
        all_pieces.append((i, clean_class_name("item", i.name)))

    # Write class map file
    class_dirnames = sorted(list(set(dirname for _, dirname in all_pieces)))
    with open(os.path.join(out_dir, "classes_classifier.txt"), "w") as fh:
        fh.write("\n".join(class_dirnames) + "\n")

    # Process each piece
    for idx, (piece, class_dirname) in enumerate(all_pieces):
        # Open base RGBA image
        try:
            piece_img = piece.image()
        except Exception as e:
            print(f"Skipping '{piece.name}': failed to load image ({e})")
            continue

        # Create split folders
        train_class_dir = os.path.join(out_dir, "train", class_dirname)
        val_class_dir = os.path.join(out_dir, "val", class_dirname)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        # Generate train variants
        for i in range(n_train):
            variant = generate_item_variant(piece_img, size, rng)
            variant.save(os.path.join(train_class_dir, f"img_{i:04d}.png"))

        # Generate val variants
        for i in range(n_val):
            variant = generate_item_variant(piece_img, size, rng)
            variant.save(os.path.join(val_class_dir, f"img_{i:04d}.png"))

        if (idx + 1) % 50 == 0 or (idx + 1) == len(all_pieces):
            print(f"  {idx + 1}/{len(all_pieces)} classes processed")

    print(f"\nSuccessfully generated classification dataset at {os.path.abspath(out_dir)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "classifier_dataset"))
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-val", type=int, default=10)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    
    export_classifier(args.out, n_train=args.n_train, n_val=args.n_val,
                      size=args.size, seed=args.seed)


if __name__ == "__main__":
    main()
