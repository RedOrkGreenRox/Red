"""
export_yolo_2classes.py
-----------------------
Stage 2 (Modified): Turn generated builds into a 2-class YOLOv8 **segmentation** dataset
with cropped grid (1080x720) to focus strictly on the active inventory.

Classes:
  0: bag
  1: item

Cropping:
  The active play area starts at (342, 34) and has size 1080x720.
  All images are cropped to this box, and all label coordinates are adjusted.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil

import yaml
from PIL import Image

from build_generator import BuildGenerator
from catalog import Catalog
from geometry import clip_poly, polygons_for_placement
from renderer import render_build, render_debug

# Cropping Box Constants
CROP_X = 342
CROP_Y = 34
CROP_W = 1080  # 9 cells * 120 px
CROP_H = 720   # 6 cells * 120 px


# --------------------------------------------------------------------------- #
#  Class definitions (Fixed to 2 classes)
# --------------------------------------------------------------------------- #
NAMES = ["bag", "item"]
NAME_TO_ID = {"bag": 0, "item": 1}


# --------------------------------------------------------------------------- #
#  One build -> (image, label lines)
# --------------------------------------------------------------------------- #
def build_labels_2classes(build: dict, cat: Catalog, img_w: int, img_h: int):
    """Return a list of 2-class YOLO-seg label strings for one build,
    adjusted to the cropped grid coordinates."""
    g = build["grid"]
    oxy = (g["origin_x"], g["origin_y"])
    cell = g["cell"]
    lines = []

    for coll, kind in ((build["bags"], "bag"), (build["items"], "item")):
        cid = NAME_TO_ID[kind]
        for p in coll:
            polys = polygons_for_placement(p, oxy, cell)
            if not polys:
                continue
            # YOLO-seg: one polygon per instance. Keep the largest.
            poly = max(polys, key=lambda pl: _poly_area(pl))
            
            # 1. Translate coordinates from full frame to cropped frame
            translated_poly = []
            for x, y in poly:
                tx = x - CROP_X
                ty = y - CROP_Y
                translated_poly.append((tx, ty))
                
            # 2. Clip to the cropped boundary [0, CROP_W] x [0, CROP_H]
            clipped_poly = clip_poly(translated_poly, img_w, img_h)
            
            # 3. Normalize coordinates relative to the cropped width and height
            coords = []
            for x, y in clipped_poly:
                coords.append(f"{x / img_w:.6f}")
                coords.append(f"{y / img_h:.6f}")
            lines.append(f"{cid} " + " ".join(coords))
    return lines


def _poly_area(poly):
    """Shoelace area (absolute)."""
    n = len(poly)
    a = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


# --------------------------------------------------------------------------- #
#  Dataset writer
# --------------------------------------------------------------------------- #
def export_2classes(n: int, out_dir: str, seed: int = 0, val_frac: float = 0.1,
                    debug: bool = False, bagmode_frac: float = 0.5):
    cat = Catalog()

    # fresh output tree
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        d = os.path.join(out_dir, sub)
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    if debug:
        os.makedirs(os.path.join(out_dir, "debug"), exist_ok=True)

    rng = random.Random(seed)
    n_val = max(1, int(round(n * val_frac))) if n > 1 else 0
    val_idx = set(rng.sample(range(n), n_val)) if n_val else set()

    stats = {"train": 0, "val": 0, "instances": 0, "empty": 0}
    for i in range(n):
        gen = BuildGenerator(cat, seed=seed * 100000 + i)
        build = gen.generate()
        bag_mode = rng.random() < bagmode_frac

        # Render original scene
        scene = render_build(build, cat, bag_mode=bag_mode).convert("RGB")
        
        # Crop the scene to the active grid area
        cropped_scene = scene.crop((CROP_X, CROP_Y, CROP_X + CROP_W, CROP_Y + CROP_H))
        w, h = cropped_scene.size  # Should be 1080x720
        
        # Generate labels using cropped dimensions
        labels = build_labels_2classes(build, cat, w, h)
        if not labels:
            stats["empty"] += 1

        split = "val" if i in val_idx else "train"
        stem = f"img_{i:05d}"
        
        # Save cropped image and labels
        cropped_scene.save(os.path.join(out_dir, "images", split, stem + ".png"))
        with open(os.path.join(out_dir, "labels", split, stem + ".txt"), "w") as fh:
            fh.write("\n".join(labels) + ("\n" if labels else ""))

        stats[split] += 1
        stats["instances"] += len(labels)
        
        if debug:
            dbg = render_debug(build, cat, bag_mode=bag_mode).convert("RGB")
            cropped_dbg = dbg.crop((CROP_X, CROP_Y, CROP_X + CROP_W, CROP_Y + CROP_H))
            cropped_dbg.save(os.path.join(out_dir, "debug", stem + "_debug.png"))
            
        if (i + 1) % 25 == 0 or i + 1 == n:
            print(f"  {i + 1}/{n} scenes done")

    # data.yaml + classes.txt
    data = {
        "path": os.path.abspath(out_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": len(NAMES),
        "names": NAMES,
    }
    with open(os.path.join(out_dir, "data.yaml"), "w") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)
    with open(os.path.join(out_dir, "classes.txt"), "w") as fh:
        fh.write("\n".join(NAMES) + "\n")

    print("\n=== 2-class export summary ===")
    print(f"classes      : {len(NAMES)} {NAMES}")
    print(f"train images : {stats['train']}")
    print(f"val images   : {stats['val']}")
    print(f"instances    : {stats['instances']}")
    print(f"empty scenes : {stats['empty']}")
    print(f"output       : {os.path.abspath(out_dir)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset_cropped_2classes"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--bagmode-frac", type=float, default=0.5)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    export_2classes(args.n, args.out, seed=args.seed, val_frac=args.val_frac,
                    debug=args.debug, bagmode_frac=args.bagmode_frac)


if __name__ == "__main__":
    main()
