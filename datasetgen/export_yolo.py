"""
export_yolo.py
--------------
Stage 2: turn generated builds into a YOLOv8 **segmentation** dataset.

For every build it:
  1. renders the RGB scene  -> images/{split}/img_XXXXX.png
  2. traces each piece's alpha silhouette -> normalised polygons
     -> labels/{split}/img_XXXXX.txt   (one line per instance:
        `class_id x1 y1 x2 y2 ...`  all in [0,1])
  3. writes data.yaml + classes.txt

Classes
=======
Every distinct piece is its own class.  Bags are kept separate from items by
prefixing the class name:  ``bag:<name>`` / ``item:<name>``  (so bags are
explicitly labelled objects, as required).  The class list is built from the
whole catalog so ids are stable regardless of which builds are sampled.

Usage
=====
    python export_yolo.py --n 50 --val-frac 0.1 --seed 0 \
        --out ../dataset --debug
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
from geometry import (alpha_polygons, bbox_of_polys, clip_poly, piece_paste,
                      polygons_for_placement)
from renderer import render_build, render_debug


# --------------------------------------------------------------------------- #
#  Class registry
# --------------------------------------------------------------------------- #
def class_name(kind: str, name: str) -> str:
    return f"{kind}:{name}"


def build_class_map(cat: Catalog):
    """Stable id <-> name maps over the whole catalog (bags first)."""
    names = [class_name("bag", b.name) for b in sorted(cat.bags, key=lambda p: p.name)]
    names += [class_name("item", i.name) for i in sorted(cat.items, key=lambda p: p.name)]
    name_to_id = {n: i for i, n in enumerate(names)}
    return names, name_to_id


# --------------------------------------------------------------------------- #
#  One build -> (image, label lines)
# --------------------------------------------------------------------------- #
def build_labels(build: dict, cat: Catalog, name_to_id: dict,
                 img_w: int, img_h: int):
    """Return a list of YOLO-seg label strings for one build."""
    g = build["grid"]
    oxy = (g["origin_x"], g["origin_y"])
    cell = g["cell"]
    lines = []

    for coll, kind in ((build["bags"], "bag"), (build["items"], "item")):
        for p in coll:
            cid = name_to_id[class_name(kind, p["name"])]
            polys = polygons_for_placement(p, oxy, cell)
            if not polys:
                continue
            # YOLO-seg: one polygon per instance. If a piece traced into
            # several disjoint blobs, keep the largest (rare; bags are 1 blob).
            poly = max(polys, key=lambda pl: _poly_area(pl))
            poly = clip_poly(poly, img_w, img_h)
            coords = []
            for x, y in poly:
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
def export(n: int, out_dir: str, seed: int = 0, val_frac: float = 0.1,
           debug: bool = False, bagmode_frac: float = 0.5):
    cat = Catalog()
    names, name_to_id = build_class_map(cat)

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

        scene = render_build(build, cat, bag_mode=bag_mode).convert("RGB")
        w, h = scene.size
        labels = build_labels(build, cat, name_to_id, w, h)
        if not labels:
            stats["empty"] += 1

        split = "val" if i in val_idx else "train"
        stem = f"img_{i:05d}"
        scene.save(os.path.join(out_dir, "images", split, stem + ".png"))
        with open(os.path.join(out_dir, "labels", split, stem + ".txt"), "w") as fh:
            fh.write("\n".join(labels) + ("\n" if labels else ""))

        stats[split] += 1
        stats["instances"] += len(labels)
        if debug:
            dbg = render_debug(build, cat, bag_mode=bag_mode).convert("RGB")
            dbg.save(os.path.join(out_dir, "debug", stem + "_debug.png"))
        if (i + 1) % 25 == 0 or i + 1 == n:
            print(f"  {i + 1}/{n} scenes done")

    # data.yaml + classes.txt
    data = {
        "path": os.path.abspath(out_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": names,
    }
    with open(os.path.join(out_dir, "data.yaml"), "w") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)
    with open(os.path.join(out_dir, "classes.txt"), "w") as fh:
        fh.write("\n".join(names) + "\n")

    print("\n=== export summary ===")
    print(f"classes      : {len(names)}")
    print(f"train images : {stats['train']}")
    print(f"val images   : {stats['val']}")
    print(f"instances    : {stats['instances']}")
    print(f"empty scenes : {stats['empty']}")
    print(f"output       : {os.path.abspath(out_dir)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--bagmode-frac", type=float, default=0.5,
                    help="fraction of scenes using the BagMode background")
    ap.add_argument("--debug", action="store_true",
                    help="also write debug/ overlays for visual checking")
    args = ap.parse_args()
    export(args.n, args.out, seed=args.seed, val_frac=args.val_frac,
           debug=args.debug, bagmode_frac=args.bagmode_frac)


if __name__ == "__main__":
    main()
