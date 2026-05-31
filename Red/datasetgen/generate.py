"""
generate.py  -  CLI for stage 1 (valid builds + previews)
=========================================================

Usage:
    python generate.py --n 12 --out ../builds_preview --seed 0 [--debug] [--bagmode]

Produces, per build:
    build_XXXX.png         the rendered scene (RGB)
    build_XXXX_debug.png   optional grid + occupancy overlay  (--debug)
    build_XXXX.json        the build description (without _piece objects)

This is the "diverse valid builds" deliverable; the YOLO-seg export will be
layered on top of these build dicts in stage 2.
"""

from __future__ import annotations

import argparse
import copy
import json
import os

from build_generator import BuildGenerator
from catalog import Catalog
from renderer import render_build, render_debug


def _strip(build: dict) -> dict:
    b = copy.deepcopy({k: v for k, v in build.items() if k != "_piece"})
    for coll in ("bags", "items"):
        for p in b[coll]:
            p.pop("_piece", None)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "builds_preview"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--bagmode", action="store_true",
                    help="use the BagMode background")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cat = Catalog()
    print(cat.summary())

    for i in range(args.n):
        gen = BuildGenerator(cat, seed=args.seed + i)
        build = gen.generate()
        scene = render_build(build, cat, bag_mode=args.bagmode)
        scene.convert("RGB").save(os.path.join(args.out, f"build_{i:04d}.png"))
        with open(os.path.join(args.out, f"build_{i:04d}.json"), "w") as fh:
            json.dump(_strip(build), fh, indent=1)
        if args.debug:
            dbg = render_debug(build, cat, bag_mode=args.bagmode)
            dbg.convert("RGB").save(os.path.join(args.out, f"build_{i:04d}_debug.png"))
        print(f"build {i:04d}: bags={len(build['bags'])} items={len(build['items'])}")

    print(f"\nwrote {args.n} builds to {args.out}")


if __name__ == "__main__":
    main()
