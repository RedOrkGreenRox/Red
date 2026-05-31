"""
renderer.py
-----------
Composites a build (from build_generator) onto an inventory background and,
optionally, produces a debug overlay (grid + **true alpha silhouettes** traced
from each piece's PNG) so a human can confirm the build and the future YOLO-seg
labels are correct.

Pixel placement lives in geometry.py and is shared with the YOLO exporter, so
what you see outlined here is exactly what gets written to the label files.
"""

from __future__ import annotations

import os

from PIL import Image, ImageDraw

from catalog import Catalog
from geometry import piece_paste, polygons_for_placement


def _bg_path(catalog: Catalog, bag_mode: bool) -> str:
    name = "InventoryBagMode.png" if bag_mode else "Inventory.png"
    return os.path.join(catalog.root, "Backgrounds", name)


def render_build(build: dict, catalog: Catalog,
                 bag_mode: bool = False) -> Image.Image:
    """Return the composed RGBA scene."""
    g = build["grid"]
    oxy = (g["origin_x"], g["origin_y"])
    cell = g["cell"]

    bg = Image.open(_bg_path(catalog, bag_mode)).convert("RGBA")
    canvas = bg.copy()

    # bags first, then items on top
    for b in build["bags"]:
        img, x, y = piece_paste(b, oxy, cell)
        canvas.alpha_composite(img, (x, y))
    for it in build["items"]:
        img, x, y = piece_paste(it, oxy, cell)
        canvas.alpha_composite(img, (x, y))

    return canvas


def render_debug(build: dict, catalog: Catalog,
                 bag_mode: bool = False) -> Image.Image:
    """Scene + grid lines + per-piece alpha silhouette outlines."""
    g = build["grid"]
    ox, oy = g["origin_x"], g["origin_y"]
    cell = g["cell"]
    cols, rows = g["cols"], g["rows"]
    oxy = (ox, oy)

    scene = render_build(build, catalog, bag_mode).copy()
    draw = ImageDraw.Draw(scene)

    # grid
    for c in range(cols + 1):
        x = ox + c * cell
        draw.line([(x, oy), (x, oy + rows * cell)], fill=(255, 255, 255, 90), width=1)
    for r in range(rows + 1):
        y = oy + r * cell
        draw.line([(ox, y), (ox + cols * cell, y)], fill=(255, 255, 255, 90), width=1)

    # true silhouettes from the alpha channel
    for coll, color in ((build["bags"], (0, 200, 255, 255)),
                        (build["items"], (255, 70, 70, 255))):
        for p in coll:
            for poly in polygons_for_placement(p, oxy, cell):
                if len(poly) >= 2:
                    draw.line([(int(x), int(y)) for x, y in poly] +
                              [(int(poly[0][0]), int(poly[0][1]))],
                              fill=color, width=2)

    return scene


if __name__ == "__main__":
    from build_generator import BuildGenerator

    cat = Catalog()
    gen = BuildGenerator(cat, seed=7)
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "builds_preview")
    os.makedirs(out_dir, exist_ok=True)
    for i in range(4):
        b = gen.generate()
        scene = render_build(b, cat, bag_mode=(i % 2 == 1))
        dbg = render_debug(b, cat, bag_mode=(i % 2 == 1))
        scene.convert("RGB").save(os.path.join(out_dir, f"build_{i}.png"))
        dbg.convert("RGB").save(os.path.join(out_dir, f"build_{i}_debug.png"))
        print("saved build", i, "bags", len(b["bags"]), "items", len(b["items"]))
