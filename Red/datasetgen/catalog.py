"""
catalog.py
----------
Loads the item/bag database (items_4_1_0.json) and reconciles it with the
PNG assets in Bags/ and Items/.

Core concepts
=============
The game board is a 6 (rows) x 9 (cols) grid. One cell = 120 px.
Inside the background image the play-field starts at pixel origin
(GRID_ORIGIN_X, GRID_ORIGIN_Y) (auto-detected, overridable).

Two kinds of objects can be placed:

* Bag  - a container.  It occupies a set of grid cells (its ``shape``) and is
         drawn from a PNG that is ``cols*120 + 2*FRAME`` x ``rows*120 + 2*FRAME``
         (FRAME = 7.5 px studded border that overhangs the cells on every side).
         Bags may be non-rectangular (e.g. an L shape); missing cells are
         transparent in the PNG.

* Item - placed *inside* bag cells only.  Its PNG is exactly
         ``bbox_w*120`` x ``bbox_h*120`` and is snapped to the grid.

Every object carries:
  name, kind ('bag'|'item'), cells (list of (cx,cy) offsets, normalised so the
  bbox top-left is (0,0)), image path, and the pixel size of its image.

A few PNGs are stored rotated 90 deg relative to their JSON ``itemShape``.  We
detect this by comparing the image aspect ratio to the shape bbox and, if they
are transposed, we rotate the *shape* to match the image (the image pixels are
ground truth for what gets drawn / masked).
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from PIL import Image

# --------------------------------------------------------------------------- #
#  Geometry constants
# --------------------------------------------------------------------------- #
CELL = 120                 # px per grid cell
GRID_COLS = 9              # board width  (cells)
GRID_ROWS = 6              # board height (cells)
BAG_FRAME = 7.5            # bag border overhang (px) on each side
GRID_ORIGIN_X = 342        # play-field top-left inside the background image
GRID_ORIGIN_Y = 34

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
#  Data structures
# --------------------------------------------------------------------------- #
Cell = Tuple[int, int]     # (col, row) offset within the object's own bbox


@dataclass(eq=False)
class Piece:
    name: str
    kind: str                      # 'bag' | 'item'
    cells: List[Cell]              # normalised offsets, bbox starts at (0,0)
    img_path: str
    img_size: Tuple[int, int]      # (w, h) px of the PNG
    item_types: List[str] = field(default_factory=list)
    rarity: Optional[str] = None

    def __hash__(self):            # identity hash so lru_cache on image() works
        return id(self)

    # ----- derived -----
    @property
    def cols(self) -> int:
        return max(c for c, _ in self.cells) + 1

    @property
    def rows(self) -> int:
        return max(r for _, r in self.cells) + 1

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    @property
    def cellset(self) -> frozenset:
        return frozenset(self.cells)

    def rotations(self):
        """Yield (rot_deg, cells) for 0/90/180/270 deg.  Bags are *not*
        rotated in-game so only items use this."""
        cur = list(self.cells)
        for deg in (0, 90, 180, 270):
            yield deg, _normalise(cur)
            cur = [(-r, c) for c, r in cur]   # rotate 90 deg CW

    @lru_cache(maxsize=None)
    def image(self) -> Image.Image:
        return Image.open(self.img_path).convert("RGBA")


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _normalise(cells: List[Cell]) -> List[Cell]:
    minx = min(c for c, _ in cells)
    miny = min(r for _, r in cells)
    return sorted((c - minx, r - miny) for c, r in cells)


def _shape_from_json(shape_json) -> List[Cell]:
    return _normalise([(p["x"], -p["y"]) for p in shape_json])


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


# --------------------------------------------------------------------------- #
#  Catalog loader
# --------------------------------------------------------------------------- #
class Catalog:
    def __init__(self, root: str = REPO_ROOT):
        self.root = root
        self.json_path = os.path.join(root, "items_4_1_0.json")
        with open(self.json_path, encoding="utf-8") as fh:
            self.raw = json.load(fh)

        self.bag_files = {_stem(p): p for p in glob.glob(os.path.join(root, "Bags", "*.png"))}
        self.item_files = {_stem(p): p for p in glob.glob(os.path.join(root, "Items", "*.png"))}

        self.bags: List[Piece] = []
        self.items: List[Piece] = []
        self._build()

    # ------------------------------------------------------------------ #
    def _match_bag_file(self, name: str) -> Optional[str]:
        # bags sometimes have an " open" suffix in the filename
        for key in (name, f"{name} open"):
            if key in self.bag_files:
                return self.bag_files[key]
        # last resort: prefix match
        for key, path in self.bag_files.items():
            if key.startswith(name):
                return path
        return None

    def _reconcile(self, cells: List[Cell], img_w: int, img_h: int,
                   is_bag: bool) -> List[Cell]:
        """Return shape cells matching the image orientation."""
        cells = _normalise(cells)
        cols = max(c for c, _ in cells) + 1
        rows = max(r for _, r in cells) + 1
        if is_bag:
            exp_w, exp_h = cols * CELL + 15, rows * CELL + 15
        else:
            exp_w, exp_h = cols * CELL, rows * CELL

        def close(a, b):
            return abs(a - b) <= 16

        if close(img_w, exp_w) and close(img_h, exp_h):
            return cells
        # try transposed shape (image rotated 90 deg vs json)
        tcells = _normalise([(r, c) for c, r in cells])
        tcols = max(c for c, _ in tcells) + 1
        trows = max(r for _, r in tcells) + 1
        if is_bag:
            texp_w, texp_h = tcols * CELL + 15, trows * CELL + 15
        else:
            texp_w, texp_h = tcols * CELL, trows * CELL
        if close(img_w, texp_w) and close(img_h, texp_h):
            return tcells
        # give up - keep original (renderer will fit by scaling)
        return cells

    def _build(self):
        for it in self.raw["items"]:
            name = it["name"]
            types = it.get("itemTypes", [])
            shape = it.get("itemShape")
            if not shape:
                continue
            cells = _shape_from_json(shape)
            is_bag = "Bag" in types

            if is_bag:
                path = self._match_bag_file(name)
            else:
                path = self.item_files.get(name)
            if not path or not os.path.exists(path):
                continue

            with Image.open(path) as im:
                w, h = im.size
            cells = self._reconcile(cells, w, h, is_bag)

            piece = Piece(
                name=name,
                kind="bag" if is_bag else "item",
                cells=cells,
                img_path=path,
                img_size=(w, h),
                item_types=types,
                rarity=it.get("rarity"),
            )
            (self.bags if is_bag else self.items).append(piece)

        # dedupe by (name, cellset) keeping first
        self.bags = _dedupe(self.bags)
        self.items = _dedupe(self.items)

    # ------------------------------------------------------------------ #
    def summary(self) -> str:
        from collections import Counter
        bc = Counter((b.cols, b.rows, b.n_cells) for b in self.bags)
        ic = Counter(i.n_cells for i in self.items)
        lines = [
            f"bags : {len(self.bags)}",
            f"items: {len(self.items)}",
            f"item cell-count dist: {dict(sorted(ic.items()))}",
        ]
        return "\n".join(lines)


def _dedupe(pieces: List[Piece]) -> List[Piece]:
    seen = set()
    out = []
    for p in pieces:
        key = (p.name, p.cellset, p.img_path)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


if __name__ == "__main__":
    cat = Catalog()
    print(cat.summary())
    print("\nexample bags:")
    for b in cat.bags[:5]:
        print(f"  {b.name:24s} {b.cols}x{b.rows} cells={b.n_cells} img={b.img_size}")
    print("example items:")
    for i in cat.items[:5]:
        print(f"  {i.name:24s} {i.cols}x{i.rows} cells={i.n_cells} img={i.img_size}")
