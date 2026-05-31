"""
build_generator.py
------------------
Generates *valid* inventory builds.

A build is a placement of:
  1. Bags onto the 6x9 board   (bags never overlap; must stay on board).
  2. Items inside bag cells     (items only sit on cells covered by some bag;
                                 items never overlap each other or empty space).

Output is a plain Python dict (JSON-serialisable):

    {
      "grid":   {"cols": 9, "rows": 6, "cell": 120,
                 "origin_x": 342, "origin_y": 34},
      "bags":   [ {placement}, ... ],
      "items":  [ {placement}, ... ],
    }

Each placement:
    {
      "name": str,
      "kind": "bag" | "item",
      "cells": [[col,row], ...],      # absolute board cells it occupies
      "anchor": [col, row],          # board cell of the piece's bbox top-left
      "rot": 0|90|180|270,           # item rotation applied to the PNG
      "img_path": str,
    }

The generator is deterministic given a seed.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from catalog import (CELL, GRID_COLS, GRID_ROWS, GRID_ORIGIN_X, GRID_ORIGIN_Y,
                     Catalog, Piece)

Cell = Tuple[int, int]


# --------------------------------------------------------------------------- #
def _abs_cells(cells: List[Cell], ax: int, ay: int) -> List[Cell]:
    return [(ax + c, ay + r) for c, r in cells]


def _in_board(cells: List[Cell]) -> bool:
    return all(0 <= c < GRID_COLS and 0 <= r < GRID_ROWS for c, r in cells)


class BuildGenerator:
    def __init__(self, catalog: Catalog, seed: Optional[int] = None):
        self.cat = catalog
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------ #
    #  Bag placement
    # ------------------------------------------------------------------ #
    def _place_bags(self, target_fill: float) -> Tuple[List[dict], Dict[Cell, int]]:
        """Greedily drop bags until ``target_fill`` of the board is covered or
        no more fit.  Returns (placements, occupancy map cell->bag_index)."""
        occ: Dict[Cell, int] = {}
        placements: List[dict] = []
        board_cells = GRID_COLS * GRID_ROWS

        bags = list(self.cat.bags)
        self.rng.shuffle(bags)
        # bias toward placing larger bags first for nicer packing
        bags.sort(key=lambda b: -b.n_cells)
        # but add some randomness so builds differ
        for i in range(len(bags) - 1):
            if self.rng.random() < 0.35:
                j = self.rng.randrange(i, len(bags))
                bags[i], bags[j] = bags[j], bags[i]

        attempts_per_bag = 60
        max_bags = self.rng.randint(3, 8)
        n_placed = 0
        for bag in bags:
            if n_placed >= max_bags:
                break
            if len(occ) / board_cells >= target_fill:
                break
            placed = False
            # candidate anchors in random order
            anchors = [(ax, ay)
                       for ay in range(GRID_ROWS)
                       for ax in range(GRID_COLS)]
            self.rng.shuffle(anchors)
            for ax, ay in anchors[:attempts_per_bag]:
                acells = _abs_cells(bag.cells, ax, ay)
                if not _in_board(acells):
                    continue
                if any(c in occ for c in acells):
                    continue
                idx = len(placements)
                for c in acells:
                    occ[c] = idx
                placements.append({
                    "name": bag.name,
                    "kind": "bag",
                    "cells": [list(c) for c in acells],
                    "anchor": [ax, ay],
                    "rot": 0,
                    "img_path": bag.img_path,
                    "_piece": bag,
                })
                placed = True
                n_placed += 1
                break
            # if a bag could not be placed we just skip it
        return placements, occ

    # ------------------------------------------------------------------ #
    #  Item placement
    # ------------------------------------------------------------------ #
    def _place_items(self, bag_occ: Dict[Cell, int],
                     fill: float) -> List[dict]:
        """Fill bag cells with items.  Items may only sit on cells that belong
        to a bag and that are still free."""
        free = set(bag_occ.keys())          # all bag cells are valid targets
        item_occ: set = set()
        placements: List[dict] = []
        total = len(free)
        if total == 0:
            return placements

        items = list(self.cat.items)
        self.rng.shuffle(items)

        # Build a working list, occasionally favouring small items so we can
        # densely fill the remaining holes.
        target_filled = int(total * fill)
        guard = 0
        while len(item_occ) < target_filled and guard < total * 12:
            guard += 1
            remaining = target_filled - len(item_occ)
            # pick an item that could plausibly fit the remaining budget
            piece = self.rng.choice(items)
            if piece.n_cells > remaining + 2:
                # prefer smaller items late in the fill
                small = [p for p in items if p.n_cells <= max(1, remaining)]
                if small:
                    piece = self.rng.choice(small)
            if self._try_place_item(piece, free, item_occ, bag_occ, placements):
                continue
        return placements

    def _try_place_item(self, piece: Piece, free: set, item_occ: set,
                        bag_occ: Dict[Cell, int], placements: List[dict]) -> bool:
        rots = list(piece.rotations())
        self.rng.shuffle(rots)
        candidate_cells = list(free - item_occ)
        if not candidate_cells:
            return False
        self.rng.shuffle(candidate_cells)

        for rot, rcells in rots:
            # try to anchor the piece so that one of its cells lands on a free cell
            for (tc, tr) in candidate_cells[:40]:
                # align piece cell (0,0)?  better: try each piece cell as anchor onto (tc,tr)
                for pc, pr in rcells[:6]:
                    ax, ay = tc - pc, tr - pr
                    acells = _abs_cells(rcells, ax, ay)
                    if not all(c in bag_occ for c in acells):
                        continue
                    if any(c in item_occ for c in acells):
                        continue
                    for c in acells:
                        item_occ.add(c)
                    placements.append({
                        "name": piece.name,
                        "kind": "item",
                        "cells": [list(c) for c in acells],
                        "anchor": [ax, ay],
                        "rot": rot,
                        "img_path": piece.img_path,
                        "_piece": piece,
                    })
                    return True
        return False

    # ------------------------------------------------------------------ #
    def generate(self,
                 bag_fill: Tuple[float, float] = (0.45, 0.95),
                 item_fill: Tuple[float, float] = (0.55, 1.0)) -> dict:
        bf = self.rng.uniform(*bag_fill)
        itf = self.rng.uniform(*item_fill)
        bags, occ = self._place_bags(bf)
        items = self._place_items(occ, itf)

        build = {
            "grid": {
                "cols": GRID_COLS, "rows": GRID_ROWS, "cell": CELL,
                "origin_x": GRID_ORIGIN_X, "origin_y": GRID_ORIGIN_Y,
            },
            "bags": bags,
            "items": items,
        }
        validate_build(build)
        return build


# --------------------------------------------------------------------------- #
#  Validation
# --------------------------------------------------------------------------- #
def validate_build(build: dict) -> None:
    """Raise AssertionError if the build breaks any rule."""
    bag_cells: set = set()
    for b in build["bags"]:
        cs = [tuple(c) for c in b["cells"]]
        for c in cs:
            assert 0 <= c[0] < GRID_COLS and 0 <= c[1] < GRID_ROWS, \
                f"bag {b['name']} off board: {c}"
            assert c not in bag_cells, f"bag overlap at {c} ({b['name']})"
            bag_cells.add(c)

    item_cells: set = set()
    for it in build["items"]:
        cs = [tuple(c) for c in it["cells"]]
        for c in cs:
            assert c in bag_cells, \
                f"item {it['name']} on non-bag cell {c}"
            assert c not in item_cells, \
                f"item overlap at {c} ({it['name']})"
            item_cells.add(c)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    cat = Catalog()
    gen = BuildGenerator(cat, seed=0)
    for k in range(3):
        b = gen.generate()
        nb, ni = len(b["bags"]), len(b["items"])
        bc = sum(len(x["cells"]) for x in b["bags"])
        ic = sum(len(x["cells"]) for x in b["items"])
        print(f"build {k}: bags={nb} ({bc} cells)  items={ni} ({ic} cells)  "
              f"item-fill={ic}/{bc}")
