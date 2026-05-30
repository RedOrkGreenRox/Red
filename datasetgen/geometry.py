"""
geometry.py
-----------
Single source of truth for *where* a piece's PNG lands on the canvas and for
turning that piece's alpha channel into a YOLO-seg polygon.

Both renderer.py (compositing + debug overlay) and export_yolo.py (label
generation) import from here, so the drawn pixels and the annotated silhouette
are guaranteed to agree.

Coordinate conventions
======================
Board cell (col,row) top-left pixel inside the background:
    px = origin_x + col*CELL
    py = origin_y + row*CELL

Bags : PNG is (cols*CELL + 2*BAG_FRAME) x (rows*CELL + 2*BAG_FRAME); the studded
       frame overhangs the cell block by BAG_FRAME on every side, so the PNG is
       pasted at (px - BAG_FRAME, py - BAG_FRAME).  Bags are never rotated.

Items: PNG is bbox*CELL and is centred on its cell block; rotation 'rot' (CW
       degrees) is applied to the PNG first (expand=True), then it is centred.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from catalog import BAG_FRAME, CELL


def rotated_image(img: Image.Image, rot: int) -> Image.Image:
    """Rotate an RGBA image by `rot` CW degrees (PIL rotates CCW)."""
    if rot % 360 == 0:
        return img
    return img.rotate(-rot, expand=True)


def piece_paste(placement: dict, oxy: Tuple[int, int],
                cell: int = CELL) -> Tuple[Image.Image, int, int]:
    """Return (image_to_paste, paste_x, paste_y) for one placement dict.

    `placement` must carry a live `_piece` (preferred) or an `img_path`.
    """
    ox, oy = oxy
    piece = placement.get("_piece")
    img = piece.image() if piece is not None else \
        Image.open(placement["img_path"]).convert("RGBA")

    cells = placement["cells"]
    ax, ay = placement["anchor"]
    px = ox + ax * cell
    py = oy + ay * cell

    if placement["kind"] == "bag":
        f = int(round(BAG_FRAME))
        return img, px - f, py - f

    # item: rotate then centre on the cell block
    img = rotated_image(img, placement.get("rot", 0))
    cols = max(c[0] for c in cells) - min(c[0] for c in cells) + 1
    rows = max(c[1] for c in cells) - min(c[1] for c in cells) + 1
    block_w, block_h = cols * cell, rows * cell
    offx = (block_w - img.width) // 2
    offy = (block_h - img.height) // 2
    return img, px + offx, py + offy


# --------------------------------------------------------------------------- #
#  Silhouette tracing (alpha -> polygons)
# --------------------------------------------------------------------------- #
def alpha_polygons(img: Image.Image, paste_xy: Tuple[int, int],
                   alpha_thresh: int = 8,
                   approx_eps_frac: float = 0.0015,
                   min_area_frac: float = 0.002) -> List[List[Tuple[float, float]]]:
    """Trace the silhouette of an RGBA image from its alpha channel.

    Returns a list of polygons (each a list of absolute (x, y) pixel points,
    offset by `paste_xy`).  Multiple polygons are returned when the piece has
    disjoint opaque regions (e.g. a non-rectangular bag).

    Parameters
    ----------
    alpha_thresh   : pixels with alpha > this are considered opaque.
    approx_eps_frac: Douglas-Peucker epsilon as a fraction of the contour
                     perimeter (smaller = more faithful, more points).
    min_area_frac  : drop contours whose area is below this fraction of the
                     image area (kills stray antialias specks).
    """
    px, py = paste_xy
    a = np.array(img.split()[-1])                      # alpha channel (H, W)
    mask = (a > alpha_thresh).astype(np.uint8) * 255
    if mask.sum() == 0:
        return []

    # close 1-px gaps so antialiased edges trace as one contour
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    img_area = img.width * img.height
    polys: List[List[Tuple[float, float]]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_frac * img_area:
            continue
        eps = approx_eps_frac * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, max(eps, 1.0), True)
        pts = [(float(p[0][0] + px), float(p[0][1] + py)) for p in approx]
        if len(pts) >= 3:
            polys.append(pts)
    return polys


def polygons_for_placement(placement: dict, oxy: Tuple[int, int],
                           cell: int = CELL, **kw
                           ) -> List[List[Tuple[float, float]]]:
    """Convenience: paste-position a placement and trace its silhouette."""
    img, px, py = piece_paste(placement, oxy, cell)
    return alpha_polygons(img, (px, py), **kw)


def bbox_of_polys(polys: List[List[Tuple[float, float]]]
                  ) -> Optional[Tuple[float, float, float, float]]:
    """Axis-aligned bbox (xmin, ymin, xmax, ymax) over all polygon points."""
    if not polys:
        return None
    xs = [x for poly in polys for x, _ in poly]
    ys = [y for poly in polys for _, y in poly]
    return min(xs), min(ys), max(xs), max(ys)


def clip_poly(poly: List[Tuple[float, float]], w: int, h: int
              ) -> List[Tuple[float, float]]:
    """Clamp polygon points into the [0,w]x[0,h] image (cheap edge clip)."""
    return [(min(max(x, 0.0), w), min(max(y, 0.0), h)) for x, y in poly]
