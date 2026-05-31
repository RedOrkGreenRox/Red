# Red — YOLO dataset generator

Synthetic inventory scenes for training a YOLO **segmentation** model
(silhouette masks via the alpha channel).

## Data model (auto-discovered)

| thing            | value                                                        |
|------------------|--------------------------------------------------------------|
| Board            | **6 rows × 9 cols**, cell = **120 px**                       |
| Grid origin      | **(342, 34)** inside the 1763×790 background (auto-detected) |
| Backgrounds      | `Backgrounds/Inventory.png`, `InventoryBagMode.png`         |
| Bags             | 28 usable (`Bags/`), PNG = `cols·120+15 × rows·120+15` (7.5 px studded frame overhang) |
| Items            | 837 placeable (`Items/`), PNG = `bbox·120` snapped to grid   |
| Shapes           | from `items_4_1_0.json` → `itemShape` (multi-cell, may be non-rectangular / L-shaped) |

Some item PNGs are stored rotated 90° vs their JSON shape; the catalog
auto-reconciles shape ↔ image orientation.

## Rules enforced (validated on every build)

1. **Bags are placed first** on the board, never overlapping, fully on-board.
2. **Items go only on cells covered by a bag**, never overlapping each other.
3. Bags are also objects to annotate (each is a labelled silhouette).
4. No literal overlaps; layout stays diverse (random bag/item mix, rotations).

## Files

- `catalog.py` — loads JSON + PNGs, reconciles shapes, exposes `Piece` objects.
- `build_generator.py` — `BuildGenerator.generate()` → valid build dict + `validate_build()`.
- `renderer.py` — composites a build onto a background; `render_debug()` adds
  grid + per-cell occupancy overlay for human checking.
- `generate.py` — CLI: render N builds + dump build JSON (+ optional debug).

## Stage 1 — make diverse valid builds (current)

```bash
cd datasetgen
python generate.py --n 12 --seed 0 --debug          # previews + overlays
python generate.py --n 12 --seed 0 --bagmode        # other background
```

Outputs go to `../builds_preview/` (`build_XXXX.png`, `_debug.png`, `.json`).

## Stage 2 — YOLO-seg export (done)

`export_yolo.py` reuses the build dicts to write a ready-to-train YOLOv8
**segmentation** dataset:

```bash
cd datasetgen
python export_yolo.py --n 50 --val-frac 0.1 --seed 0 --debug   # test batch
python export_yolo.py --n 1000 --val-frac 0.1 --seed 0         # full run
```

Output tree (`../dataset/`):

```
dataset/
  data.yaml              # path/train/val/nc/names  (Ultralytics format)
  classes.txt           # one class per line
  images/train|val/img_XXXXX.png
  labels/train|val/img_XXXXX.txt   # one line per instance:
                                   #   class_id x1 y1 x2 y2 …  (all normalised)
  debug/img_XXXXX_debug.png        # only with --debug
```

- **Silhouette labels** are traced from each piece's **alpha channel**
  (`geometry.alpha_polygons` → `cv2.findContours` → `approxPolyDP`), not boxes.
- **Every piece is its own class.** Bags are kept distinct via a name prefix:
  `bag:<name>` / `item:<name>` (865 classes total). Bags are labelled objects.
- `geometry.py` is the single source of truth for piece placement, shared by
  the renderer and the exporter, so drawn pixels == annotated silhouette.

### Verified
- 50-image test batch: 754 instances, 0 empty scenes, ~0.8 s/image.
- All polygon coords ∈ [0,1], all class ids valid, every image ↔ label paired.
- Independent overlay (`labels/*.txt` redrawn onto the image) matches pixels.
