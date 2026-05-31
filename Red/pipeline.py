"""
pipeline.py
-----------
End-to-End 2-Stage Production Inference Pipeline.

Given a full screenshot of the game (1763x790):
  1. Crop the active inventory board grid (1080x720 at x=342, y=34).
  2. Run YOLOv8m-seg (2 classes) on the grid to detect and segment bags & items.
  3. For each detected mask:
     - Nullify (black out) everything outside the instance mask to prevent background bleed.
     - Crop the instance bounding box.
     - Add square padding and resize to 224x224.
     - Run the ResNet50 Classifier to determine the exact name (1 of 865 classes).
  4. Draw colored outlines and names of exact items on the output image.
"""

from __future__ import annotations

import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------- #
#  Rarity Colors for beautiful drawing
# --------------------------------------------------------------------------- #
RARITY_COLORS = {
    "Common": (200, 200, 200, 255),       # Grey
    "Uncommon": (50, 200, 50, 255),       # Green
    "Rare": (50, 100, 255, 255),          # Blue
    "Legendary": (200, 50, 250, 255),     # Purple
    "Mythic": (250, 150, 0, 255),         # Gold/Orange
}


# --------------------------------------------------------------------------- #
#  Model Loader helpers
# --------------------------------------------------------------------------- #
def load_models(yolo_path: str, classifier_path: str, classes_txt_path: str, device: str):
    """Load both YOLOv8-seg and PyTorch ResNet50 models."""
    import torch
    import torch.nn as nn
    from torchvision import models
    from ultralytics import YOLO

    print(f"Loading YOLOv8-seg model from: {yolo_path}")
    yolo_model = YOLO(yolo_path)

    print(f"Loading class index map from: {classes_txt_path}")
    with open(classes_txt_path, "r") as fh:
        class_names = [line.strip() for line in fh if line.strip()]

    print(f"Loading ResNet50 Classifier weights from: {classifier_path}")
    classifier_model = models.resnet50(pretrained=False)
    num_ftrs = classifier_model.fc.in_features
    classifier_model.fc = nn.Linear(num_ftrs, len(class_names))
    
    # Load state dict
    state_dict = torch.load(classifier_path, map_location=device)
    classifier_model.load_state_dict(state_dict)
    classifier_model = classifier_model.to(device)
    classifier_model.eval()

    return yolo_model, classifier_model, class_names


# --------------------------------------------------------------------------- #
#  Image processing helpers
# --------------------------------------------------------------------------- #
def prepare_for_classifier(cropped_grid: Image.Image, mask_coords: np.ndarray, bbox: list[int]):
    """Nullify pixels outside the polygon mask, crop the bbox, and pad to 224x224."""
    import cv2
    
    # 1. Generate binary mask over the cropped_grid canvas size
    gw, gh = cropped_grid.size
    mask_canvas = np.zeros((gh, gw), dtype=np.uint8)
    
    # Fill poly in OpenCV
    poly_pts = mask_coords.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask_canvas, [poly_pts], 255)
    
    # 2. Nullify pixels outside the mask in the PIL image (black background)
    grid_arr = np.array(cropped_grid)
    grid_arr[mask_canvas == 0] = [0, 0, 0, 0] if grid_arr.shape[2] == 4 else [0, 0, 0]
    masked_img = Image.fromarray(grid_arr)
    
    # 3. Crop the bounding box
    x1, y1, x2, y2 = bbox
    cropped_item = masked_img.crop((x1, y1, x2, y2))
    
    # 4. Pad to a square to prevent distortion
    w, h = cropped_item.size
    max_dim = max(w, h)
    square_canvas = Image.new("RGBA", (max_dim, max_dim), (0, 0, 0, 255))
    
    px = (max_dim - w) // 2
    py = (max_dim - h) // 2
    square_canvas.paste(cropped_item, (px, py))
    
    # 5. Resize to 224x224
    final_img = square_canvas.convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
    return final_img


# --------------------------------------------------------------------------- #
#  Full pipeline inference
# --------------------------------------------------------------------------- #
def run_pipeline(img_path: str, yolo_model, classifier_model, class_names: list[str], device: str):
    """Run full two-stage inference on a full game screenshot."""
    import torch
    from torchvision import transforms

    # 1. Crop active board grid (1080x720 at x=342, y=34)
    print(f"Loading game screenshot: {img_path}")
    full_scene = Image.open(img_path).convert("RGB")
    
    crop_box = (342, 34, 342 + 1080, 34 + 720)
    grid_img = full_scene.crop(crop_box)
    grid_w, grid_h = grid_img.size
    
    # 2. Run YOLOv8m-seg (2 classes: 0=bag, 1=item)
    print("Stage 1: Running YOLOv8m-seg detection...")
    results = yolo_model(grid_img, conf=0.25, verbose=False)[0]
    
    if results.masks is None or len(results.masks) == 0:
        print("No items or bags detected by YOLOv8-seg!")
        return grid_img, []

    # Get detections
    boxes = results.boxes.xyxy.cpu().numpy()  # [N, 4] (x1, y1, x2, y2)
    classes = results.boxes.cls.cpu().numpy()  # [N]
    masks_xy = results.masks.xy  # List of polygon coordinate arrays [N, pt_count, 2]

    # Pre-processing transforms for ResNet50
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    output_canvas = grid_img.copy()
    draw = ImageDraw.Draw(output_canvas)
    
    detected_items = []
    
    print(f"Stage 2: Classifying {len(boxes)} segmented instances...")
    for idx, (bbox, cls_id, poly_xy) in enumerate(zip(boxes, classes, masks_xy)):
        if len(poly_xy) < 3:
            continue
            
        # Isolate and prepare image for PyTorch ResNet50
        x1, y1, x2, y2 = [int(round(coord)) for coord in bbox]
        item_img = prepare_for_classifier(grid_img, poly_xy, [x1, y1, x2, y2])
        
        # Classify using ResNet50
        input_tensor = preprocess(item_img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = classifier_model(input_tensor)
            pred_class_id = outputs.argmax(dim=1).item()
            predicted_class_name = class_names[pred_class_id]
            
        # Parse names: 'bag_Armor_Pack' or 'item_Adamantite_Axepick'
        prefix, clean_name = predicted_class_name.split("_", 1)
        clean_name = clean_name.replace("_", " ")
        
        print(f"  Instance {idx:2d}: Detected {prefix.upper()} -> '{clean_name}'")
        
        # Store detection info
        detected_items.append({
            "idx": idx,
            "type": prefix,
            "name": clean_name,
            "bbox": [x1, y1, x2, y2],
            "polygon": poly_xy.tolist()
        })
        
        # Draw on canvas
        # 1. Draw segment outline
        outline_color = (255, 70, 70, 255) if prefix == "item" else (0, 200, 255, 255)
        poly_pts = [(float(pt[0]), float(pt[1])) for pt in poly_xy]
        draw.line(poly_pts + [poly_pts[0]], fill=outline_color, width=2)
        
        # 2. Draw text label
        text_pos = (x1, max(0, y1 - 18))
        draw.text(text_pos, f"{clean_name}", fill=outline_color)
        
    return output_canvas, detected_items


# --------------------------------------------------------------------------- #
#  Main execution
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="Path to full game screenshot (1763x790)")
    ap.add_argument("--yolo", default="yolov8m_2classes.pt", help="Path to trained YOLOv8m-seg weight best.pt")
    ap.add_argument("--classifier", default="resnet50_classifier_best.pth", help="Path to trained ResNet50 weight")
    ap.add_argument("--class-txt", default="classifier_dataset/classes_classifier.txt", help="Path to classes_classifier.txt")
    ap.add_argument("--out", default="result_output.png", help="Path to save visual output")
    args = ap.parse_args()

    # Hardware check
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing pipeline on device: {device}")

    # Paths check
    for path in (args.img, args.yolo, args.classifier, args.class_txt):
        if not os.path.exists(path):
            print(f"Error: Missing file {path}. Please train your models first!")
            return

    # Load models
    yolo_model, classifier, class_names = load_models(args.yolo, args.classifier, args.class_txt, device)

    # Run full pipeline
    output_img, detections = run_pipeline(args.img, yolo_model, classifier, class_names, device)
    
    # Save output image
    output_img.save(args.out)
    print(f"\nSaved annotated output image to: {args.out}")
    print(f"Total objects identified: {len(detections)}")


if __name__ == "__main__":
    main()
