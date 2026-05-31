import json
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import cv2
from ultralytics import YOLO
from pathlib import Path

# --- 1. НАСТРОЙКА ПУТЕЙ ---
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent

MODEL_YOLO_PATH = ROOT_DIR / "models" / "last.pt" 
MODEL_RESNET_PATH = ROOT_DIR / "models" / "resnet50_classifier_best.pth" 
IMAGE_PATH = ROOT_DIR / "test_object" / "image_5.png"
TXT_PATH = ROOT_DIR / "classes_classifier.txt" 

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_IMAGE_PATH = RESULTS_DIR / "final_annotated.png"

print("🚀 Запускаем полный ИИ-конвейер...")

# --- 2. ЗАГРУЗКА СЛОВАРЯ КЛАССОВ ИЗ TXT ---
with open(TXT_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines() if line.strip()]
    class_mapping = {i: name for i, name in enumerate(class_names)}
    
print(f"✅ Загружено {len(class_mapping)} классов из TXT")

# --- 3. ЗАГРУЗКА МОДЕЛЕЙ ---
yolo_model = YOLO(str(MODEL_YOLO_PATH))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_model = models.resnet50(weights=None)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = torch.nn.Linear(num_ftrs, len(class_mapping))

resnet_model.load_state_dict(torch.load(str(MODEL_RESNET_PATH), map_location=device))
resnet_model.to(device)
resnet_model.eval() 
print("✅ Модели YOLO и ResNet успешно загружены!")

# --- 4. НАСТРОЙКА ЗРЕНИЯ ДЛЯ RESNET ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 5. ПОЕХАЛИ! РАБОТА С КАРТИНКОЙ ---
img_cv = cv2.imread(str(IMAGE_PATH))
yolo_results = yolo_model.predict(source=str(IMAGE_PATH), conf=0.85, verbose=False)

boxes = yolo_results[0].boxes
masks = yolo_results[0].masks.xy

print(f"\n🔍 YOLO нашла {len(boxes)} объектов. Начинаем распознавание...\n")

item_count = 0
bag_count = 0


for i, box in enumerate(boxes):
    class_id = int(box.cls[0]) 
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    polygon = np.array(masks[i], np.int32) 
    
    # === СУМКИ ===
    if class_id == 0:
        bag_count += 1
        color = (255, 144, 30) 
        
        cv2.polylines(img_cv, [polygon], isClosed=True, color=color, thickness=2)
        cv2.putText(img_cv, "Bag", (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # === ПРЕДМЕТЫ ===
    elif class_id == 1:
        item_count += 1
        
        cropped_img = img_cv[y1:y2, x1:x2]
        cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cropped_img_rgb)
        
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = resnet_model(input_tensor)
            _, predicted_idx = torch.max(outputs, 1) 
            
        final_class_id = predicted_idx.item()
        final_class_name = class_mapping[final_class_id]
        
        print(f"Предмет #{item_count}: Распознан как: {final_class_name}")
        
        color = (0, 255, 0) 
        cv2.polylines(img_cv, [polygon], isClosed=True, color=color, thickness=2)
        cv2.putText(img_cv, final_class_name, (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# --- 6. [Новое!] СОХРАНЯЕМ ИТОГОВУЮ КАРТИНКУ ---
cv2.imwrite(str(OUTPUT_IMAGE_PATH), img_cv)

print("\n=== ВЕСЬ ИНВЕНТАРЬ ПРОЧИТАН! ===")
print(f"🖼️ Картинка с разметкой сохранена сюда: {OUTPUT_IMAGE_PATH}")