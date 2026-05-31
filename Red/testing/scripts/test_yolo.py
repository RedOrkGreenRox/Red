from ultralytics import YOLO
from pathlib import Path

# 1. Автоматически вычисляем корень (папку testing), отталкиваясь от скрипта
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent  # Это твоя папка C:\...\testing

# 2. Прописываем пути строго по твоим папкам
MODEL_PATH = ROOT_DIR / "models" / "last.pt"
IMAGE_PATH = ROOT_DIR / "test_object" / "image_2.png"
RESULTS_DIR = ROOT_DIR / "results"

print("🚀 Начинаем краш-тест YOLO...")
print(f"🧠 Модель: {MODEL_PATH}")
print(f"🖼 Картинка: {IMAGE_PATH}")
print(f"📁 Папка для сохранения: {RESULTS_DIR}")

# 3. Загружаем модель
model = YOLO(str(MODEL_PATH))
print(model.names)
# 4. Запускаем поиск и принудительно сохраняем в твою папку results
results = model.predict(
    source=str(IMAGE_PATH),
    conf=0.25,       # Порог уверенности
    save=True,       # Обязательно сохраняем
    show=False,      # Не пытаемся открыть окно (чтобы не повесить редактор)
    line_width=2,    # Толщина рамок
    project=str(RESULTS_DIR), # Заставляем YOLO сохранять сюда...
    name="test_run",          # ...в подпапку test_run
    exist_ok=True             # Если папка уже есть - просто сохраним туда же
)

print("\n=== ✅ ГОТОВО! ===")
print(f"Ищи разрисованный скриншот здесь: {RESULTS_DIR / 'test_run'}")