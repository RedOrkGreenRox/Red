"""
train_kaggle.py
---------------
Training script designed to run in Kaggle (with GPU enabled).
It handles:
  1. Training YOLOv8m-seg on the cropped 2-class dataset.
  2. Training a ResNet50 Image Classifier on the augmented 865-class dataset.

To run this in Kaggle:
  1. Upload your generated 'dataset_cropped_2classes' and 'classifier_dataset' folders as a dataset or zip file.
  2. Copy and paste this script into a Kaggle Notebook.
  3. Turn on GPU (Tesla T4) and execute.
"""

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time

# =========================================================================== #
#  1. Train YOLOv8m-seg (General Segmentation)
# =========================================================================== #
def train_yolo_segmentation():
    print("\n" + "="*50)
    print("  STAGE 1: Training YOLOv8m-seg on 2 Classes ('bag', 'item')")
    print("="*50)
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Installing ultralytics...")
        os.system("pip install ultralytics -q")
        from ultralytics import YOLO

    # Path to your uploaded data.yaml
    # Update this path based on where Kaggle uploads your data (e.g., /kaggle/input/...)
    data_yaml_path = "/kaggle/input/your-dataset/dataset_cropped_2classes/data.yaml"
    if not os.path.exists(data_yaml_path):
        # Fallback to local path if running locally
        data_yaml_path = "./dataset_cropped_2classes/data.yaml"
        if not os.path.exists(data_yaml_path):
            print(f"Error: Could not find data.yaml. Please check the path. (Tried: {data_yaml_path})")
            return

    # Load YOLOv8m-seg pre-trained on COCO (Medium model)
    checkpoint_yolo = "yolo_backpack_hero/yolov8m_2classes/weights/last.pt"
    
    if os.path.exists(checkpoint_yolo):
        print(f"Found YOLO checkpoint at '{checkpoint_yolo}'. Resuming training...")
        model = YOLO(checkpoint_yolo)
        resume_arg = True
    else:
        print("Starting fresh YOLO training...")
        model = YOLO("yolov8m-seg.pt")
        resume_arg = False

    # Start training
    model.train(
        data=data_yaml_path,
        epochs=40,            # 40 epochs is plenty for 2 classes on a cropped grid
        imgsz=640,            # High detail on cropped 1080x720 scene
        batch=16,
        device=0,             # GPU index 0
        workers=4,
        save=True,
        resume=resume_arg,    # Automatically resumes from last.pt if True
        project="yolo_backpack_hero",
        name="yolov8m_2classes"
    )
    print("YOLOv8m-seg training completed! Model saved inside 'yolo_backpack_hero/yolov8m_2classes/weights/best.pt'.")


# =========================================================================== #
#  2. Train ResNet50 Classifier (865 Classes)
# =========================================================================== #
def train_resnet_classifier():
    print("\n" + "="*50)
    print("  STAGE 2: Training ResNet50 Classifier on 865 Classes")
    print("="*50)
    
    # Paths
    dataset_dir = "/kaggle/input/your-dataset/classifier_dataset"
    if not os.path.exists(dataset_dir):
        dataset_dir = "./classifier_dataset"
        if not os.path.exists(dataset_dir):
            print(f"Error: Could not find classifier_dataset. Please check the path. (Tried: {dataset_dir})")
            return

    # 1. Transforms with standard PyTorch Normalization
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Datasets & Loaders
    train_dataset = datasets.ImageFolder(os.path.join(dataset_dir, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(dataset_dir, "val"), transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    num_classes = len(train_dataset.classes)
    print(f"Loaded train dataset: {len(train_dataset)} images, {num_classes} classes.")
    print(f"Loaded val dataset: {len(val_dataset)} images.")

    # 3. Model Definition (ResNet50 with Pretrained weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = models.resnet50(pretrained=True)
    # Replace final fully connected layer to match our 865 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    # 4. Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    # --- Resuming Mechanism (Zero-Trust Resilience) ---
    checkpoint_path = "resnet50_classifier_checkpoint.pth"
    start_epoch = 0
    best_acc = 0.0

    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint: '{checkpoint_path}'. Resuming training...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint["best_acc"]
        # Fast-forward scheduler to match resumed epoch
        for _ in range(start_epoch):
            scheduler.step()
        print(f"Resuming from Epoch {start_epoch+1} with Best Val Acc: {best_acc:.4f}")

    # 5. Training Loop
    num_epochs = 15
    
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)
        
        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        scheduler.step()
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = val_corrects.double() / len(val_dataset)
        
        duration = time.time() - start_time
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        print(f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f} ({duration:.1f}s)")
        
        # Save checkpoint at the end of EVERY epoch (for crash recovery)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc
        }, checkpoint_path)
        print(f"Saved periodic epoch checkpoint to: {checkpoint_path}")

        # Save best model
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            torch.save(model.state_dict(), "resnet50_classifier_best.pth")
            print("=> Saved new best classifier model weight: resnet50_classifier_best.pth")
            
    print("\nResNet50 training completed!")
    print(f"Best Validation Accuracy: {best_acc:.4f}")


# =========================================================================== #
#  Main Execution
# =========================================================================== #
if __name__ == "__main__":
    train_yolo_segmentation()
    train_resnet_classifier()
