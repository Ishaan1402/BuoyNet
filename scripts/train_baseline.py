#!/usr/bin/env python3
# train_baseline.py
# Implementation of MobileNetV3-Small with Colab setup
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import pandas as pd
import os
from pathlib import Path

# Paths
DATA_DIR = Path('data')
MODELS_DIR = Path('models')
LOGS_DIR = Path('logs')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparams
NUM_CLASSES = 6
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 10

# Data pipeline
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_model():
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    # Replace head
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    return model.to(device)

def train(model, train_loader, val_loader, epochs=EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    logs = []
    best_val = 0.0

    print("Starting training loop...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Eval
        model.eval()
        val_acc, val_loss = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, labels).item()
                preds = outputs.argmax(dim=1)
                val_acc += (preds == labels).float().sum().item()

        val_acc /= len(val_loader.dataset)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        logs.append({'epoch': epoch, 'train_loss': train_loss / len(train_loader), 'val_loss': val_loss, 'val_acc': val_acc})
        print(f"Epoch {epoch}: Train Loss {train_loss:.4f} | Val Acc {val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), MODELS_DIR / 'baseline_fp32.pth')
            print(f"Saved new best model with Val Acc: {val_acc:.4f}")

    pd.DataFrame(logs).to_csv(LOGS_DIR / 'baseline_training.csv', index=False)

if __name__ == '__main__':
    train_path = DATA_DIR / 'train'
    val_path = DATA_DIR / 'val'

    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
    train_count = sum(1 for p in train_path.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)

    if train_path.exists() and train_count > 0:
        val_count = sum(1 for p in val_path.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
        
        print(f"Total valid training images detected: {train_count}")
        if train_count > 0 and val_count > 0:
            print(f"Real dataset found at {train_path}. Loading via ImageFolder...")
            from PIL import Image, UnidentifiedImageError
            def safe_pil_loader(path):
                try:
                    with open(path, 'rb') as f:
                        img = Image.open(f)
                        return img.convert('RGB')
                except (UnidentifiedImageError, OSError, IOError) as e:
                    print(f"Warning: Skipping corrupted image: {path} - {str(e)}")
                    # Return a blank block image to avoid crashing the epoch
                    return Image.new('RGB', (224, 224), (0, 0, 0))

            train_ds = ImageFolder(root=str(train_path), transform=train_transform, loader=safe_pil_loader)
            val_ds = ImageFolder(root=str(val_path), transform=val_transform, loader=safe_pil_loader)
            
            NUM_CLASSES = len(train_ds.classes)
            print(f"Classes detected ({NUM_CLASSES}): {train_ds.classes}")
        else:
            raise ValueError("Training directories found but no valid images detected! Ensure data exists.")
    else:
        raise ValueError("No real images found at path. Pipeline testing requires real data.")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = get_model()
    train(model, train_dl, val_dl)